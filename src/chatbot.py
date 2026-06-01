# src/chatbot.py

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_conversation_chain(vector_store):

    # ✅ streaming=True add kiya
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        streaming=True  # ← Yahi enable karta hai streaming
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt_template = """
         You are a helpful assistant. Answer ONLY based on the provided context.
         Do NOT mix information from different documents unless the question specifically asks to compare them.
         If the answer is not available in the context, say:
         "Answer is not available in the provided documents."

         Context:
         {context}

         Chat History: 
         {chat_history}

         Question:     
         {question}

         Answer:
         """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain