from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

def get_conversation_chain(vector_store):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, streaming=True)
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, say:
    "Answer is not available in the provided documents."
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 4}), combine_docs_chain_kwargs={"prompt": prompt}, return_source_documents=True, verbose=False)
    return chain
