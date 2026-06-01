# src/vector_store.py

from langchain_community.vectorstores import FAISS

def create_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(  # ✅ from_texts → from_documents
        documents=text_chunks,
        embedding=embeddings
    )
    return vector_store