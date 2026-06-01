from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks