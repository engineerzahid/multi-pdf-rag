from pypdf import PdfReader
from langchain_core.documents import Document
import io

def extract_text_from_pdfs(files):
    documents = []
    for pdf in files:
        pdf.seek(0)
        reader = PdfReader(io.BytesIO(pdf.read()))
        filename = pdf.name

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num
                        }
                    )
                )
    return documents