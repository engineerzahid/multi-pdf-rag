# Multi-PDF RAG Research Assistant

Ask questions across multiple PDFs and get cited, accurate answers powered by LangChain + FAISS + Groq.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Add your API keys
python main.py                  # Test Day 1 setup
```

## Run

```bash
streamlit run app/ui.py
```

## Stack
- LangChain — RAG orchestration
- FAISS — vector similarity search
- HuggingFace — embeddings (all-MiniLM-L6-v2)
- Groq — LLM inference (llama-3.1-8b-instant)
- Streamlit — UI
