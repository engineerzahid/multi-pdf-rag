import streamlit as st
from dotenv import load_dotenv
from src.pdf_handler import extract_text_from_pdfs
from src.text_splitter import get_text_chunks
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.chatbot import get_conversation_chain

load_dotenv()

st.set_page_config(page_title="Multi PDF RAG")
st.title("Multi PDF RAG Chatbot")

# ── Session state initialize ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Your PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    process_btn = st.button("Process PDFs")

# ── Process PDFs ──────────────────────────────────────────────────────────────
if process_btn and uploaded_files:
    with st.spinner("Processing PDFs..."):
        st.write("Step 1: Extracting text...")
        raw_text = extract_text_from_pdfs(uploaded_files)
        st.write(f"Done — {len(raw_text)} characters extracted")

        st.write("Step 2: Splitting into chunks...")
        text_chunks = get_text_chunks(raw_text)
        st.write(f"Done — {len(text_chunks)} chunks created")

        st.write("Step 3: Loading embedding model...")
        embeddings = get_embedding_model()
        st.write("Done — model loaded")

        st.write("Step 4: Building vector store...")
        vector_store = create_vector_store(text_chunks, embeddings)
        st.write("Done — vector store ready")

        st.session_state.vector_store = vector_store
        st.session_state.chain = get_conversation_chain(vector_store)
        st.session_state.chat_history = []

    st.sidebar.success("✅ Ready! Ab sawal poocho.")

# ── Chat UI ───────────────────────────────────────────────────────────────────
if st.session_state.vector_store is not None:
    st.write("---")

    user_question = st.chat_input("Ask a question about your PDFs...")

    if user_question:
        # Step 1: User message history mein daalo
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        # Step 2: Purani history display karo
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Step 3: Streaming response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            last_chunk  = None

            for chunk in st.session_state.chain.stream(
                {"question": user_question}
            ):
                token = chunk.get("answer", "")
                full_answer += token
                placeholder.write(full_answer + "▌")
                last_chunk = chunk

            # Final answer — cursor hatao
            placeholder.write(full_answer)

            # Step 4: Sources — loop ke baad
            if last_chunk:
                source_docs = last_chunk.get("source_documents", [])
                if source_docs:
                    with st.expander("📄 Sources"):
                        sources_seen = set()
                        for doc in source_docs:
                            source = doc.metadata.get("source", "Unknown")
                            page   = doc.metadata.get("page", "?")
                            key    = f"{source}_p{page}"
                            if key not in sources_seen:
                                sources_seen.add(key)
                                try:
                                    page_num = int(page) + 1
                                except (ValueError, TypeError):
                                    page_num = page
                                st.write(f"**{source}** — Page {page_num}")

        # Step 5: Full answer history mein save karo
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_answer}
        )

    else:
        # Naya question nahi — sirf history dikhao
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

else:
    st.info("⬅️ Upload PDFs from the sidebar and click Process PDFs to begin.")