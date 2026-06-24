import sys
import shutil
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import UPLOADED_DOCS_DIR
from src.ingestion_pipeline import DocumentIngestionPipeline
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📄",
    layout="wide"
)


def save_uploaded_file(uploaded_file) -> Path:
    UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    file_path = UPLOADED_DOCS_DIR / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def reset_chat():
    st.session_state.messages = []


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

    if "processing_summary" not in st.session_state:
        st.session_state.processing_summary = None


initialize_session_state()


st.title("📄 RAG-Based Document Q&A System")
st.write(
    "Upload a PDF document, process it into vector embeddings, "
    "and ask source-grounded questions from the document."
)

st.sidebar.header("Document Processing")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

process_button = st.sidebar.button("Process Document")

if process_button:
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF file first.")
    else:
        with st.spinner("Processing document..."):
            try:
                pdf_path = save_uploaded_file(uploaded_file)

                ingestion_pipeline = DocumentIngestionPipeline()

                summary = ingestion_pipeline.ingest_pdf(
                    pdf_path=pdf_path,
                    reset_collection=True
                )

                st.session_state.document_processed = True
                st.session_state.processing_summary = summary
                st.session_state.messages = []

                st.sidebar.success("Document processed successfully.")

            except Exception as error:
                st.sidebar.error(f"Error: {error}")


if st.sidebar.button("Reset Chat"):
    reset_chat()
    st.sidebar.success("Chat reset successfully.")


st.sidebar.header("Processing Summary")

if st.session_state.processing_summary:
    summary = st.session_state.processing_summary

    st.sidebar.write(f"**File:** {summary['file_name']}")
    st.sidebar.write(f"**Pages:** {summary['total_pages']}")
    st.sidebar.write(f"**Chunks:** {summary['total_chunks']}")
    st.sidebar.write(f"**Embedding Model:** {summary['embedding_model']}")
    st.sidebar.write(f"**Embedding Dimension:** {summary['embedding_dimension']}")
    st.sidebar.write(f"**Vector DB Count:** {summary['vector_db_count']}")
else:
    st.sidebar.info("No document processed yet.")


vector_store = VectorStore()

if vector_store.count() > 0:
    st.session_state.document_processed = True


if not st.session_state.document_processed:
    st.info("Upload and process a PDF document to start asking questions.")
else:
    st.success("Document is ready. You can ask questions now.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]

                st.write(f"**Confidence:** {metadata['confidence']}")
                st.write(f"**Best Score:** {metadata['best_score']}")
                st.write(f"**Answer Type:** {metadata['answer_type']}")

                if metadata["sources"]:
                    st.write("**Sources:**")
                    for index, source in enumerate(metadata["sources"], start=1):
                        st.write(
                            f"{index}. `{source['file_name']}` | "
                            f"Page {source['page_number']} | "
                            f"Chunk {source['chunk_number']} | "
                            f"Score {source['score']}"
                        )

                with st.expander("View Retrieved Context"):
                    for index, chunk in enumerate(metadata["retrieved_context"], start=1):
                        chunk_metadata = chunk["metadata"]

                        st.markdown(
                            f"**Context {index}:** "
                            f"{chunk_metadata['file_name']} | "
                            f"Page {chunk_metadata['page_number']} | "
                            f"Chunk {chunk_metadata['chunk_number']}"
                        )

                        st.write(chunk["text"])
                        st.divider()

    user_question = st.chat_input("Ask a question from your document...")

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Generating grounded answer..."):
                try:
                    rag_pipeline = RAGPipeline()
                    response = rag_pipeline.ask(user_question)

                    st.write(response["answer"])

                    st.write(f"**Confidence:** {response['confidence']}")
                    st.write(f"**Best Score:** {response['best_score']}")
                    st.write(f"**Answer Type:** {response['answer_type']}")

                    if response["sources"]:
                        st.write("**Sources:**")
                        for index, source in enumerate(response["sources"], start=1):
                            st.write(
                                f"{index}. `{source['file_name']}` | "
                                f"Page {source['page_number']} | "
                                f"Chunk {source['chunk_number']} | "
                                f"Score {source['score']}"
                            )
                    else:
                        st.write("**Sources:** No strong source found.")

                    with st.expander("View Retrieved Context"):
                        for index, chunk in enumerate(
                            response["retrieved_context"],
                            start=1
                        ):
                            chunk_metadata = chunk["metadata"]

                            st.markdown(
                                f"**Context {index}:** "
                                f"{chunk_metadata['file_name']} | "
                                f"Page {chunk_metadata['page_number']} | "
                                f"Chunk {chunk_metadata['chunk_number']}"
                            )

                            st.write(chunk["text"])
                            st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "metadata": {
                                "confidence": response["confidence"],
                                "best_score": response["best_score"],
                                "answer_type": response["answer_type"],
                                "sources": response["sources"],
                                "retrieved_context": response["retrieved_context"]
                            }
                        }
                    )

                except Exception as error:
                    st.error(f"Error: {error}")