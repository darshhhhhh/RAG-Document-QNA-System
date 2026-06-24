from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
UPLOADED_DOCS_DIR = DATA_DIR / "uploaded_docs"

VECTOR_DB_DIR = BASE_DIR / "vector_db"

COLLECTION_NAME = "rag_document_qa_collection"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_TOP_K = 8
RERANK_TOP_N = 3