from pathlib import Path
from typing import Any

from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingGenerator
from src.text_splitter import MetadataAwareTextSplitter
from src.vector_store import VectorStore
from src.logger import get_logger


logger = get_logger(__name__)


class DocumentIngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    This pipeline converts a PDF document into searchable vector embeddings
    and stores them in ChromaDB with page-level and chunk-level metadata.
    """

    def __init__(
        self,
        document_loader: DocumentLoader | None = None,
        text_splitter: MetadataAwareTextSplitter | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: VectorStore | None = None
    ):
        self.document_loader = document_loader or DocumentLoader()

        self.text_splitter = text_splitter or MetadataAwareTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            min_chunk_size=250
        )

        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        reset_collection: bool = True
    ) -> dict[str, Any]:
        pdf_path = Path(pdf_path)

        logger.info(f"Starting ingestion pipeline for: {pdf_path.name}")

        pages = self.document_loader.load_pdf(pdf_path)

        chunks = self.text_splitter.split_pages(pages)

        if not chunks:
            raise ValueError("No valid text chunks were created from the PDF.")

        chunk_texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_generator.embed_texts(chunk_texts)

        if reset_collection:
            self.vector_store.reset_collection()

        inserted_count = self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings
        )

        summary = {
            "status": "success",
            "message": "Document processed successfully.",
            "file_name": pdf_path.name,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "inserted_chunks": inserted_count,
            "embedding_model": self.embedding_generator.model_name,
            "embedding_dimension": self.embedding_generator.embedding_dimension,
            "vector_db_count": self.vector_store.count()
        }

        logger.info(
            f"Ingestion completed for {pdf_path.name}. "
            f"Chunks inserted: {inserted_count}"
        )

        return summary