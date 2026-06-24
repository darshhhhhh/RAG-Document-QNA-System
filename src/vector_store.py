from typing import Any

import chromadb

from src.config import COLLECTION_NAME, VECTOR_DB_DIR
from src.logger import get_logger


logger = get_logger(__name__)


class VectorStore:
    """
    Handles ChromaDB operations for storing and retrieving document chunks.
    """

    def __init__(
        self,
        persist_directory: str = str(VECTOR_DB_DIR),
        collection_name: str = COLLECTION_NAME
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Connected to ChromaDB collection: {self.collection_name}")

    def reset_collection(self) -> None:
        """
        Deletes and recreates the collection.
        Useful during development to avoid duplicate chunk IDs.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            logger.info("No existing collection found to delete.")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Created fresh collection: {self.collection_name}")

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings
    ) -> int:
        """
        Adds chunks, embeddings, and metadata to ChromaDB.
        """
        if not chunks:
            raise ValueError("chunks cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        embeddings_list = embeddings.tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

        logger.info(f"Added {len(chunks)} chunks to ChromaDB.")

        return len(chunks)

    def search(
        self,
        query_embedding,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Searches ChromaDB and returns the most relevant chunks.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances
        ):
            relevance_score = max(0, min(1, 1 - distance))

            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": round(distance, 4),
                    "relevance_score": round(relevance_score, 4)
                }
            )

        return retrieved_chunks

    def count(self) -> int:
        """
        Returns the number of stored chunks.
        """
        return self.collection.count()