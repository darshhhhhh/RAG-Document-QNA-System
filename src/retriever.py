from typing import Any

from src.config import RETRIEVAL_TOP_K, RERANK_TOP_N
from src.embeddings import EmbeddingGenerator
from src.reranker import CrossEncoderReranker
from src.vector_store import VectorStore
from src.logger import get_logger


logger = get_logger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks using vector search and re-ranking.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: VectorStore | None = None,
        reranker: CrossEncoderReranker | None = None,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        rerank_top_n: int = RERANK_TOP_N
    ):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker or CrossEncoderReranker()

        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        logger.info(f"Retrieving chunks for query: {query}")

        query_embedding = self.embedding_generator.embed_query(query)

        initial_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.retrieval_top_k
        )

        logger.info(f"Initial retrieved chunks: {len(initial_results)}")

        reranked_results = self.reranker.rerank(
            query=query,
            retrieved_chunks=initial_results,
            top_n=self.rerank_top_n
        )

        logger.info(f"Final reranked chunks: {len(reranked_results)}")

        return reranked_results