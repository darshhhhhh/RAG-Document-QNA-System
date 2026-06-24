from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL_NAME
from src.logger import get_logger


logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a Cross-Encoder model.

    Vector search is fast but sometimes ranks loosely related chunks highly.
    Cross-Encoder re-ranking improves final context quality by scoring
    the question and chunk together.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name

        logger.info(f"Loading re-ranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("Re-ranker model loaded successfully.")

    def rerank(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        top_n: int = 3
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        if not retrieved_chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]

        raw_scores = self.model.predict(pairs)
        raw_scores = np.array(raw_scores, dtype=float)

        normalized_scores = self._normalize_scores(raw_scores)

        reranked_chunks = []

        for chunk, raw_score, normalized_score in zip(
            retrieved_chunks,
            raw_scores,
            normalized_scores
        ):
            updated_chunk = chunk.copy()
            updated_chunk["rerank_score"] = round(float(raw_score), 4)
            updated_chunk["normalized_rerank_score"] = round(
                float(normalized_score),
                4
            )

            reranked_chunks.append(updated_chunk)

        reranked_chunks = sorted(
            reranked_chunks,
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_chunks[:top_n]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)