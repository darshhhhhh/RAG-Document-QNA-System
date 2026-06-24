from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME
from src.logger import get_logger


logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates dense vector embeddings for document chunks and user queries.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dimension = self.model.get_embedding_dimension()
        else:
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Embedding model loaded successfully. "
            f"Dimension: {self.embedding_dimension}"
        )

    def embed_texts(
        self,
        texts: Iterable[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Converts a list of text chunks into embeddings.
        """
        texts = list(texts)

        if not texts:
            raise ValueError("texts cannot be empty.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True
        )

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Converts a user query into an embedding.
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings
        )

        return np.array(embedding)