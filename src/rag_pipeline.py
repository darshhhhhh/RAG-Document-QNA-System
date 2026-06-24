from typing import Any

from src.config import RERANK_TOP_N
from src.llm import LocalAnswerGenerator
from src.prompt_builder import PromptBuilder
from src.retriever import Retriever
from src.logger import get_logger


logger = get_logger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline:
    query -> retrieve -> rerank -> answer -> confidence -> sources
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        answer_generator: LocalAnswerGenerator | None = None,
        prompt_builder: PromptBuilder | None = None
    ):
        self.retriever = retriever or Retriever()
        self.answer_generator = answer_generator or LocalAnswerGenerator()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def ask(self, question: str) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("question cannot be empty.")

        logger.info(f"Running RAG pipeline for question: {question}")

        retrieved_chunks = self.retriever.retrieve(question)

        if not retrieved_chunks:
            return self._fallback_response(question)

        answer_result = self.answer_generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        if not answer_result["answer_found"]:
            return self._fallback_response(question)

        best_score = retrieved_chunks[0].get("normalized_rerank_score", 0)

        prompt = self.prompt_builder.build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        confidence = self._calculate_confidence(
            best_score=best_score,
            answer_type=answer_result["answer_type"]
        )

        sources = self._build_sources(retrieved_chunks)

        return {
            "question": question,
            "answer": answer_result["answer"],
            "confidence": confidence,
            "best_score": round(best_score, 4),
            "answer_type": answer_result["answer_type"],
            "sources": sources,
            "retrieved_context": retrieved_chunks,
            "prompt": prompt
        }

    def _calculate_confidence(
        self,
        best_score: float,
        answer_type: str
    ) -> str:
        if answer_type == "fallback":
            return "Low"

        if answer_type == "endpoint_extraction":
            return "High"

        if best_score >= 0.75:
            return "High"

        if best_score >= 0.55:
            return "Medium"

        return "Low"

    def _build_sources(
        self,
        retrieved_chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        sources = []

        for chunk in retrieved_chunks[:RERANK_TOP_N]:
            metadata = chunk["metadata"]

            score = chunk.get("normalized_rerank_score", 0)

            source = {
                "file_name": metadata["file_name"],
                "page_number": metadata["page_number"],
                "chunk_number": metadata["chunk_number"],
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 4)
            }

            sources.append(source)

        return sources

    def _fallback_response(self, question: str) -> dict[str, Any]:
        return {
            "question": question,
            "answer": LocalAnswerGenerator.FALLBACK_MESSAGE,
            "confidence": "Low",
            "best_score": 0,
            "answer_type": "fallback",
            "sources": [],
            "retrieved_context": [],
            "prompt": ""
        }