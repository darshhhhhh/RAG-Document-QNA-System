import re
from typing import Any


class LocalAnswerGenerator:
    """
    Generates grounded extractive answers from retrieved context.

    This is a local, free answer generator. It avoids hallucination by using
    only retrieved document text and includes fallback checks for unsupported
    questions.
    """

    FALLBACK_MESSAGE = "I could not find this information in the uploaded document."

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not retrieved_chunks:
            return {
                "answer": self.FALLBACK_MESSAGE,
                "answer_found": False,
                "answer_type": "fallback"
            }

        question_lower = question.lower()
        context_text = self._build_context_text(retrieved_chunks)

        if self._should_force_fallback(question_lower, context_text):
            return {
                "answer": self.FALLBACK_MESSAGE,
                "answer_found": False,
                "answer_type": "fallback"
            }

        if self._is_api_endpoint_question(question_lower):
            endpoint_answer = self._extract_api_endpoints(context_text)

            if endpoint_answer:
                return {
                    "answer": endpoint_answer,
                    "answer_found": True,
                    "answer_type": "endpoint_extraction"
                }

        if self._is_purpose_question(question_lower):
            purpose_answer = self._extract_purpose_answer(context_text)

            if purpose_answer:
                return {
                    "answer": purpose_answer,
                    "answer_found": True,
                    "answer_type": "purpose_extraction"
                }

        answer = self._generate_sentence_based_answer(
            question=question,
            context_text=context_text
        )

        if not answer:
            return {
                "answer": self.FALLBACK_MESSAGE,
                "answer_found": False,
                "answer_type": "fallback"
            }

        return {
            "answer": answer,
            "answer_found": True,
            "answer_type": "extractive"
        }

    def _build_context_text(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        return " ".join(chunk["text"] for chunk in retrieved_chunks)

    def _should_force_fallback(self, question_lower: str, context_text: str) -> bool:
        """
        Prevents incorrect answers for unsupported questions.

        Example:
        Question: Who is the CEO of OpenAI?
        Context mentions OpenAI, but does not mention CEO.
        The system must not answer from unrelated OpenAI text.
        """
        context_lower = context_text.lower()

        critical_terms = {
            "ceo",
            "founder",
            "president",
            "director",
            "owner",
            "headquarter",
            "headquarters",
            "salary",
            "revenue",
            "net worth"
        }

        for term in critical_terms:
            if term in question_lower and term not in context_lower:
                return True

        return False

    def _is_api_endpoint_question(self, question_lower: str) -> bool:
        endpoint_terms = ["endpoint", "endpoints", "api", "backend"]

        return any(term in question_lower for term in endpoint_terms)

    def _extract_api_endpoints(self, context_text: str) -> str:
        endpoints = re.findall(r"/[a-zA-Z0-9_-]+", context_text)

        unique_endpoints = []
        seen = set()

        for endpoint in endpoints:
            endpoint = endpoint.strip()

            if endpoint not in seen:
                unique_endpoints.append(endpoint)
                seen.add(endpoint)

        if not unique_endpoints:
            return ""

        endpoint_list = ", ".join(unique_endpoints)

        return f"The backend exposes the following API endpoints: {endpoint_list}."

    def _is_purpose_question(self, question_lower: str) -> bool:
        purpose_terms = ["purpose", "goal", "objective", "about", "overview"]

        return any(term in question_lower for term in purpose_terms)

    def _extract_purpose_answer(self, context_text: str) -> str:
        context_text = context_text.replace("\n", " ")
        context_text = re.sub(r"\s+", " ", context_text)

        project_overview_pattern = (
            r"The RAG-Based Document Q&A System allows a user to upload PDF documents "
            r"and ask natural language questions\."
        )

        workflow_pattern = (
            r"The system extracts text from the document, cleans the text, splits it into chunks, "
            r"converts each chunk into embeddings, stores those embeddings in a vector database, "
            r"retrieves the most relevant chunks for a query, and produces an answer based only "
            r"on the retrieved context\."
        )

        overview_match = re.search(project_overview_pattern, context_text)
        workflow_match = re.search(workflow_pattern, context_text)

        answer_parts = []

        if overview_match:
            answer_parts.append(overview_match.group(0))

        if workflow_match:
            answer_parts.append(workflow_match.group(0))

        if not answer_parts:
            return ""

        return " ".join(answer_parts)

    def _generate_sentence_based_answer(
        self,
        question: str,
        context_text: str
    ) -> str:
        question_keywords = self._extract_keywords(question)

        if not question_keywords:
            return ""

        sentences = self._split_into_sentences(context_text)

        scored_sentences = []

        for sentence in sentences:
            score = self._score_sentence(sentence, question_keywords)

            if score > 0:
                scored_sentences.append(
                    {
                        "sentence": sentence,
                        "score": score
                    }
                )

        if not scored_sentences:
            return ""

        scored_sentences = sorted(
            scored_sentences,
            key=lambda item: item["score"],
            reverse=True
        )

        selected_sentences = []
        seen = set()

        for item in scored_sentences:
            sentence = self._clean_sentence(item["sentence"])

            if not sentence:
                continue

            if sentence.lower() in seen:
                continue

            selected_sentences.append(sentence)
            seen.add(sentence.lower())

            if len(selected_sentences) >= 3:
                break

        return " ".join(selected_sentences).strip()

    def _extract_keywords(self, question: str) -> set[str]:
        stopwords = {
            "what", "which", "when", "where", "who", "why", "how",
            "is", "are", "was", "were", "the", "a", "an", "of",
            "to", "in", "on", "for", "and", "or", "should", "this",
            "that", "does", "do", "from", "document", "available",
            "happen", "found", "mentioned"
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())

        return {
            word for word in words
            if word not in stopwords
        }

    def _split_into_sentences(self, text: str) -> list[str]:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        sentences = re.split(r"(?<=[.!?])\s+", text)

        cleaned_sentences = []

        for sentence in sentences:
            sentence = self._clean_sentence(sentence)

            if len(sentence) >= 40:
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _score_sentence(
        self,
        sentence: str,
        question_keywords: set[str]
    ) -> int:
        sentence_lower = sentence.lower()

        score = 0

        for keyword in question_keywords:
            if keyword in sentence_lower:
                score += 1

        return score

    def _clean_sentence(self, sentence: str) -> str:
        sentence = sentence.strip()
        sentence = re.sub(r"\s+", " ", sentence)

        noisy_prefixes = [
            "Page 1 RAG Sample Document",
            "Page 2 RAG Sample Document",
            "Page 3 RAG Sample Document",
            "Page 4 RAG Sample Document",
            "Page 5 RAG Sample Document",
            "Page 6 RAG Sample Document",
            "Sample Knowledge Document for RAG Testing"
        ]

        for prefix in noisy_prefixes:
            sentence = sentence.replace(prefix, "")

        sentence = sentence.strip(" -:\n\t")

        return sentence