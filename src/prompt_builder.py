from typing import Any


class PromptBuilder:
    """
    Builds a grounded RAG prompt using retrieved document chunks.
    """

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]]
    ) -> str:
        if not question or not question.strip():
            raise ValueError("question cannot be empty.")

        if not retrieved_chunks:
            return ""

        context_blocks = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk["metadata"]

            context_block = f"""
Source {index}
File: {metadata["file_name"]}
Page: {metadata["page_number"]}
Chunk: {metadata["chunk_number"]}
Text:
{chunk["text"]}
""".strip()

            context_blocks.append(context_block)

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the context provided below.
Do not use outside knowledge.
If the answer is not present in the context, say:
"I could not find this information in the uploaded document."

Question:
{question}

Context:
{context}

Answer:
""".strip()

        return prompt