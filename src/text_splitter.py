from typing import Any

from src.text_cleaner import TextCleaner


class MetadataAwareTextSplitter:
    """
    Splits page-wise document text into overlapping chunks while preserving metadata.

    This splitter avoids creating very small leftover chunks because tiny chunks
    reduce retrieval quality in RAG systems.
    """

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        min_chunk_size: int = 250
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if min_chunk_size >= chunk_size:
            raise ValueError("min_chunk_size must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.cleaner = TextCleaner()

    def split_pages(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_chunks = []

        for page in pages:
            cleaned_text = self.cleaner.clean(page["text"])

            if not cleaned_text:
                continue

            page_chunks = self._split_single_page(
                text=cleaned_text,
                file_name=page["file_name"],
                page_number=page["page_number"]
            )

            all_chunks.extend(page_chunks)

        return all_chunks

    def _split_single_page(
        self,
        text: str,
        file_name: str,
        page_number: int
    ) -> list[dict[str, Any]]:
        chunks = []
        start = 0
        chunk_number = 1

        while start < len(text):
            remaining_chars = len(text) - start

            if chunks and remaining_chars < self.min_chunk_size:
                remaining_text = text[start:].strip()

                if remaining_text:
                    chunks[-1]["text"] = (
                        chunks[-1]["text"] + " " + remaining_text
                    ).strip()

                    chunks[-1]["metadata"]["char_count"] = len(chunks[-1]["text"])
                    chunks[-1]["metadata"]["word_count"] = len(
                        chunks[-1]["text"].split()
                    )

                break

            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = self._create_chunk_id(
                    file_name=file_name,
                    page_number=page_number,
                    chunk_number=chunk_number
                )

                chunk = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "file_name": file_name,
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    }
                }

                chunks.append(chunk)
                chunk_number += 1

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _create_chunk_id(
        self,
        file_name: str,
        page_number: int,
        chunk_number: int
    ) -> str:
        clean_file_name = file_name.replace(".pdf", "").replace(" ", "_").lower()
        return f"{clean_file_name}_page_{page_number}_chunk_{chunk_number}"