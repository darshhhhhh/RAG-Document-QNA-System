import re


class TextCleaner:
    """
    Cleans extracted PDF text before chunking.
    """

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self._fix_common_pdf_artifacts(text)
        text = self._remove_extra_whitespace(text)
        text = self._remove_repeated_newlines(text)
        text = self._normalize_spacing(text)

        return text.strip()

    def _fix_common_pdf_artifacts(self, text: str) -> str:
        """
        Fixes common PDF extraction issues.
        """
        text = text.replace("Q&A;", "Q&A")
        text = text.replace("Document Q&A;", "Document Q&A")
        text = text.replace("Retrieval-Augmented", "Retrieval-Augmented")
        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        """
        Converts tabs and multiple spaces into single spaces.
        """
        text = re.sub(r"[ \t]+", " ", text)
        return text

    def _remove_repeated_newlines(self, text: str) -> str:
        """
        Reduces multiple new lines.
        """
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text

    def _normalize_spacing(self, text: str) -> str:
        """
        Removes unnecessary spaces around punctuation.
        """
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)
        return text