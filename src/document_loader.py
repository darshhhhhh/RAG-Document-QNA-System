from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.logger import get_logger


logger = get_logger(__name__)


class DocumentLoader:
    """
    Loads PDF documents page by page and returns structured text with metadata.
    """

    def load_pdf(self, file_path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        logger.info(f"Loading PDF: {file_path.name}")

        reader = PdfReader(str(file_path))
        pages = []

        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            page_data = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "page_number": page_index,
                "text": text.strip(),
                "char_count": len(text.strip())
            }

            pages.append(page_data)

        logger.info(f"Loaded {len(pages)} pages from {file_path.name}")

        return pages