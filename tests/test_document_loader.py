from src.config import SAMPLE_DOCS_DIR
from src.document_loader import DocumentLoader


def test_pdf_loader_loads_pages():
    pdf_path = SAMPLE_DOCS_DIR / "sample.pdf"

    loader = DocumentLoader()
    pages = loader.load_pdf(pdf_path)

    assert isinstance(pages, list)
    assert len(pages) > 0
    assert "file_name" in pages[0]
    assert "page_number" in pages[0]
    assert "text" in pages[0]
    assert "char_count" in pages[0]
    assert pages[0]["file_name"] == "sample.pdf"
    assert pages[0]["page_number"] == 1
    assert pages[0]["char_count"] > 0