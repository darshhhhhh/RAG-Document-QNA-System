import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import SAMPLE_DOCS_DIR
from src.document_loader import DocumentLoader


def main():
    pdf_path = SAMPLE_DOCS_DIR / "sample.pdf"

    loader = DocumentLoader()
    pages = loader.load_pdf(pdf_path)

    print("\nPDF LOADING SUMMARY")
    print("=" * 60)
    print(f"File Name     : {pages[0]['file_name']}")
    print(f"Total Pages   : {len(pages)}")
    print(f"Total Chars   : {sum(page['char_count'] for page in pages)}")
    print("=" * 60)

    print("\nFIRST PAGE PREVIEW")
    print("=" * 60)
    print(f"Page Number    : {pages[0]['page_number']}")
    print(f"Character Count: {pages[0]['char_count']}")
    print("-" * 60)
    print(pages[0]["text"][:1000])
    print("=" * 60)


if __name__ == "__main__":
    main()