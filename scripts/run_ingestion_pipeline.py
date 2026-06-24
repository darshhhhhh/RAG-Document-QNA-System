import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import SAMPLE_DOCS_DIR
from src.ingestion_pipeline import DocumentIngestionPipeline


def main():
    pdf_path = SAMPLE_DOCS_DIR / "sample.pdf"

    pipeline = DocumentIngestionPipeline()

    summary = pipeline.ingest_pdf(
        pdf_path=pdf_path,
        reset_collection=True
    )

    print("\nDOCUMENT INGESTION SUMMARY")
    print("=" * 80)
    print(f"Status              : {summary['status']}")
    print(f"Message             : {summary['message']}")
    print(f"File Name           : {summary['file_name']}")
    print(f"Total Pages         : {summary['total_pages']}")
    print(f"Total Chunks        : {summary['total_chunks']}")
    print(f"Inserted Chunks     : {summary['inserted_chunks']}")
    print(f"Embedding Model     : {summary['embedding_model']}")
    print(f"Embedding Dimension : {summary['embedding_dimension']}")
    print(f"Vector DB Count     : {summary['vector_db_count']}")
    print("=" * 80)


if __name__ == "__main__":
    main()