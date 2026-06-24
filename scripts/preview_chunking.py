import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import SAMPLE_DOCS_DIR
from src.document_loader import DocumentLoader
from src.text_splitter import MetadataAwareTextSplitter


def main():
    pdf_path = SAMPLE_DOCS_DIR / "sample.pdf"

    loader = DocumentLoader()
    pages = loader.load_pdf(pdf_path)

    splitter = MetadataAwareTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        min_chunk_size=250
    )

    chunks = splitter.split_pages(pages)

    print("\nCHUNKING SUMMARY")
    print("=" * 70)
    print(f"File Name        : {pages[0]['file_name']}")
    print(f"Total Pages      : {len(pages)}")
    print(f"Total Chunks     : {len(chunks)}")
    print(f"Chunk Size       : {splitter.chunk_size}")
    print(f"Chunk Overlap    : {splitter.chunk_overlap}")
    print(f"Min Chunk Size   : {splitter.min_chunk_size}")
    print("=" * 70)

    print("\nFIRST CHUNK PREVIEW")
    print("=" * 70)
    print(f"Chunk ID     : {chunks[0]['chunk_id']}")
    print(f"File Name    : {chunks[0]['metadata']['file_name']}")
    print(f"Page Number  : {chunks[0]['metadata']['page_number']}")
    print(f"Chunk Number : {chunks[0]['metadata']['chunk_number']}")
    print(f"Char Count   : {chunks[0]['metadata']['char_count']}")
    print(f"Word Count   : {chunks[0]['metadata']['word_count']}")
    print("-" * 70)
    print(chunks[0]["text"][:1000])
    print("=" * 70)

    print("\nALL CHUNKS")
    print("=" * 70)

    for chunk in chunks:
        metadata = chunk["metadata"]
        print(
            f"{chunk['chunk_id']} | "
            f"Page: {metadata['page_number']} | "
            f"Chars: {metadata['char_count']} | "
            f"Words: {metadata['word_count']}"
        )


if __name__ == "__main__":
    main()