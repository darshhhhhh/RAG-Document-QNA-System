import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import SAMPLE_DOCS_DIR
from src.document_loader import DocumentLoader
from src.text_splitter import MetadataAwareTextSplitter
from src.embeddings import EmbeddingGenerator


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

    chunk_texts = [chunk["text"] for chunk in chunks]

    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.embed_texts(chunk_texts)

    sample_query = "What is the purpose of this RAG system?"
    query_embedding = embedding_generator.embed_query(sample_query)

    print("\nEMBEDDING GENERATION SUMMARY")
    print("=" * 70)
    print(f"Embedding Model     : {embedding_generator.model_name}")
    print(f"Total Chunks        : {len(chunks)}")
    print(f"Embeddings Shape    : {embeddings.shape}")
    print(f"Embedding Dimension : {embedding_generator.embedding_dimension}")
    print(f"Query Shape         : {query_embedding.shape}")
    print("=" * 70)

    print("\nFIRST CHUNK DETAILS")
    print("=" * 70)
    print(f"Chunk ID     : {chunks[0]['chunk_id']}")
    print(f"Page Number  : {chunks[0]['metadata']['page_number']}")
    print(f"Word Count   : {chunks[0]['metadata']['word_count']}")
    print("-" * 70)
    print(chunks[0]["text"][:500])
    print("=" * 70)

    print("\nFIRST EMBEDDING VECTOR PREVIEW")
    print("=" * 70)
    print(embeddings[0][:10])
    print("=" * 70)

    print("\nQUERY EMBEDDING PREVIEW")
    print("=" * 70)
    print(f"Query: {sample_query}")
    print(query_embedding[:10])
    print("=" * 70)


if __name__ == "__main__":
    main()