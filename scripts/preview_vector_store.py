import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import SAMPLE_DOCS_DIR
from src.document_loader import DocumentLoader
from src.text_splitter import MetadataAwareTextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


def main():
    pdf_path = SAMPLE_DOCS_DIR / "sample.pdf"

    print("\nSTEP 1: Loading document")
    print("=" * 70)
    loader = DocumentLoader()
    pages = loader.load_pdf(pdf_path)

    print("\nSTEP 2: Creating chunks")
    print("=" * 70)
    splitter = MetadataAwareTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        min_chunk_size=250
    )
    chunks = splitter.split_pages(pages)

    print(f"Total chunks created: {len(chunks)}")

    print("\nSTEP 3: Generating embeddings")
    print("=" * 70)
    embedding_generator = EmbeddingGenerator()
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_generator.embed_texts(chunk_texts)

    print(f"Embeddings shape: {embeddings.shape}")

    print("\nSTEP 4: Storing in ChromaDB")
    print("=" * 70)
    vector_store = VectorStore()
    vector_store.reset_collection()
    inserted_count = vector_store.add_chunks(chunks, embeddings)

    print(f"Inserted chunks: {inserted_count}")
    print(f"Collection count: {vector_store.count()}")

    print("\nSTEP 5: Testing vector search")
    print("=" * 70)

    sample_question = "What is the purpose of this RAG system?"
    query_embedding = embedding_generator.embed_query(sample_question)

    search_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3
    )

    print(f"Question: {sample_question}")
    print("-" * 70)

    for index, result in enumerate(search_results, start=1):
        metadata = result["metadata"]

        print(f"\nResult {index}")
        print("-" * 70)
        print(f"Chunk ID        : {result['chunk_id']}")
        print(f"File Name       : {metadata['file_name']}")
        print(f"Page Number     : {metadata['page_number']}")
        print(f"Chunk Number    : {metadata['chunk_number']}")
        print(f"Distance        : {result['distance']}")
        print(f"Relevance Score : {result['relevance_score']}")
        print("\nText Preview:")
        print(result["text"][:500])


if __name__ == "__main__":
    main()