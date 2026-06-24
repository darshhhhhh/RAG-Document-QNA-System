import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retriever import Retriever
from src.vector_store import VectorStore


def print_results(question: str, results: list[dict]):
    print("\nQUESTION")
    print("=" * 80)
    print(question)

    print("\nRETRIEVED + RERANKED RESULTS")
    print("=" * 80)

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nResult {index}")
        print("-" * 80)
        print(f"Chunk ID                : {result['chunk_id']}")
        print(f"File Name               : {metadata['file_name']}")
        print(f"Page Number             : {metadata['page_number']}")
        print(f"Chunk Number            : {metadata['chunk_number']}")
        print(f"Vector Relevance Score  : {result['relevance_score']}")
        print(f"Vector Distance         : {result['distance']}")
        print(f"Rerank Score            : {result['rerank_score']}")
        print(f"Normalized Rerank Score : {result['normalized_rerank_score']}")
        print("\nText Preview:")
        print(result["text"][:500])


def main():
    vector_store = VectorStore()

    if vector_store.count() == 0:
        print("Vector DB is empty. Run this first:")
        print("python scripts\\preview_vector_store.py")
        return

    print(f"Vector DB Chunk Count: {vector_store.count()}")

    retriever = Retriever(vector_store=vector_store)

    test_questions = [
        "What is the purpose of this RAG system?",
        "Which API endpoints are available in the backend?",
        "What are the recommended evaluation metrics?",
        "What should happen when the answer is not found in the document?"
    ]

    for question in test_questions:
        results = retriever.retrieve(question)
        print_results(question, results)


if __name__ == "__main__":
    main()