import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


def print_rag_response(response: dict):
    print("\nQUESTION")
    print("=" * 80)
    print(response["question"])

    print("\nANSWER")
    print("=" * 80)
    print(response["answer"])

    print("\nCONFIDENCE")
    print("=" * 80)
    print(response["confidence"])
    print(f"Best Score: {response['best_score']}")

    print("\nSOURCES")
    print("=" * 80)

    if not response["sources"]:
        print("No strong source found.")
    else:
        for index, source in enumerate(response["sources"], start=1):
            print(
                f"{index}. {source['file_name']} | "
                f"Page {source['page_number']} | "
                f"Chunk {source['chunk_number']} | "
                f"Score {source['score']}"
            )


def main():
    vector_store = VectorStore()

    if vector_store.count() == 0:
        print("Vector DB is empty. Run this first:")
        print("python scripts\\preview_vector_store.py")
        return

    rag = RAGPipeline()

    questions = [
        "What is the purpose of this RAG system?",
        "Which API endpoints are available in the backend?",
        "What are the recommended evaluation metrics?",
        "What should happen when the answer is not found in the document?",
        "Who is the CEO of OpenAI?"
    ]

    for question in questions:
        response = rag.ask(question)
        print_rag_response(response)


if __name__ == "__main__":
    main()