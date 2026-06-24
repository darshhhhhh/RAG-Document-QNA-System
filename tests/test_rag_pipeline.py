from src.rag_pipeline import RAGPipeline


class FakeRetriever:
    def retrieve(self, query: str):
        return []


def test_rag_pipeline_returns_fallback_when_no_chunks_found():
    rag_pipeline = RAGPipeline(
        retriever=FakeRetriever()
    )

    response = rag_pipeline.ask("Who is the CEO of OpenAI?")

    assert response["answer"] == "I could not find this information in the uploaded document."
    assert response["confidence"] == "Low"
    assert response["answer_type"] == "fallback"
    assert response["sources"] == []