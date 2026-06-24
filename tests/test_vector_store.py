import numpy as np

from src.vector_store import VectorStore


def test_vector_store_adds_and_searches_chunks(tmp_path):
    vector_store = VectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_collection"
    )

    vector_store.reset_collection()

    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Machine learning is a field of artificial intelligence.",
            "metadata": {
                "file_name": "test.pdf",
                "page_number": 1,
                "chunk_number": 1,
                "char_count": 58,
                "word_count": 8
            }
        },
        {
            "chunk_id": "chunk_2",
            "text": "FastAPI is used to build backend APIs.",
            "metadata": {
                "file_name": "test.pdf",
                "page_number": 2,
                "chunk_number": 1,
                "char_count": 40,
                "word_count": 7
            }
        }
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
    )

    inserted_count = vector_store.add_chunks(chunks, embeddings)

    assert inserted_count == 2
    assert vector_store.count() == 2

    query_embedding = np.array([1.0, 0.0, 0.0])
    results = vector_store.search(query_embedding, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_1"