from src.text_splitter import MetadataAwareTextSplitter


def test_text_splitter_creates_chunks_with_metadata():
    pages = [
        {
            "file_name": "test.pdf",
            "file_path": "test.pdf",
            "page_number": 1,
            "text": (
                "This is a sample document for testing chunk creation. "
                "It contains enough text to verify that the splitter works correctly. "
                "The chunk should preserve metadata such as file name and page number. "
            ) * 10,
            "char_count": 1000
        }
    ]

    splitter = MetadataAwareTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        min_chunk_size=100
    )

    chunks = splitter.split_pages(pages)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    first_chunk = chunks[0]

    assert "chunk_id" in first_chunk
    assert "text" in first_chunk
    assert "metadata" in first_chunk
    assert first_chunk["metadata"]["file_name"] == "test.pdf"
    assert first_chunk["metadata"]["page_number"] == 1
    assert first_chunk["metadata"]["chunk_number"] == 1
    assert first_chunk["metadata"]["char_count"] > 0
    assert first_chunk["metadata"]["word_count"] > 0