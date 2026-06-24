from src.text_cleaner import TextCleaner


def test_text_cleaner_removes_extra_spaces():
    cleaner = TextCleaner()

    raw_text = "Machine     Learning   is   useful."
    cleaned_text = cleaner.clean(raw_text)

    assert cleaned_text == "Machine Learning is useful."


def test_text_cleaner_fixes_pdf_artifacts():
    cleaner = TextCleaner()

    raw_text = "Document Q&A; system"
    cleaned_text = cleaner.clean(raw_text)

    assert "Q&A;" not in cleaned_text
    assert "Q&A" in cleaned_text