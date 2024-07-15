import pytest
from just_rag.universal_loader import PDFLoader


@pytest.fixture
def instance():
    return PDFLoader()


class TestCanLoad:
    def test_can_load_path_with_pdf_extension(self, instance: PDFLoader):
        assert instance.can_load("test.pdf") is True

    def test_cannot_load_other_extensions_then_pdf(self, instance: PDFLoader):
        assert instance.can_load("test.docx") is False

    def test_can_load_pdf_url(self, instance: PDFLoader):
        assert instance.can_load("http://test.com/some.pdf") is True
        assert instance.can_load("https://test.com/some.pdf") is True
