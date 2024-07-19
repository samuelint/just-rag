import os
import pytest

from just_rag.document_loader.universal_loader import UniversalDocumentLoader

assets_directory_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)


class TestUniversalLoader:
    def test_load_local_pdf(self):
        file_path = os.path.join(assets_directory_path, "sample.pdf")
        loader = UniversalDocumentLoader(paths=[file_path])

        result = loader.load()

        assert len(result) > 0
        assert "this document is a placeholder" in result[0].page_content.lower()

    def test_load_local_markdown(self):
        file_path = os.path.join(assets_directory_path, "sample.mdx")
        loader = UniversalDocumentLoader(paths=[file_path])

        result = loader.load()

        assert len(result) > 0
        assert "birdie!!" in result[-1].page_content.lower()

    def test_load_local_html(self):
        file_path = os.path.join(assets_directory_path, "sample.html")
        loader = UniversalDocumentLoader(paths=[file_path])

        result = loader.load()

        assert len(result) > 0
        assert "I likes blueberries" in result[-1].page_content


if __name__ == "__main__":
    pytest.main([__file__])
