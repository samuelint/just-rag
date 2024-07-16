import os
import pytest

from just_rag.universal_loader import UniversalLoader

assets_directory_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)


class TestUniversalLoader:
    @pytest.fixture()
    def loader(self) -> UniversalLoader:
        return UniversalLoader()

    def test_load_local_pdf(self, loader: UniversalLoader):
        file_path = os.path.join(assets_directory_path, "sample.pdf")

        result = loader.load([file_path])

        assert len(result) > 0
        assert "this document is a placeholder" in result[0].page_content.lower()

    def test_load_local_markdown(self, loader: UniversalLoader):
        file_path = os.path.join(assets_directory_path, "sample.mdx")

        result = loader.load([file_path])

        assert len(result) > 0
        assert "birdie!!" in result[-1].page_content.lower()

    def test_load_local_html(self, loader: UniversalLoader):
        file_path = os.path.join(assets_directory_path, "sample.html")

        result = loader.load([file_path])

        assert len(result) > 0
        assert "I likes blueberries" in result[-1].page_content


if __name__ == "__main__":
    pytest.main([__file__])
