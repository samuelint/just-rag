import pytest
from just_rag.document_loader.universal_loader import MarkdownLoader


@pytest.fixture
def instance():
    return MarkdownLoader()


class TestCanLoad:
    def test_can_load_path_with_md_extension(self, instance: MarkdownLoader):
        assert instance.can_load("test.md") is True

    def test_can_load_path_with_mdx_extension(self, instance: MarkdownLoader):
        assert instance.can_load("test.mdx") is True

    def test_cannot_load_other_extensions_then_markdown(self, instance: MarkdownLoader):
        assert instance.can_load("test.docx") is False

    def test_can_load_md_url(self, instance: MarkdownLoader):
        assert instance.can_load("http://test.com/some.md") is True
        assert instance.can_load("https://test.com/some.md") is True
