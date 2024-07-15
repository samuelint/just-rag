import pytest

from just_rag.universal_loader import WebsiteLoader


@pytest.fixture
def instance():
    return WebsiteLoader()


class TestCanLoad:
    def test_can_load_any_http_url(self, instance: WebsiteLoader):
        assert instance.can_load("http://test.com") is True

    def test_can_load_any_https_url(self, instance: WebsiteLoader):
        assert instance.can_load("https://test.com") is True

    def test_cannot_load_local_file(self, instance: WebsiteLoader):
        assert instance.can_load("./test.docx") is False
