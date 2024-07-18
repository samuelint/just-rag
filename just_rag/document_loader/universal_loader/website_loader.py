from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders import SeleniumURLLoader
from .loader import FileOrUrlLoader


class WebsiteLoader(FileOrUrlLoader):

    def can_load(self, path_or_url: str) -> bool:
        try:
            result = urlparse(path_or_url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def load(self, path_or_url: str) -> list[Document]:
        return self.load_many([path_or_url])

    def load_many(self, paths_or_urls: list[str]) -> list[Document]:
        loader = SeleniumURLLoader(urls=paths_or_urls)

        return loader.load()
