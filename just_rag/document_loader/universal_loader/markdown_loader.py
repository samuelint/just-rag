from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from .loader import FileOrUrlLoader


class MarkdownLoader(FileOrUrlLoader):

    def can_load(self, path_or_url: str) -> bool:
        return path_or_url.lower().endswith(".md") or path_or_url.lower().endswith(
            ".mdx"
        )

    def load(self, path_or_url: str) -> list[Document]:
        loader = UnstructuredMarkdownLoader(file_path=path_or_url, mode="elements")

        return loader.load()

    def load_many(self, paths_or_urls: list[str]) -> list[Document]:
        return self.load(paths_or_urls)
