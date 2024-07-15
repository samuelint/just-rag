from typing import Optional
from langchain_core.documents import Document

from .loader import FileOrUrlLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .website_loader import WebsiteLoader


_DEFAULT_LOADERS = [PDFLoader(), MarkdownLoader(), WebsiteLoader()]


class UniversalLoader:
    def __init__(
        self, loaders: Optional[list[FileOrUrlLoader]] = _DEFAULT_LOADERS
    ) -> None:
        self.loaders = loaders

    def load(self, paths_or_urls: list[str]) -> list[Document]:
        documents = []

        for path_or_url in paths_or_urls:
            for loader in self.loaders:
                if loader.can_load(path_or_url=path_or_url) is True:
                    result = loader.load(path_or_url=path_or_url)
                    documents.extend(result)
                    break

        return documents
