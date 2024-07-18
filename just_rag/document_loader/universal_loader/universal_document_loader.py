from typing import Optional, Union
from langchain_core.documents import Document

from .loader import FileOrUrlLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .website_loader import WebsiteLoader
from .unstructured_file_loader import UnstructuredFileLoader


_DEFAULT_LOADERS = [
    PDFLoader(),
    MarkdownLoader(),
    WebsiteLoader(),
    UnstructuredFileLoader(),
]


class UniversalDocumentLoader:
    def __init__(
        self,
        paths: list[str] = [],
        loaders: Optional[list[FileOrUrlLoader]] = _DEFAULT_LOADERS,
    ) -> None:
        self.paths_or_urls = paths
        self.loaders = loaders

    def add(self, path_or_url: Union[str, list[str]]) -> None:
        if isinstance(path_or_url, list):
            self.paths_or_urls.extend(path_or_url)
        else:
            self.paths_or_urls.append(path_or_url)

    def load(self) -> list[Document]:
        loaded_documents = []

        for path_or_url in self.paths_or_urls:
            for loader in self.loaders:
                if loader.can_load(path_or_url=path_or_url) is True:
                    result = loader.load(path_or_url=path_or_url)
                    loaded_documents.extend(result)
                    break

        return loaded_documents
