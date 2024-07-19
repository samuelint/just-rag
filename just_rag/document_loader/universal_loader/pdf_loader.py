from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .loader import FileOrUrlLoader


class PDFLoader(FileOrUrlLoader):

    def can_load(self, path_or_url: str) -> bool:
        return path_or_url.lower().endswith(".pdf")

    def load(self, path_or_url: str) -> list[Document]:
        loader = PyPDFLoader(file_path=path_or_url)

        return loader.load()
