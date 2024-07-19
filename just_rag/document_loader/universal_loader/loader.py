from abc import ABC, abstractmethod
from langchain_core.documents import Document


class FileOrUrlLoader(ABC):
    @abstractmethod
    def can_load(self, path_or_url: str) -> bool:
        pass

    @abstractmethod
    def load(self, path_or_url: str) -> list[Document]:
        pass

    def load_many(self, paths_or_urls: list[str]) -> list[Document]:
        documents: list[Document] = []
        for path_or_url in paths_or_urls:
            result = self.load(path_or_url)
            documents.extend(result)

        return documents
