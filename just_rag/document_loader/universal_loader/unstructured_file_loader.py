from typing import Literal
from langchain_core.documents import Document
from unstructured.cleaners.core import clean
from .loader import FileOrUrlLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader


class UnstructuredFileLoader(FileOrUrlLoader):
    mode: Literal["single", "elements", "paged"]
    strategy: Literal["hi_res", "fast"]

    def __init__(
        self,
        mode: Literal["single", "elements", "paged"] = "elements",
        strategy: Literal["hi_res", "fast"] = "hi_res",
    ) -> None:
        self.mode = mode
        self.strategy = strategy

    def can_load(self, path_or_url: str) -> bool:
        return (
            path_or_url.lower().endswith(".html")
            or path_or_url.lower().endswith(".htm")
            or path_or_url.lower().endswith(".xml")
        )

    def load(self, path_or_url: str) -> list[Document]:
        loader = UnstructuredHTMLLoader(
            path_or_url,
            mode=self.mode,
            strategy=self.strategy,
            post_processors=[clean],
        )

        return loader.load()
