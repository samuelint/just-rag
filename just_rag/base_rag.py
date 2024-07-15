from abc import ABC, abstractmethod
from typing import List, TypedDict
from langchain_core.runnables import Runnable
from langchain_core.documents import Document


default_system_prompt: str = (
    "You're a helpful AI assistant. Given a user question, "
    "and some article snippets, answer the user question. "
    "If none of the articles answer the question, "
    "just say you don't know."
    "\n\nHere are the articles: "
    "{context}"
)


class RagState(TypedDict):
    input: str
    context: List[Document]
    answer: str


class BaseRag(ABC):
    @abstractmethod
    def build(self) -> Runnable:
        pass
