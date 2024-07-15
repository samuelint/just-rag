from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    question: str  # User question
    result: str  # LLM generation (answer)
    documents: List[Document]

    retry_count: int = 0
    max_retry: int = 3
