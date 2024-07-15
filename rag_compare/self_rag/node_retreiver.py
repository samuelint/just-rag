import logging
from langchain_core.retrievers import BaseRetriever
from .graph_state import GraphState

logger = logging.getLogger(__name__)


class RetrieverNode:
    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def __call__(self, state: GraphState):

        logger.info("---RETRIEVE---")
        input = state["input"]

        documents = self.retriever.invoke(input)

        return {"documents": documents, "input": input}
