from langchain_core.retrievers import BaseRetriever
from .graph_state import GraphState


class RetrieverNode:
    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def __call__(self, state: GraphState):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)

        return {"documents": documents, "question": question}
