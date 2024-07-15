from langgraph.graph import StateGraph, END
from langchain_core.retrievers import BaseRetriever
from langgraph.graph.graph import CompiledGraph

from just_rag.llm_factory import LLMFactory

from .graph_state import GraphState
from .node_retreiver import RetrieverNode
from .node_grade_documents import GradeDocumentsNode
from .node_generate import GenerateNode
from .node_transform_query import TransformQueryNode
from .decision_generate import DecideToGenerate
from .decision_has_hallucination import DecisionHasHallucination
from .decision_retry import DecideToRetry
from .node_increase_retry import IncreaseRetryNode


class SelfRagGraphBuilder:
    def __init__(self, retriever: BaseRetriever, llm_factory: LLMFactory) -> None:
        self.retriever = retriever
        self.llm_factory = llm_factory

    def build(self) -> CompiledGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", RetrieverNode(retriever=self.retriever))
        workflow.add_node(
            "grade_documents",
            GradeDocumentsNode(llm_factory=self.llm_factory),
        )
        workflow.add_node("generate", GenerateNode(llm_factory=self.llm_factory))
        workflow.add_node(
            "transform_query", TransformQueryNode(llm_factory=self.llm_factory)
        )
        workflow.add_node("try_generate", IncreaseRetryNode())
        workflow.add_node("try_transform_query", IncreaseRetryNode())

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            DecideToGenerate(),
            {
                "all_documents_not_relevant_to_question": "try_transform_query",
                "relevant_documents": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("try_generate", "generate")
        workflow.add_conditional_edges(
            "generate",
            DecisionHasHallucination(llm_factory=self.llm_factory),
            {
                "not_factual": "try_generate",
                "useful": END,
                "not_useful": "try_transform_query",
                "max_retry_count_reached": END,
            },
        )
        workflow.add_conditional_edges(
            "try_transform_query",
            DecideToRetry(),
            {
                "continue": "transform_query",
                "max_retry_count_reached": END,
            },
        )

        return workflow.compile()
