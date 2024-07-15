import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from ..llm_factory import LLMFactory
from .graph_state import GraphState

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)


class GradeDocumentsNode:
    def __init__(self, llm_factory: LLMFactory) -> None:
        llm = llm_factory(temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        retrieval_grader = grade_prompt | structured_llm_grader

        self.retrieval_grader = retrieval_grader

    def __call__(self, state: GraphState):
        logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        input = state["input"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": input, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {
            "documents": filtered_docs,
            "input": input,
        }
