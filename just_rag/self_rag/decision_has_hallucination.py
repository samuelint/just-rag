import logging
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from just_rag.utils.format_document import format_documents_with_sources

from .graph_state import GraphState
from .decision_retry import DecideToRetry

from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


grade_answer_system = """\
You are a grader assessing whether an answer addresses / resolves a question. \
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.\
"""
grade_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_answer_system),
        ("human", "User question: \n\n {question} \n\n LLM answer: {generation}"),
    ]
)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


grade_hallucination_system = """\
You are a grader assessing whether an LLM answer is grounded in / supported by a set of retrieved facts. \
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.\
"""
grade_hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_hallucination_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM answer: {generation}"),
    ]
)


class DecisionHasHallucination:
    def __init__(self, llm_factory: LLMFactory) -> None:
        llm = llm_factory(temperature=0)
        grade_hallucination_structured_llm_grader = llm.with_structured_output(
            GradeHallucinations
        )

        self.hallucination_grader = (
            grade_hallucination_prompt | grade_hallucination_structured_llm_grader
        )

        grade_answer_structured_llm_grader = llm.with_structured_output(GradeAnswer)
        self.answer_grader = grade_answer_prompt | grade_answer_structured_llm_grader

    def __call__(self, state: GraphState):
        should_retry = DecideToRetry()(state)
        if should_retry != "continue":
            return should_retry

        logger.info("---CHECK HALLUCINATIONS---")
        question = state["input"]
        documents = state["documents"]
        result = state["result"]

        are_documents_based_on_facts = self.are_documents_based_on_facts(
            result=result, documents=documents
        )

        if not are_documents_based_on_facts:
            return "not_factual"

        does_answer_resolve_question = self.does_answer_resolve_question(
            question=question,
            result=result,
        )
        if does_answer_resolve_question:
            return "useful"
        else:
            return "not_useful"

    def are_documents_based_on_facts(
        self, result: str, documents: List[Document]
    ) -> bool:
        score = self.hallucination_grader.invoke(
            {
                "documents": format_documents_with_sources(documents),
                "generation": result,
            }
        )
        return score.binary_score == "yes"

    def does_answer_resolve_question(self, question: str, result: str) -> bool:
        score = self.answer_grader.invoke({"question": question, "generation": result})
        return score.binary_score == "yes"
