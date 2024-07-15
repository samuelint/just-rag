from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from .decision_retry import DecideToRetry

from ..llm_factory import LLMFactory


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


grade_answer_system = """You are a grader assessing whether an answer addresses / resolves a question
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
grade_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_answer_system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


grade_hallucination_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
grade_hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_hallucination_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
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

    def __call__(self, state):
        """
        Determines whether the generation is grounded in the document.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        should_retry = DecideToRetry()(state)
        if should_retry != "continue":
            return should_retry

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["result"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        are_documents_based_on_facts = score.binary_score

        # Check hallucination
        if are_documents_based_on_facts == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            does_answer_resolve_question = score.binary_score
            if does_answer_resolve_question == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not_useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not_factual"
