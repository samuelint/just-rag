import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm_factory import LLMFactory
from .graph_state import GraphState

logger = logging.getLogger(__name__)

system = """You a question re-writer that converts an input question to a better version that is optimized
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)


class TransformQueryNode:
    def __init__(self, llm_factory: LLMFactory) -> None:
        llm = llm_factory(temperature=0)
        self.question_rewriter = re_write_prompt | llm | StrOutputParser()

    def __call__(self, state: GraphState):
        logger.info("---TRANSFORM QUERY---")
        input = state["input"]

        better_question = self.question_rewriter.invoke({"question": input})

        return {
            "input": better_question,
        }
