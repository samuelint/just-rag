from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm_factory import LLMFactory
from .graph_state import GraphState


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


class IncreaseRetryNode:
    def __call__(self, state: GraphState):

        print("---INCREASE RETRY---")
        retry_count = state.get("retry_count", 0)
        retry_count = retry_count + 1

        return {
            "retry_count": retry_count,
        }
