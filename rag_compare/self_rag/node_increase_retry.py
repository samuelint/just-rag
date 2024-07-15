from langchain_core.prompts import ChatPromptTemplate
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
        retry_count = state.get("retry_count") or 0
        retry_count = retry_count + 1

        return {
            "retry_count": retry_count,
        }
