import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ..llm_factory import LLMFactory
from .graph_state import GraphState


logger = logging.getLogger(__name__)

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:""",
    input_variables=["question", "document"],
)


class GenerateNode:
    def __init__(self, llm_factory: LLMFactory) -> None:
        llm = llm_factory(temperature=0)
        self.rag_chain = prompt | llm | StrOutputParser()

    def __call__(self, state: GraphState):
        logger.info("---GENERATE---")
        input = state["input"]
        documents = state["documents"]

        generation = self.rag_chain.invoke({"context": documents, "question": input})

        return {
            "documents": documents,
            "input": input,
            "result": generation,
        }
