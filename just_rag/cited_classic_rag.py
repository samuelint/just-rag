from typing import List
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from just_rag.base_rag import BaseRag, default_system_prompt
from just_rag.utils.format_document import format_documents_with_sources


class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    result: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class CitedClassicRag(BaseRag):
    llm: BaseChatModel
    retriever: BaseRetriever

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever):
        self.llm = llm.with_structured_output(CitedAnswer)
        self.retriever = retriever

    def build(self) -> Runnable:
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_documents_with_sources(x["context"]))
            )
            | self.__prompt()
            | self.llm
        )

        retrieve_docs = (lambda x: x["input"]) | self.retriever

        return RunnablePassthrough.assign(context=retrieve_docs).assign(
            result=rag_chain_from_docs
        )

    def __prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", default_system_prompt),
                ("human", "{input}"),
            ]
        )
