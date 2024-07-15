from typing import List
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from rag_compare.base_rag import BaseRag, default_system_prompt


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

    answer: str = Field(
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
                context=(lambda x: self.__format_docs(x["context"]))
            )
            | self.__prompt()
            | self.llm
        )

        retrieve_docs = (lambda x: x["input"]) | self.retriever

        return RunnablePassthrough.assign(context=retrieve_docs).assign(
            answer=rag_chain_from_docs
        )

    def __prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", default_system_prompt),
                ("human", "{input}"),
            ]
        )

    def __format_docs(self, docs: List[Document]) -> str:
        formatted = [
            f"Source ID: {i}\nArticle Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
            for i, doc in enumerate(docs)
        ]
        return "\n\n" + "\n\n".join(formatted)
