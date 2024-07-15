from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from just_rag.base_rag import BaseRag, default_system_prompt
from .utils.format_document import format_documents


class ClassicRag(BaseRag):
    llm: BaseChatModel
    retriever: BaseRetriever

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever

    def build(self) -> Runnable:
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_documents(x["context"]))
            )
            | self.__prompt()
            | self.llm
            | StrOutputParser()
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
