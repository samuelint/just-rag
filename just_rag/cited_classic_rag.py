from typing import List, Type
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from just_rag.base_rag import BaseRag, default_system_prompt
from .citation import BaseCitedAnswer, DefaultCitedAnswer
from just_rag.utils.format_document import format_documents_with_sources


class CitedClassicRag(BaseRag):
    llm: BaseChatModel
    retriever: BaseRetriever

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        schema: Type[BaseCitedAnswer] = DefaultCitedAnswer,
        meta_keys: List[str] = [],
    ):
        self.llm = llm.with_structured_output(schema)
        self.retriever = retriever
        self.meta_keys = self.__get_meta_keys_from_schema(
            meta_keys=meta_keys, schema=schema
        )

    def build(self) -> Runnable:
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(
                    lambda x: format_documents_with_sources(
                        x["context"], meta_keys=self.meta_keys
                    )
                )
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

    def __get_meta_keys_from_schema(
        self, meta_keys: List[str], schema: Type[BaseCitedAnswer]
    ) -> List[str]:
        citation_fields = schema.__fields__["citations"].type_.__fields__
        citation_field_names = list(citation_fields.keys())

        return list(set(meta_keys + citation_field_names))
