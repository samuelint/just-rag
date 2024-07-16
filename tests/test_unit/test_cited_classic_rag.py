from typing import List
from decoy import Decoy
from langchain_core.pydantic_v1 import Field
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
import pytest

from just_rag.citation import BaseCitation, BaseCitedAnswer
from just_rag.cited_classic_rag import CitedClassicRag


class SomeCitation(BaseCitation):
    some: str = Field(
        ...,
        description="Page content from the specified source that justifies the answer.",
    )
    title: str = Field(
        ...,
        description="The TITLE quote from the specified source that justifies the answer.",
    )


class SomeCitedAnswer(BaseCitedAnswer[SomeCitation]):
    citations: List[SomeCitation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class TestCitedClassicRag:

    @pytest.fixture
    def llm(self, decoy: Decoy) -> BaseChatModel:
        service = decoy.mock(cls=BaseChatModel)

        return service

    @pytest.fixture
    def retriever(self, decoy: Decoy) -> BaseRetriever:
        service = decoy.mock(cls=BaseRetriever)

        return service

    def test_meta_keys_are_defined_from_citations_schema(self, llm, retriever):
        rag = CitedClassicRag(
            llm=llm,
            retriever=retriever,
            schema=SomeCitedAnswer,
        )

        assert "some" in rag.meta_keys
        assert "title" in rag.meta_keys

    def test_meta_keys_and_citations_fields_are_not_duplicated(self, llm, retriever):
        rag = CitedClassicRag(
            llm=llm, retriever=retriever, schema=SomeCitedAnswer, meta_keys=["title"]
        )

        assert "title" in rag.meta_keys
        assert rag.meta_keys.count("title") == 1
