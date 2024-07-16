import pprint
from typing import List
from langchain_core.pydantic_v1 import Field
import pytest
from just_rag import CitedClassicRag
from langchain_community.retrievers import WikipediaRetriever
from just_rag.citation import BaseCitation, BaseCitedAnswer
from tests.llm import all_llms


retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)


class Citation(BaseCitation):
    page_content: str = Field(
        ...,
        description="Page content from the specified source that justifies the answer.",
    )
    title: str = Field(
        ...,
        description="The TITLE quote from the specified source that justifies the answer.",
    )


class CitedAnswer(BaseCitedAnswer[Citation]):
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class TestFunctionalCitedClassicRag:
    @pytest.mark.parametrize("llm", all_llms, ids=[llm._llm_type for llm in all_llms])
    def test_cited_classic_rag(self, llm):
        chain = CitedClassicRag(
            llm=llm,
            retriever=retriever,
            schema=CitedAnswer,
        ).build()

        result = chain.invoke({"input": "How fast are cheetahs?"})

        assert result["input"] == "How fast are cheetahs?"
        assert len(result["context"]) > 0
        assert len(result["result"].result) > 0
        assert len(result["result"].citations) > 0
        assert result["result"].citations[0].source_id is not None
        assert len(result["result"].citations[0].page_content) > 0
        assert len(result["result"].citations[0].title) > 0
        pprint.pprint(result["result"])
