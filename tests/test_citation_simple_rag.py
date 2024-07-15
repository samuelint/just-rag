import pprint
import pytest
from rag_compare.simple_citation_rag import SimpleCitationRag
from langchain_community.retrievers import WikipediaRetriever
from tests.llm import all_llms


retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)


@pytest.mark.parametrize("llm", all_llms, ids=[llm._llm_type for llm in all_llms])
def test_citation_simple_rag(llm):
    chain = SimpleCitationRag(llm=llm, retriever=retriever).build()

    result = chain.invoke({"input": "How fast are cheetahs?"})

    assert result["input"] == "How fast are cheetahs?"
    assert len(result["context"]) > 0
    assert len(result["answer"].answer) > 0
    assert len(result["answer"].citations) > 0
    pprint.pprint(result["answer"])
