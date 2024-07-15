import pytest
from rag_compare.simple_rag import SimpleRag
from langchain_community.retrievers import WikipediaRetriever
from tests.llm import all_llms
import time

retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)


@pytest.mark.parametrize("llm", all_llms, ids=[llm._llm_type for llm in all_llms])
def test_simple_rag(llm):

    start = time.time()
    chain = SimpleRag(llm=llm, retriever=retriever).build()
    result = chain.invoke({"input": "How fast are cheetahs?"})
    end = time.time()
    exec_time = f"Execution time: {end - start:.2f} seconds"
    print(exec_time)

    assert result["input"] == "How fast are cheetahs?"
    assert len(result["context"]) > 0
    assert len(result["answer"]) > 0
    print(["answer"])


if __name__ == "__main__":
    pytest.main([__file__])
