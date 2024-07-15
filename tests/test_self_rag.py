import pytest

from rag_compare import SelfRagGraphBuilder
from rag_compare.llm_factory import LLMFactory
from tests.llm import all_llm_factories
from langchain_community.retrievers import WikipediaRetriever
from langgraph.graph.graph import CompiledGraph


class TestSelfRag:
    @pytest.fixture(
        params=all_llm_factories, ids=[llm._llm_type for llm in all_llm_factories]
    )
    def llm_factory(self, request) -> LLMFactory:
        return request.param

    @pytest.fixture
    def retriever(self):
        # Note:
        # The number of retreived documents should be inferior to the local llm context size.
        # 6 * 1000 < n_ctx
        return WikipediaRetriever(
            top_k_results=6,
            doc_content_chars_max=1000,
        )

    @pytest.fixture
    def self_rag(self, llm_factory, retriever) -> CompiledGraph:
        return SelfRagGraphBuilder(retriever=retriever, llm_factory=llm_factory).build()

    def test_simple_grounded_with_facts(self, self_rag: CompiledGraph):
        result = self_rag.invoke(
            {
                "input": "Who is René Lévesque?",
                "max_retry": 1,
            }
        )

        assert len(result["result"]) > 0
        assert len(result["documents"]) > 0

    def test_generate_graph(self, self_rag: CompiledGraph):
        marmaid_graph = self_rag.get_graph().draw_mermaid()

        print(marmaid_graph)

    def test_no_possible_answer_exit(self, self_rag: CompiledGraph):
        """
        Test that it does not infinitly loop when no possible answer is found
        """
        result = self_rag.invoke(
            {
                "input": "What is gn230r9jfq9g34g0f9m?",
                "max_retry": 1,
            }
        )

        assert len(result["documents"]) == 0
        assert result["retry_count"] > result["max_retry"]


if __name__ == "__main__":
    pytest.main([__file__])
