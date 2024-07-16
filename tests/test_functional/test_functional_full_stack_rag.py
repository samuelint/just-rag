import os

import pytest

from just_rag import CitedClassicRag
from just_rag.universal_loader import UniversalDocumentLoader
from just_rag.vector_store import ChromaDBEmbeddingBuilder
from tests.llm import openai_llm


assets_directory_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)


class TestEmbedding:

    @pytest.fixture
    def documents_paths(self) -> list[str]:
        return [
            os.path.join(
                assets_directory_path, "Charte canadienne des droits et libertés.html"
            )
        ]

    @pytest.fixture
    def document_loader(self, documents_paths: list[str]) -> UniversalDocumentLoader:
        loader = UniversalDocumentLoader()
        loader.add(documents_paths)

        return loader

    def test_embedding(self, document_loader: UniversalDocumentLoader):
        documents = document_loader.load()

        embedding_builder = ChromaDBEmbeddingBuilder(documents=documents)
        retriever = embedding_builder.to_retriever("test")
        chain = CitedClassicRag(llm=openai_llm, retriever=retriever).build()

        result = chain.invoke(
            {
                "input": "En temps que citoyen, est-ce que j'ai le droit d'entrer et sortir du canada quand je veux? Repondre oui ou non.",
            }
        )

        assert "oui" in result["result"].result.lower()
