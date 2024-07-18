import os
import shutil
import pytest

from just_rag import CitedClassicRag
from tests.llm import openai_llm
from tests.test_functional.simple_huggingface_chroma_vector_store_builder import (
    SimpleChromaVectorStoreBuilder,
)


assets_directory_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)

test_chromadb_path = "./tests_chroma_db"
test_record_manager_db_path = "tests_record_manager_cache.sql"
test_record_manager_db_url = f"sqlite:///{test_record_manager_db_path}"


@pytest.fixture(scope="session", autouse=True)
def delete_files_before_test():
    if os.path.exists(test_chromadb_path):
        shutil.rmtree(test_chromadb_path)

    if os.path.exists(test_record_manager_db_path):
        os.remove(test_record_manager_db_path)


class TestEmbedding:

    @pytest.fixture
    def documents_paths(self) -> list[str]:
        return [
            os.path.join(
                assets_directory_path, "Charte canadienne des droits et libertés.html"
            )
        ]

    def test_embedding(self, documents_paths: list[str]):
        builder = SimpleChromaVectorStoreBuilder(
            file_or_urls=documents_paths,
            collection_name="charte_canadienne_des_droits_et_libertes",
            record_manager_db_url=test_record_manager_db_url,
            chroma_persist_directory=test_chromadb_path,
        )
        builder.sync()
        retriever = builder.get_retriever()
        chain = CitedClassicRag(llm=openai_llm, retriever=retriever).build()

        result = chain.invoke(
            {
                "input": "En temps que citoyen, est-ce que j'ai le droit d'entrer et sortir du canada quand je veux? Repondre oui ou non.",
            }
        )

        assert "oui" in result["result"].result.lower()


class TestVectorStoreSync:

    @pytest.fixture
    def documents_paths(self) -> list[str]:
        return [
            os.path.join(
                assets_directory_path, "Charte canadienne des droits et libertés.html"
            )
        ]

    def test_not_syncing_does_not_load_documents(self, documents_paths: list[str]):
        builder = SimpleChromaVectorStoreBuilder(
            collection_name="vector_store_not_synced",
            file_or_urls=documents_paths,
            record_manager_db_url="sqlite:///:memory:",
            chroma_persist_directory="./tests_chroma_db2",
        )

        retriever = builder.get_retriever()
        chain = CitedClassicRag(llm=openai_llm, retriever=retriever).build()

        result = chain.invoke(
            {
                "input": "En temps que citoyen, est-ce que j'ai le droit d'entrer et sortir du canada quand je veux? Repondre oui ou non.",
            }
        )

        assert "non" in result["result"].result.lower()

    def test_vector_store_content_is_persisted(self, documents_paths: list[str]):
        builder1 = SimpleChromaVectorStoreBuilder(
            collection_name="vector_store_sync",
            file_or_urls=documents_paths,
            record_manager_db_url=test_record_manager_db_url,
            chroma_persist_directory=test_chromadb_path,
        )
        builder1.sync()

        builder2 = SimpleChromaVectorStoreBuilder(
            collection_name="vector_store_sync",
            file_or_urls=documents_paths,
            record_manager_db_url=test_record_manager_db_url,
            chroma_persist_directory=test_chromadb_path,
        )

        retriever = builder2.get_retriever()
        chain = CitedClassicRag(llm=openai_llm, retriever=retriever).build()

        result = chain.invoke(
            {
                "input": "En temps que citoyen, est-ce que j'ai le droit d'entrer et sortir du canada quand je veux? Repondre oui ou non.",
            }
        )

        assert "oui" in result["result"].result.lower()
