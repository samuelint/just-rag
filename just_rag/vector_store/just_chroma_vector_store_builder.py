from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.indexes import SQLRecordManager
from langchain_core.indexing import RecordManager

from just_rag.universal_loader.universal_document_loader import UniversalDocumentLoader
from just_rag.vector_store.base_vector_store_builder import BaseVectorStoreBuilder


class JustChromaVectorStoreBuilder(BaseVectorStoreBuilder):
    """
    An opiniated vector store builder that just works.
    """

    def __init__(
        self,
        collection_name: str,
        file_or_urls: list[str] = [],
        record_manager_db_url: Optional[str] = None,
        chroma_persist_directory: Optional[str] = None,
    ) -> None:
        self.document_loader = UniversalDocumentLoader(paths=file_or_urls)
        self.collection_name = collection_name
        self.chroma_persist_directory = chroma_persist_directory
        self.record_manager_db_url = record_manager_db_url

    def _create_embeddings(self) -> Embeddings:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _create_record_manager(self) -> RecordManager:
        namespace = f"chroma/{self.collection_name}"
        record_manager = SQLRecordManager(namespace, db_url=self.record_manager_db_url)
        record_manager.create_schema()

        return record_manager

    def _create_vector_store(self, embedding: Embeddings) -> VectorStore:
        return Chroma(
            persist_directory=self.chroma_persist_directory,
            embedding_function=embedding,
        )

    def _load_documents(self) -> List[Document]:
        documents = self.document_loader.load()

        return filter_complex_metadata(documents)
