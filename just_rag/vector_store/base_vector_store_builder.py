from abc import abstractmethod
from typing import List, Literal, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.indexing import RecordManager, index, IndexingResult


class BaseVectorStoreBuilder:
    embedding: Embeddings = None
    vector_store: VectorStore = None
    record_manager: RecordManager = None

    def get_vector_store(self) -> VectorStore:
        if self.vector_store is None:
            embedding = self._get_embeddings()
            self.vector_store = self._create_vector_store(embedding=embedding)

        return self.vector_store

    def get_retriever(self) -> BaseRetriever:
        vector_store = self.get_vector_store()

        return vector_store.as_retriever()

    def sync(
        self,
        cleanup: Literal["incremental", "full", None] = None,
        source_id_key: Optional[str] = "source",
    ) -> IndexingResult:
        """
        Docs about cleanup strategy: https://python.langchain.com/v0.2/docs/how_to/indexing/#none-deletion-mode
        """
        documents = self._load_documents()
        vector_store = self.get_vector_store()
        record_manager = self._get_record_manager()

        return index(
            docs_source=documents,
            record_manager=record_manager,
            vector_store=vector_store,
            cleanup=cleanup,
            source_id_key=source_id_key,
        )

    @abstractmethod
    def _create_embeddings(self) -> Embeddings:
        pass

    @abstractmethod
    def _create_record_manager(self) -> RecordManager:
        pass

    @abstractmethod
    def _create_vector_store(self, embedding: Embeddings) -> VectorStore:
        pass

    @abstractmethod
    def _load_documents(self) -> List[Document]:
        pass

    def _get_record_manager(
        self,
    ) -> RecordManager:
        if self.record_manager is None:
            self.record_manager = self._create_record_manager()
        return self.record_manager

    def _get_embeddings(self) -> Embeddings:
        if self.embedding is None:
            self.embedding = self._create_embeddings()
        return self.embedding
