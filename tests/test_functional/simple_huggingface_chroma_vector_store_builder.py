from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from just_rag.document_loader.universal_loader.universal_document_loader import (
    UniversalDocumentLoader,
)
from just_rag.vector_store.just_chroma_vector_store_builder import (
    JustChromaVectorStoreBuilder,
)


class SimpleHuggingFaceChromaVectorStoreBuilder(JustChromaVectorStoreBuilder):
    def __init__(
        self,
        file_or_urls: list[str] = [],
        huggingface_embedding_model_name: Optional[str] = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> None:
        self.document_loader = UniversalDocumentLoader(paths=file_or_urls)
        self.huggingface_embedding_model_name = huggingface_embedding_model_name

        super().__init__(**kwargs)

    def _create_embeddings(self) -> Embeddings:
        return HuggingFaceEmbeddings(model_name=self.huggingface_embedding_model_name)

    def _load_documents(self) -> List[Document]:
        documents = self.document_loader.load()

        return filter_complex_metadata(documents)
