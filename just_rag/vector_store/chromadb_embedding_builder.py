from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_community.vectorstores.utils import filter_complex_metadata


class ChromaDBEmbeddingBuilder:
    def __init__(self, documents: list[Document] = []) -> None:
        self.documents = documents

    def to_chroma(self, collection_name: str) -> Chroma:
        langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embedding = create_langchain_embedding(langchain_embeddings)

        docs = filter_complex_metadata(self.documents)
        return Chroma.from_documents(
            documents=docs,
            collection_name=collection_name,
            embedding=embedding,
        )

    def to_retriever(self, collection_name: str) -> BaseRetriever:
        return self.to_chroma(collection_name).as_retriever()
