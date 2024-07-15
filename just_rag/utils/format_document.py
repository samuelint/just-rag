from typing import List
from langchain_core.documents import Document


def format_documents(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def format_documents_with_sources(docs: List[Document]):
    formatted = [
        f"Source ID: {i}\nTitle: {doc.metadata['title']}\nContent: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n".join(formatted)
