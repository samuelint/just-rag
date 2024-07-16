from typing import List
from langchain_core.documents import Document


def format_documents(documents: List[Document]):
    return "\n\n".join(doc.page_content for doc in documents)


def format_documents_with_sources(documents: List[Document], meta_keys: List[str] = []):
    formatted = [
        f"Source ID: {i}"
        + "".join(
            f"\n{key}: {value}"
            for key, value in doc.metadata.items()
            if key in meta_keys
        )
        + f"\nContent: {doc.page_content}"
        for i, doc in enumerate(documents)
    ]
    return "\n\n".join(formatted)
