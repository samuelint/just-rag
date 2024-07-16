from typing import Generic, List, TypeVar
from langchain_core.pydantic_v1 import BaseModel, Field


class BaseCitation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )


TCitation = TypeVar("TCitation", bound=BaseCitation)


class BaseCitedAnswer(BaseModel, Generic[TCitation]):
    """Answer the user question based only on the given sources, and cite the sources used."""

    result: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[TCitation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class DefaultCitation(BaseCitation):
    page_content: str = Field(
        ...,
        description="Page content from the specified source that justifies the answer.",
    )


class DefaultCitedAnswer(BaseCitedAnswer[DefaultCitation]):
    citations: List[DefaultCitation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
