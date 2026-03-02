"""Request / response schemas for the search endpoint."""

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The search query string.",
    )
    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="The number of top results to return.",
    )
    search_type: Literal["dense", "sparse", "hybrid"] = Field(
        "hybrid",
        description="The type of search to perform: 'dense', 'sparse', or 'hybrid'.",
    )
    language: Optional[str] = Field(
        None,
        description="Optional language hint for sparse / hybrid search.",
    )


class SearchResult(BaseModel):
    doc_id: int = Field(..., description="The unique identifier of the document.")
    title: Optional[str] = Field(None, description="The title of the document chunk.")
    text: str = Field(..., description="The text content of the document.")
    score: Optional[float] = Field(
        None, description="The relevance score of the document."
    )
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional metadata associated with the document."
    )


class SearchResponse(BaseModel):
    query: str = Field(..., description="The original query string (echo-back).")
    collection_name: str = Field(
        ..., description="The name of the collection searched."
    )
    search_type: str = Field(..., description="The search strategy that was used.")
    total_results: int = Field(..., description="Number of results returned.")
    results: list[SearchResult] = Field(
        ..., description="A list of search results matching the query."
    )
