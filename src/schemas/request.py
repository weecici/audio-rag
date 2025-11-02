from pydantic import BaseModel, Field
from typing import Literal


class IngestRequest(BaseModel):
    file_paths: list[str] = Field(
        default=[], description="List of file paths to ingest documents from"
    )
    file_dir: str = Field(
        default="", description="Directory containing files to ingest"
    )
    collection_name: str = Field(
        default="documents", description="Name of the Qdrant collection"
    )
    sparse_process_method: Literal["sparse_embedding", "inverted_index"] = Field(
        default="sparse_embedding",
        description="Method for sparse encoding",
    )


class RetrievalQuery(BaseModel):
    queries: list[str] = Field(..., description="The list of query texts")
    collection_name: str = Field(
        default="documents", description="Name of the Qdrant collection"
    )
    top_k: int = Field(
        default=5, description="Number of top similar documents to retrieve"
    )
    mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid", description="The retrieval mode to use"
    )
    sparse_process_method: Literal["sparse_embedding", "inverted_index"] = Field(
        default="sparse_embedding",
        description="Method for sparse encoding",
    )
    overfetch_mul: float = Field(
        default=2.0,
        description="Multiplier for overfetching in hybrid retrieval",
    )
