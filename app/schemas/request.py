from pydantic import BaseModel, Field
from typing import Literal


class DocumentIngestionRequest(BaseModel):
    file_paths: list[str] = Field(
        default=[], description="List of file paths to ingest documents from"
    )
    file_dir: str = Field(
        default="", description="Directory containing files to ingest"
    )
    collection_name: str = Field(
        default="documents", description="Name of the collection"
    )


class AudioIngestionRequest(BaseModel):
    file_paths: list[str] = Field(
        default=[], description="List of file paths to ingest audio files from"
    )
    urls: list[str] = Field(
        default=[], description="List of YouTube URLs to ingest audio sources from"
    )
    collection_name: str = Field(
        default="documents", description="Name of the collection"
    )


class RetrievalRequest(BaseModel):
    queries: list[str] = Field(..., description="The list of query texts")
    collection_name: str = Field(
        default="documents", description="Name of the collection"
    )
    top_k: int = Field(
        default=5, description="Number of top similar documents to retrieve"
    )
    mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid", description="The retrieval mode to use"
    )
    overfetch_mul: float = Field(
        default=2.0,
        description="Multiplier for overfetching in hybrid retrieval",
    )
    rerank_enabled: bool = Field(
        default=False, description="Whether to rerank retrieved results"
    )


class GenerationRequest(RetrievalRequest):
    model_name: Literal[
        "gpt-oss-120b",
        "llama-3.3-70b",
        "qwen-3-235b-a22b-thinking-2507",
        "qwen-3-coder-480b",
    ] = Field(
        default="gpt-oss-120b",
        description="Name of the LLM to use for generation",
    )
    summarization_enabled: bool = Field(
        default=False, description="Whether to perform document summarization"
    )
