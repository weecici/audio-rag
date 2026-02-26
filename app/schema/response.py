from pydantic import BaseModel, Field
from .doc import Document


class IngestionResponse(BaseModel):
    status: int = Field(..., description="HTTP status code of the ingestion process")
    message: str = Field(..., description="Detailed message about the ingestion")


class RetrievalResponse(BaseModel):
    status: int = Field(..., description="HTTP status code of the retrieval process")
    results: list[list[Document]] = Field(
        ..., description="List of retrieved documents with their metadata"
    )


class GenerationResponse(BaseModel):
    status: int = Field(..., description="HTTP status code of the generation process")
    responses: list[str] = Field(
        ..., description="List of generated responses corresponding to the queries"
    )
    summarized_docs_list: list[list[Document]] = Field(
        ...,
        description="List of summarizations from retrieved documents with their metadata",
    )
