from pydantic import BaseModel, Field


class PostingEntry(BaseModel):
    doc_id: str = Field(..., description="Unique identifier of the document")
    term_freq: int = Field(..., description="Frequency of the term in the document")


class TermEntry(BaseModel):
    doc_freq: int = Field(..., description="Number of documents containing the term")
    postings: list[PostingEntry] = Field(
        ..., description="List of postings for the term"
    )
