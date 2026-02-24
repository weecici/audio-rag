from pydantic import BaseModel, Field
from typing import Union


class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="The ID of the document")
    title: str = Field(..., description="The title of the document")
    file_name: str = Field(..., description="The name of the file")
    file_path: str = Field(..., description="The path to the file")


class DocumentPayload(BaseModel):
    text: str = Field(..., description="The text content of the document")
    metadata: DocumentMetadata = Field(..., description="The metadata of the document")


class RetrievedDocument(BaseModel):
    id: Union[int, str] = Field(..., description="The ID of the retrieved document")
    score: float = Field(
        ..., description="The similarity score of the document with the query"
    )
    payload: DocumentPayload = Field(..., description="The payload of the document")
