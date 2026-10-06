from pydantic import BaseModel, Field
from typing import List, Optional


class InitializeResponse(BaseModel):
    message: str
    documents_count: int
    files_processed: int


class UploadResponse(BaseModel):
    message: str
    file_id: str
    filename: str
    documents_count: int


class SourceNode(BaseModel):
    text: str
    score: Optional[float] = None
    source_file: Optional[str] = None
    page_number: Optional[str] = None


class GroundingClaim(BaseModel):
    claim: str
    score: float
    supported: bool


class GroundingResult(BaseModel):
    passed: bool
    faithfulness: float
    num_claims: int
    flagged: List[str] = []
    claims: List[GroundingClaim] = []


class QueryResponse(BaseModel):
    query: str
    response: str
    source_nodes: List[SourceNode] = []
    grounding: Optional[GroundingResult] = None


class DocumentChunk(BaseModel):
    chunk_id: int
    text: str
    metadata: dict = Field(default_factory=dict)


class DocumentContentResponse(BaseModel):
    doc_id: str
    chunks: List[DocumentChunk]


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    message: str
    version: str
    description: str
    index_initialized: bool


class DeleteResponse(BaseModel):
    message: str
    doc_id: str


class StatsResponse(BaseModel):
    vector_count: int
    index_initialized: bool


class DocumentListItem(BaseModel):
    id: str
    filename: str
    size: int
    upload_time: float
    source: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int


class TopicItem(BaseModel):
    topic: str
    count: int


class TopicListResponse(BaseModel):
    topics: List[TopicItem]
    total: int
