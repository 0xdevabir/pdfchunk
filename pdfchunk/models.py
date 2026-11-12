"""Data models shared across the pipeline, API and CLI."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- extraction


class ElementType(str, Enum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"
    list_item = "list_item"


class Element(BaseModel):
    """A logical piece of content extracted from the PDF, in reading order."""

    type: ElementType
    text: str
    page: int  # 1-based
    level: int | None = None  # heading level (1 = top)
    font_size: float | None = None
    bbox: tuple[float, float, float, float] | None = None


class DocumentMetadata(BaseModel):
    filename: str
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    page_count: int
    has_outline: bool = False
    scanned_pages: list[int] = Field(default_factory=list)
    removed_header_footer_lines: int = 0
    tables_found: int = 0


class ExtractedDocument(BaseModel):
    metadata: DocumentMetadata
    elements: list[Element]
    body_font_size: float | None = None
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- chunking

ChunkStrategy = Literal["structural", "fixed", "semantic"]
SummaryLength = Literal["brief", "standard", "detailed"]


class ChunkingOptions(BaseModel):
    strategy: ChunkStrategy = "structural"
    max_tokens: int = Field(default=800, ge=100, le=8000)
    min_tokens: int = Field(default=120, ge=0, le=4000)
    overlap_tokens: int = Field(default=80, ge=0, le=2000)
    # semantic strategy: percentile of adjacent-sentence distance above which we split
    semantic_breakpoint_percentile: float = Field(default=90.0, ge=50.0, le=99.9)


class Chunk(BaseModel):
    id: str
    index: int
    text: str
    token_count: int
    page_start: int
    page_end: int
    section_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    element_types: list[ElementType] = Field(default_factory=list)
    # filled by summarizer
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @property
    def contextual_text(self) -> str:
        """Text prefixed with its section breadcrumb – handy for embeddings / RAG."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n\n" + self.text


class Section(BaseModel):
    id: str
    title: str
    level: int  # 0 = synthetic root / preamble
    page_start: int
    page_end: int
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    token_count: int = 0
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- summarization


class SummarizationOptions(BaseModel):
    enabled: bool = True
    length: SummaryLength = "standard"
    language: str | None = None  # e.g. "English"; None = same as the document
    focus: str | None = None  # optional user instruction: "focus on financial risks"


class DocumentSummary(BaseModel):
    title: str
    document_type: str
    executive_summary: str
    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    audience: str | None = None


class ProcessingOptions(BaseModel):
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    summarization: SummarizationOptions = Field(default_factory=SummarizationOptions)
    detect_tables: bool = True


class ProcessingStats(BaseModel):
    extraction_seconds: float = 0.0
    chunking_seconds: float = 0.0
    summarization_seconds: float = 0.0
    total_tokens: int = 0
    chunk_count: int = 0
    section_count: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


class ProcessedDocument(BaseModel):
    id: str
    created_at: str
    metadata: DocumentMetadata
    options: ProcessingOptions
    sections: list[Section]
    chunks: list[Chunk]
    summary: DocumentSummary | None = None
    stats: ProcessingStats = Field(default_factory=ProcessingStats)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- jobs


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    id: str
    filename: str
    status: JobStatus = JobStatus.queued
    stage: str = "queued"
    progress: float = 0.0  # 0..1
    message: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ChunkUpdate(BaseModel):
    """Partial update for inspect/edit workflow (option B)."""

    text: str | None = None
    summary: str | None = None
    key_points: list[str] | None = None
    keywords: list[str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
