"""Hierarchical (map-reduce) summarization over the section tree.

  1. map     : every chunk  -> summary + key points + keywords      (parallel)
  2. reduce  : every section -> summary from its chunk summaries and
               its children's section summaries                     (bottom-up, level by level)
  3. reduce  : document -> executive summary, key points, topics, type

Sections with a single chunk and no children reuse the chunk summary (no extra call).
Inputs larger than ``REDUCE_TOKEN_BUDGET`` are reduced in batches first.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel, Field

from .models import Chunk, DocumentMetadata, DocumentSummary, Section, SummarizationOptions
from .textutils import estimate_tokens

REDUCE_TOKEN_BUDGET = 60_000
T = TypeVar("T", bound=BaseModel)

_LENGTH = {
    "brief": {"chunk": "1-2 sentences", "points": "2-3", "section": "2-3 sentences", "doc": "one short paragraph"},
    "standard": {"chunk": "2-4 sentences", "points": "3-5", "section": "one paragraph (4-6 sentences)",
                 "doc": "2-3 paragraphs"},
    "detailed": {"chunk": "4-6 sentences", "points": "4-7", "section": "two paragraphs",
                 "doc": "4-5 paragraphs"},
}

SYSTEM = (
    "You are an expert analyst who writes faithful, information-dense summaries of documents. "
    "Use only facts present in the provided text; never invent numbers, names, or claims. "
    "Preserve important specifics (figures, dates, names, definitions, conclusions). "
    "Write in plain prose without filler such as 'This section discusses'."
)


class LLM(Protocol):
    async def generate_json(self, prompt: str, schema: type[T], system: str,
                            temperature: float = 0.2, max_output_tokens: int = 8192) -> T: ...


class ChunkSummaryOut(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class SectionSummaryOut(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)


class DocumentSummaryOut(BaseModel):
    title: str
    document_type: str = Field(description="e.g. research paper, contract, annual report, manual")
    executive_summary: str
    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    audience: str | None = None


ProgressFn = Callable[[int, int], None]


class Summarizer:
    def __init__(self, llm: LLM, options: SummarizationOptions):
        self.llm = llm
        self.opts = options
        self.len = _LENGTH[options.length]

    def _instructions(self) -> str:
        parts = []
        if self.opts.language:
            parts.append(f"Write the output in {self.opts.language}.")
        else:
            parts.append("Write the output in the same language as the source text.")
        if self.opts.focus:
            parts.append(f"Reader's focus (prioritize this, but stay faithful): {self.opts.focus}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ map

    async def summarize_chunk(self, chunk: Chunk, doc_title: str) -> ChunkSummaryOut:
        location = " > ".join(chunk.heading_path) or "(document start)"
        prompt = (
            f"Document: {doc_title}\nSection: {location}\nPages: {chunk.page_start}-{chunk.page_end}\n\n"
            f"Summarize the passage below in {self.len['chunk']}. Then list {self.len['points']} key points "
            "(each a self-contained fact or claim) and 3-8 keywords/key phrases.\n"
            f"{self._instructions()}\n\n<passage>\n{chunk.text}\n</passage>"
        )
        return await self.llm.generate_json(prompt, ChunkSummaryOut, SYSTEM)

    # ------------------------------------------------------------------ reduce helpers

    async def _compress(self, items: list[str], label: str) -> list[str]:
        """If the material is too large for one call, summarize it in batches first."""
        while estimate_tokens("\n\n".join(items)) > REDUCE_TOKEN_BUDGET and len(items) > 1:
            batches: list[list[str]] = [[]]
            for it in items:
                if batches[-1] and estimate_tokens("\n\n".join(batches[-1] + [it])) > REDUCE_TOKEN_BUDGET // 2:
                    batches.append([])
                batches[-1].append(it)
            new_items = []
            for b in batches:
                out = await self.llm.generate_json(
                    f"Combine these consecutive partial summaries of {label} into one faithful summary "
                    f"that keeps all important specifics.\n{self._instructions()}\n\n" + "\n\n".join(b),
                    SectionSummaryOut, SYSTEM,
                )
                new_items.append(out.summary)
            items = new_items
        return items

    async def summarize_section(self, section: Section, parts: list[str], doc_title: str) -> SectionSummaryOut:
        parts = await self._compress(parts, f'section "{section.title}"')
        prompt = (
            f'Document: {doc_title}\nSection: "{section.title}" (pages {section.page_start}-{section.page_end})\n\n'
            f"Below are summaries of the parts of this section, in order. Write a cohesive summary of the "
            f"whole section in {self.len['section']} and list its {self.len['points']} most important points.\n"
            f"{self._instructions()}\n\n<parts>\n" + "\n\n".join(parts) + "\n</parts>"
        )
        return await self.llm.generate_json(prompt, SectionSummaryOut, SYSTEM)

    async def summarize_document(self, meta: DocumentMetadata, parts: list[str]) -> DocumentSummaryOut:
        parts = await self._compress(parts, "the document")
        prompt = (
            f"File: {meta.filename}\nTitle (from metadata, may be missing): {meta.title or 'unknown'}\n"
            f"Pages: {meta.page_count}\n\n"
            "Below are summaries of the document's parts, in order. Produce:\n"
            "- title: the document's real title (infer if needed)\n"
            "- document_type: what kind of document this is\n"
            f"- executive_summary: {self.len['doc']} covering purpose, main content, and conclusions\n"
            "- key_points: the 5-10 most important takeaways\n"
            "- topics: 3-10 main topics\n"
            "- audience: who the document is written for\n"
            f"{self._instructions()}\n\n<parts>\n" + "\n\n".join(parts) + "\n</parts>"
        )
        return await self.llm.generate_json(prompt, DocumentSummaryOut, SYSTEM, max_output_tokens=16384)


async def summarize_tree(
    llm: LLM,
    options: SummarizationOptions,
    meta: DocumentMetadata,
    sections: list[Section],
    chunks: list[Chunk],
    progress: ProgressFn | None = None,
) -> DocumentSummary:

    s = Summarizer(llm, options)
    title = meta.title or meta.filename
    by_id = {sec.id: sec for sec in sections}
    chunk_by_id = {c.id: c for c in chunks}

    # sections that contain any content in their subtree
    def has_content(sec: Section) -> bool:
        return bool(sec.chunk_ids) or any(has_content(by_id[c]) for c in sec.children)

    content_sections = [sec for sec in sections if sec.level > 0 and has_content(sec)]
    reducible = [sec for sec in content_sections if not (len(sec.chunk_ids) == 1 and not sec.children)]
    total = len(chunks) + len(reducible) + 1
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total)

    # 1) map
    async def do_chunk(c: Chunk) -> None:
        out = await s.summarize_chunk(c, title)
        c.summary, c.key_points, c.keywords = out.summary, out.key_points, out.keywords
        tick()

    await asyncio.gather(*(do_chunk(c) for c in chunks))

    # 2) reduce sections bottom-up, one depth level at a time
    def depth(sec: Section) -> int:
        d = 0
        while sec.parent_id:
            sec = by_id[sec.parent_id]
            d += 1
        return d

    def parts_for(sec: Section) -> list[str]:
        # own chunks come first (they precede subsections in reading order)
        parts = [chunk_by_id[cid].summary or "" for cid in sec.chunk_ids]
        for child_id in sec.children:
            child = by_id[child_id]
            if child.summary:
                parts.append(f"[Subsection: {child.title}]\n{child.summary}")
        return [p for p in parts if p]

    for d in sorted({depth(sec) for sec in content_sections}, reverse=True):
        level_secs = [sec for sec in content_sections if depth(sec) == d]

        async def do_section(sec: Section) -> None:
            if len(sec.chunk_ids) == 1 and not sec.children:
                c = chunk_by_id[sec.chunk_ids[0]]
                sec.summary, sec.key_points = c.summary, list(c.key_points)
                return
            out = await s.summarize_section(sec, parts_for(sec), title)
            sec.summary, sec.key_points = out.summary, out.key_points
            tick()

        await asyncio.gather(*(do_section(sec) for sec in level_secs))

    # 3) document
    root = sections[0]
    doc_parts = parts_for(root)
    if not doc_parts:
        doc_parts = [c.summary for c in chunks if c.summary]
    out = await s.summarize_document(meta, doc_parts)
    root.summary, root.key_points = out.executive_summary, out.key_points
    tick()
    return DocumentSummary(**out.model_dump())
