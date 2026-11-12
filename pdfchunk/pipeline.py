"""End-to-end orchestration: extract -> chunk -> summarize."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Callable

from .chunking import chunk_document
from .config import Settings, settings as default_settings
from .extraction import extract_pdf
from .gemini import GeminiClient
from .models import ProcessedDocument, ProcessingOptions, ProcessingStats
from .store import now_iso
from .summarizer import LLM, summarize_tree

# progress(stage, fraction 0..1, message)
ProgressFn = Callable[[str, float, str], None]

# Overall progress budget per stage
_EXTRACT, _CHUNK = 0.15, 0.10


async def process_pdf(
    source: str | Path | bytes,
    options: ProcessingOptions | None = None,
    filename: str | None = None,
    doc_id: str | None = None,
    progress: ProgressFn | None = None,
    llm: LLM | GeminiClient | None = None,
    cfg: Settings = default_settings,
) -> ProcessedDocument:
    options = options or ProcessingOptions()
    report = progress or (lambda *_: None)
    needs_llm = options.summarization.enabled or options.chunking.strategy == "semantic"
    if needs_llm and llm is None:
        if not cfg.gemini_enabled:
            raise RuntimeError(
                "Summarization / semantic chunking need a Gemini API key. Set GEMINI_API_KEY, "
                "or disable summarization and use the 'structural' or 'fixed' strategy."
            )
        llm = GeminiClient(cfg)

    stats = ProcessingStats()

    report("extracting", 0.02, "Reading PDF layout")
    t0 = time.perf_counter()
    extracted = await asyncio.to_thread(extract_pdf, source, filename, options.detect_tables)
    stats.extraction_seconds = round(time.perf_counter() - t0, 3)
    report("chunking", _EXTRACT, f"Extracted {len(extracted.elements)} elements from "
                                 f"{extracted.metadata.page_count} pages")

    t0 = time.perf_counter()
    embed = llm.embed if options.chunking.strategy == "semantic" and hasattr(llm, "embed") else None
    sections, chunks = await chunk_document(extracted, options.chunking, embed=embed)
    stats.chunking_seconds = round(time.perf_counter() - t0, 3)
    stats.chunk_count = len(chunks)
    stats.section_count = sum(1 for s in sections if s.level > 0)
    stats.total_tokens = sum(c.token_count for c in chunks)
    report("summarizing" if options.summarization.enabled else "finalizing", _EXTRACT + _CHUNK,
           f"Created {len(chunks)} chunks in {stats.section_count} sections")

    summary = None
    warnings = list(extracted.warnings)
    if options.summarization.enabled and chunks:
        t0 = time.perf_counter()
        base = _EXTRACT + _CHUNK

        def on_progress(done: int, total: int) -> None:
            report("summarizing", base + (1 - base - 0.02) * done / max(total, 1),
                   f"Summarizing ({done}/{total} Gemini calls)")

        summary = await summarize_tree(llm, options.summarization, extracted.metadata, sections, chunks,
                                       on_progress)
        stats.summarization_seconds = round(time.perf_counter() - t0, 3)
    elif options.summarization.enabled:
        warnings.append("Nothing to summarize: no text chunks were produced.")

    if llm is not None:
        stats.llm_calls = getattr(llm, "calls", 0)
        stats.llm_input_tokens = getattr(llm, "input_tokens", 0)
        stats.llm_output_tokens = getattr(llm, "output_tokens", 0)

    report("completed", 1.0, "Done")
    return ProcessedDocument(
        id=doc_id or uuid.uuid4().hex[:12],
        created_at=now_iso(),
        metadata=extracted.metadata,
        options=options,
        sections=sections,
        chunks=chunks,
        summary=summary,
        stats=stats,
        warnings=warnings,
    )
