"""Export processed documents to Markdown and JSONL (RAG-ready)."""

from __future__ import annotations

import json

from .models import ProcessedDocument


def to_markdown(doc: ProcessedDocument, include_chunks: bool = True) -> str:
    out: list[str] = []
    title = (doc.summary.title if doc.summary else None) or doc.metadata.title or doc.metadata.filename
    out.append(f"# {title}\n")
    out.append(f"*Source:* `{doc.metadata.filename}` · {doc.metadata.page_count} pages · "
               f"{doc.stats.chunk_count} chunks · {doc.stats.total_tokens:,} tokens\n")

    if doc.summary:
        s = doc.summary
        out.append(f"**Type:** {s.document_type}" + (f" · **Audience:** {s.audience}" if s.audience else "") + "\n")
        out.append("## Executive summary\n")
        out.append(s.executive_summary + "\n")
        if s.key_points:
            out.append("### Key points\n")
            out.extend(f"- {p}" for p in s.key_points)
            out.append("")
        if s.topics:
            out.append("**Topics:** " + ", ".join(s.topics) + "\n")

    by_id = {s.id: s for s in doc.sections}
    chunk_by_id = {c.id: c for c in doc.chunks}

    def walk(sec_id: str) -> None:
        sec = by_id[sec_id]
        if sec.level > 0:
            out.append(f"{'#' * min(sec.level + 1, 6)} {sec.title}  _(p. {sec.page_start}–{sec.page_end})_\n")
            if sec.summary:
                out.append(f"> {sec.summary}\n")
        if include_chunks:
            for cid in sec.chunk_ids:
                c = chunk_by_id[cid]
                out.append(f"<details><summary>Chunk {c.index} · p. {c.page_start}–{c.page_end} · "
                           f"{c.token_count} tokens</summary>\n")
                if c.summary:
                    out.append(f"**Summary:** {c.summary}\n")
                out.append(c.text + "\n\n</details>\n")
        for child in sec.children:
            walk(child)

    if doc.sections:
        out.append("## Outline\n")
        walk(doc.sections[0].id)
    return "\n".join(out).rstrip() + "\n"


def to_jsonl(doc: ProcessedDocument) -> str:
    """One line per chunk, with breadcrumb-prefixed text for embedding pipelines."""
    lines = []
    for c in doc.chunks:
        lines.append(json.dumps({
            "id": f"{doc.id}:{c.id}",
            "document_id": doc.id,
            "source": doc.metadata.filename,
            "text": c.text,
            "contextual_text": c.contextual_text,
            "summary": c.summary,
            "keywords": c.keywords,
            "heading_path": c.heading_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "token_count": c.token_count,
        }, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")
