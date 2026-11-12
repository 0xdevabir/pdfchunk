"""Export processed documents to Markdown, JSONL (RAG-ready) and PDF."""

from __future__ import annotations

import json
import textwrap
from io import BytesIO

import pymupdf

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


def to_pdf(doc: ProcessedDocument) -> bytes:
    """Render a clean printable PDF of the document summary + outline."""
    title = (doc.summary.title if doc.summary else None) or doc.metadata.title or doc.metadata.filename
    blocks: list[tuple[str, str]] = []  # (style, text)

    blocks.append(("title", title))
    blocks.append((
        "meta",
        f"Source: {doc.metadata.filename}  ·  {doc.metadata.page_count} pages  ·  "
        f"{doc.stats.chunk_count} chunks  ·  {doc.stats.total_tokens:,} tokens",
    ))

    if doc.summary:
        s = doc.summary
        meta_bits = [f"Type: {s.document_type}"]
        if s.audience:
            meta_bits.append(f"Audience: {s.audience}")
        blocks.append(("meta", "  ·  ".join(meta_bits)))
        blocks.append(("h1", "Executive summary"))
        blocks.append(("body", s.executive_summary))
        if s.key_points:
            blocks.append(("h1", "Key points"))
            for p in s.key_points:
                blocks.append(("bullet", p))
        if s.topics:
            blocks.append(("h1", "Topics"))
            blocks.append(("body", ", ".join(s.topics)))

    by_id = {sec.id: sec for sec in doc.sections}
    if doc.sections:
        blocks.append(("h1", "Outline"))

        def walk(sec_id: str) -> None:
            sec = by_id[sec_id]
            if sec.level > 0:
                blocks.append(("h2" if sec.level == 1 else "h3",
                               f"{sec.title}  (p. {sec.page_start}–{sec.page_end})"))
                if sec.summary:
                    blocks.append(("body", sec.summary))
                if sec.key_points:
                    for p in sec.key_points:
                        blocks.append(("bullet", p))
            for child in sec.children:
                walk(child)

        walk(doc.sections[0].id)

    return _render_pdf(title, blocks)


def _render_pdf(title: str, blocks: list[tuple[str, str]]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4
    margin_x, margin_y = 54, 54
    width = page.rect.width - 2 * margin_x
    y = margin_y
    fonts = {
        "title": ("helv", 18, True),
        "h1": ("helv", 13, True),
        "h2": ("helv", 11.5, True),
        "h3": ("helv", 10.5, True),
        "meta": ("helv", 8.5, False),
        "body": ("helv", 10, False),
        "bullet": ("helv", 10, False),
    }

    def new_page() -> None:
        nonlocal page, y
        page = doc.new_page(width=595, height=842)
        y = margin_y

    for style, text in blocks:
        font, size, bold = fonts[style]
        prefix = "• " if style == "bullet" else ""
        wrap_w = 95 if style == "bullet" else 100
        lines = []
        for para in (text or "").split("\n"):
            wrapped = textwrap.wrap(prefix + para, width=wrap_w, replace_whitespace=False) or [prefix]
            lines.extend(wrapped)
            prefix = "  " if style == "bullet" else ""  # indent continuations

        line_h = size * 1.35
        gap_before = {"title": 0, "h1": 14, "h2": 10, "h3": 8, "meta": 4, "body": 6, "bullet": 3}[style]
        if y + gap_before + line_h > page.rect.height - margin_y:
            new_page()
        else:
            y += gap_before

        color = (0.25, 0.28, 0.35) if style == "meta" else (0.08, 0.09, 0.12)
        for line in lines:
            if y + line_h > page.rect.height - margin_y:
                new_page()
            page.insert_text(
                (margin_x, y + size),
                line,
                fontname=font,
                fontsize=size,
                color=color,
            )
            # bold simulation via slight overstrike (PyMuPDF helv has no bold variant by name alone)
            if bold:
                page.insert_text((margin_x + 0.25, y + size), line, fontname=font, fontsize=size, color=color)
            y += line_h
        if style in ("title", "h1", "body"):
            y += 4

    # footer with page numbers
    for i, p in enumerate(doc, start=1):
        p.insert_text(
            (p.rect.width / 2 - 10, p.rect.height - 28),
            f"{i}",
            fontname="helv",
            fontsize=8,
            color=(0.45, 0.48, 0.55),
        )

    buf = BytesIO()
    doc.save(buf, garbage=3, deflate=True)
    doc.close()
    return buf.getvalue()
