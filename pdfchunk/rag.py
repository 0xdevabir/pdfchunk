"""Simple retrieval-augmented Q&A over processed document chunks.

Uses Gemini embeddings for ranking when available; falls back to lexical
overlap scoring so the endpoint still works without embeddings.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from pydantic import BaseModel, Field

from .gemini import GeminiClient
from .models import Chunk, ProcessedDocument
from .textutils import estimate_tokens

SYSTEM = (
    "You are a careful document assistant. Answer ONLY from the provided passages. "
    "Cite page numbers like (p. 12) when you use a fact. If the passages do not "
    "contain enough information, say so clearly instead of guessing."
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


class AnswerOut(BaseModel):
    answer: str
    cited_chunk_indexes: list[int] = Field(default_factory=list)


class AskResult(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lexical_score(query: str, chunk: Chunk) -> float:
    q = set(_tokenize(query))
    if not q:
        return 0.0
    blob = " ".join([chunk.text, chunk.summary or "", " ".join(chunk.keywords), " ".join(chunk.heading_path)])
    t = set(_tokenize(blob))
    if not t:
        return 0.0
    overlap = len(q & t) / len(q)
    # lightly boost chunks whose summary already mentions the query terms
    summary_bonus = 0.15 * (len(q & set(_tokenize(chunk.summary or ""))) / len(q))
    return overlap + summary_bonus


async def retrieve(
    doc: ProcessedDocument,
    question: str,
    top_k: int = 6,
    embedder: Embedder | None = None,
) -> list[tuple[Chunk, float]]:
    chunks = doc.chunks
    if not chunks:
        return []

    scores: list[float]
    if embedder is not None:
        texts = [c.contextual_text[:6000] for c in chunks]
        vectors = await embedder.embed([question] + texts)
        qv, cvs = vectors[0], vectors[1:]
        scores = [_cosine(qv, cv) for cv in cvs]
    else:
        scores = [_lexical_score(question, c) for c in chunks]

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    # drop zero-score noise for lexical mode, keep at least one if everything is zero
    positive = [(c, s) for c, s in ranked if s > 0]
    pool = positive or ranked[:1]
    return pool[: max(1, top_k)]


def _format_passages(hits: list[tuple[Chunk, float]]) -> str:
    parts = []
    for c, score in hits:
        path = " > ".join(c.heading_path) or "(document start)"
        parts.append(
            f"[Chunk #{c.index} | pages {c.page_start}-{c.page_end} | {path} | score={score:.3f}]\n"
            f"{c.text}"
        )
    return "\n\n".join(parts)


async def ask_document(
    doc: ProcessedDocument,
    question: str,
    llm: GeminiClient,
    top_k: int = 6,
) -> AskResult:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not doc.chunks:
        return AskResult(answer="This document has no text chunks to search.", citations=[])

    hits = await retrieve(doc, question, top_k=top_k, embedder=llm)
    # pack within a safe context budget
    packed: list[tuple[Chunk, float]] = []
    used = 0
    for c, s in hits:
        t = estimate_tokens(c.text)
        if packed and used + t > 12_000:
            break
        packed.append((c, s))
        used += t

    passages = _format_passages(packed)
    title = (doc.summary.title if doc.summary else None) or doc.metadata.title or doc.metadata.filename
    prompt = (
        f'Document: "{title}" ({doc.metadata.filename}, {doc.metadata.page_count} pages)\n\n'
        f"Question: {question}\n\n"
        "Using only the passages below, write a clear answer. List the chunk indexes you relied on "
        "in cited_chunk_indexes.\n\n"
        f"<passages>\n{passages}\n</passages>"
    )
    out = await llm.generate_json(prompt, AnswerOut, SYSTEM, temperature=0.1, max_output_tokens=4096)

    by_index = {c.index: c for c, _ in packed}
    citations = []
    for idx in out.cited_chunk_indexes:
        c = by_index.get(idx)
        if not c:
            continue
        citations.append({
            "chunk_id": c.id,
            "index": c.index,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "heading_path": c.heading_path,
            "excerpt": (c.summary or c.text)[:280],
        })
    if not citations:
        for c, _ in packed[:3]:
            citations.append({
                "chunk_id": c.id,
                "index": c.index,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "heading_path": c.heading_path,
                "excerpt": (c.summary or c.text)[:280],
            })

    return AskResult(
        answer=out.answer,
        citations=citations,
        retrieved_chunk_ids=[c.id for c, _ in packed],
    )
