from __future__ import annotations

import asyncio
import hashlib
import time

import pytest

from pdfchunk.chunking import chunk_document
from pdfchunk.exporters import to_jsonl, to_markdown, to_pdf
from pdfchunk.extraction import extract_pdf
from pdfchunk.models import (
    ChunkingOptions, ElementType, ProcessingOptions, SummarizationOptions,
)
from pdfchunk.pipeline import process_pdf
from pdfchunk.rag import retrieve
from pdfchunk.summarizer import ChunkSummaryOut, DocumentSummaryOut, SectionSummaryOut
from pdfchunk.textutils import estimate_tokens, split_sentences

from .sample_pdf import build


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return build()


@pytest.fixture(scope="module")
def extracted(pdf_bytes):
    return extract_pdf(pdf_bytes, "sample.pdf")


class FakeLLM:
    """Deterministic stand-in for GeminiClient."""

    def __init__(self):
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    async def generate_json(self, prompt, schema, system, temperature=0.2, max_output_tokens=8192):
        self.calls += 1
        self.prompts.append((schema.__name__, prompt))
        if schema is ChunkSummaryOut:
            return ChunkSummaryOut(summary=f"chunk summary {self.calls}", key_points=["kp"], keywords=["kw"])
        if schema is SectionSummaryOut:
            return SectionSummaryOut(summary=f"section summary {self.calls}", key_points=["skp"])
        return DocumentSummaryOut(title="Renewable Energy Market Review 2025", document_type="report",
                                  executive_summary="exec", key_points=["a"], topics=["solar"])

    async def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            # topic-ish vectors: battery text points one way, everything else another
            base = [1.0, 0.0] if "batter" in t.lower() else [0.0, 1.0]
            h = hashlib.md5(t.encode()).digest()
            out.append(base + [b / 2550 for b in h[:6]])
        return out


# ------------------------------------------------------------------ text utils


def test_sentence_splitter_handles_abbreviations():
    s = split_sentences("Growth was 4.5 percent, e.g. in Texas. Dr. Smith agreed. Then it fell!")
    assert s == ["Growth was 4.5 percent, e.g. in Texas.", "Dr. Smith agreed.", "Then it fell!"]


def test_estimate_tokens_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)


# ------------------------------------------------------------------ extraction


def test_extraction_structure(extracted):
    md = extracted.metadata
    assert md.page_count == 2
    assert md.title == "Renewable Energy Market Review 2025"  # detected from layout
    assert md.removed_header_footer_lines >= 2
    assert md.tables_found == 1

    texts = [e.text for e in extracted.elements]
    assert not any("Confidential" in t for t in texts), "running header should be removed"
    headings = [(e.text, e.level) for e in extracted.elements if e.type == ElementType.heading]
    assert headings == [
        ("1 Introduction", 1), ("2 Solar Power", 1), ("2.1 Utility-Scale Solar", 2),
        ("2.2 Distributed Solar", 2), ("3 Energy Storage", 1), ("4 Outlook", 1),
    ]
    assert sum(e.type == ElementType.list_item for e in extracted.elements) == 3
    table = next(e for e in extracted.elements if e.type == ElementType.table)
    assert "| Lithium iron phosphate | 42 | +110% |" in table.text


def test_rejects_non_pdf():
    with pytest.raises(Exception):
        extract_pdf(b"not a pdf at all", "x.pdf")


# ------------------------------------------------------------------ chunking


def _chunks(extracted, **kw):
    return asyncio.run(chunk_document(extracted, ChunkingOptions(**kw)))


def test_structural_chunks_respect_sections_and_budget(extracted):
    sections, chunks = _chunks(extracted, max_tokens=200, min_tokens=40, overlap_tokens=30)
    by_id = {s.id: s for s in sections}
    assert chunks, "expected chunks"
    for c in chunks:
        assert c.token_count <= int(200 * 1.25), c
        assert c.section_id in by_id
        assert c.id in by_id[c.section_id].chunk_ids
    # no heading-only chunks
    assert all(len(c.text) > 40 for c in chunks)
    # the empty-bodied "2 Solar Power" heading is carried into its first subsection
    assert chunks[2].text.startswith("# 2 Solar Power\n\n## 2.1 Utility-Scale Solar")
    # every word of the document survives chunking
    all_text = " ".join(c.text for c in chunks)
    for e in extracted.elements:
        if e.type == ElementType.paragraph:
            assert e.text.split()[0] in all_text
    # tree token roll-up
    assert sections[0].token_count == sum(c.token_count for c in chunks)


def test_overlap_between_consecutive_chunks(extracted):
    _, chunks = _chunks(extracted, max_tokens=150, min_tokens=20, overlap_tokens=60)
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.section_id == b.section_id]
    assert pairs, "expected a section split into several chunks"
    assert any(split_sentences(a.text)[-1] in b.text for a, b in pairs)


def test_fixed_strategy(extracted):
    _, chunks = _chunks(extracted, strategy="fixed", max_tokens=150, overlap_tokens=30)
    assert len(chunks) >= 5
    assert all(c.token_count <= 190 for c in chunks)


def test_semantic_strategy_uses_embeddings(extracted):
    llm = FakeLLM()
    _, chunks = asyncio.run(chunk_document(
        extracted, ChunkingOptions(strategy="semantic", max_tokens=200, min_tokens=30), embed=llm.embed))
    assert llm.calls >= 1
    assert chunks and all(c.token_count <= 260 for c in chunks)


def test_semantic_requires_embedder(extracted):
    with pytest.raises(ValueError):
        _chunks(extracted, strategy="semantic")


# ------------------------------------------------------------------ summarization


def test_hierarchical_summarization(pdf_bytes):
    llm = FakeLLM()
    opts = ProcessingOptions(chunking=ChunkingOptions(max_tokens=200, min_tokens=40, overlap_tokens=30),
                             summarization=SummarizationOptions(length="brief", focus="storage"))
    events = []
    doc = asyncio.run(process_pdf(pdf_bytes, opts, filename="sample.pdf", llm=llm,
                                  progress=lambda st, f, m: events.append((st, f))))
    assert doc.summary and doc.summary.executive_summary == "exec"
    assert all(c.summary for c in doc.chunks)
    content_sections = [s for s in doc.sections if s.level > 0]
    assert all(s.summary for s in content_sections)
    # "2 Solar Power" has subsections -> its summary is built from their summaries
    solar = next(s for s in doc.sections if s.title == "2 Solar Power")
    solar_prompt = next(p for name, p in llm.prompts if name == "SectionSummaryOut" and "2 Solar Power" in p)
    assert "[Subsection: 2.1 Utility-Scale Solar]" in solar_prompt
    assert solar.summary.startswith("section summary")
    assert all("storage" in p for _, p in llm.prompts)  # focus instruction propagated
    # progress is monotonic and finishes at 1.0
    fracs = [f for _, f in events]
    assert fracs == sorted(fracs) and fracs[-1] == 1.0
    assert doc.stats.llm_calls == llm.calls

    md = to_markdown(doc)
    assert "## Executive summary" in md and "2.1 Utility-Scale Solar" in md
    assert len(to_jsonl(doc).strip().splitlines()) == len(doc.chunks)
    pdf_out = to_pdf(doc)
    assert pdf_out.startswith(b"%PDF") and len(pdf_out) > 500

    hits = asyncio.run(retrieve(doc, "battery storage lithium", top_k=3))
    assert hits and any("stor" in h[0].text.lower() or "batter" in h[0].text.lower() for h in hits)


def test_requires_key_when_summarizing(pdf_bytes, monkeypatch):
    from pdfchunk.config import Settings
    cfg = Settings(gemini_api_key=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        asyncio.run(process_pdf(pdf_bytes, ProcessingOptions(), cfg=cfg))


# ------------------------------------------------------------------ API


def test_api_chunk_only_flow(pdf_bytes, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pdfchunk.api as api
    from pdfchunk.store import Store

    monkeypatch.setattr(api, "store", Store(tmp_path))
    with TestClient(api.app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.post("/api/documents", files={"file": ("x.txt", b"hello", "text/plain")},
                           data={"options": '{"summarization": {"enabled": false}}'}).status_code == 415
        r = client.post(
            "/api/documents",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
            data={"options": '{"summarization": {"enabled": false}, "chunking": {"max_tokens": 300}}'},
        )
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "completed", job
        doc = client.get(f"/api/documents/{job_id}").json()
        assert doc["stats"]["chunk_count"] == len(doc["chunks"]) > 0
        assert client.get(f"/api/documents/{job_id}/export?format=jsonl").status_code == 200
        assert client.get(f"/api/documents/{job_id}/export?format=markdown").text.startswith("# ")
        pdf_export = client.get(f"/api/documents/{job_id}/export?format=pdf")
        assert pdf_export.status_code == 200 and pdf_export.content.startswith(b"%PDF")

        chunk_id = doc["chunks"][0]["id"]
        patched = client.patch(
            f"/api/documents/{job_id}/chunks/{chunk_id}",
            json={"summary": "edited summary", "text": doc["chunks"][0]["text"] + "\n\nEdited."},
        )
        assert patched.status_code == 200
        assert patched.json()["chunks"][0]["summary"] == "edited summary"

        batch = client.post(
            "/api/documents/batch",
            files=[("files", ("a.pdf", pdf_bytes, "application/pdf")),
                   ("files", ("b.pdf", pdf_bytes, "application/pdf"))],
            data={"options": '{"summarization": {"enabled": false}, "chunking": {"max_tokens": 300}}'},
        )
        assert batch.status_code == 202 and len(batch.json()) == 2

        assert client.delete(f"/api/documents/{job_id}").status_code == 204
        assert client.get(f"/api/documents/{job_id}").status_code == 404
