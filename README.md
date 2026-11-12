# pdfchunk: PDF Intelligent Chunking & Summarization

This tool splits PDFs along their real structure (headings, sections, lists and tables) into retrieval-ready chunks. It then uses **Google Gemini** to write hierarchical summaries at three levels: each chunk, each section and the whole document.

```
PDF ──► 1. Extract ──► 2. Structure ──► 3. Chunk ──► 4. Summarize (Gemini) ──► JSON / JSONL / Markdown / Web UI
        layout+fonts    section tree    3 strategies   chunk → section → document
```

## Features

| Stage | What it does |
|---|---|
| **Extraction** (PyMuPDF) | Reads font statistics for each line. Detects tables and renders them as Markdown. Removes repeated headers, footers and page numbers. Rejoins hyphenated words, detects list items and merges paragraphs that break across pages. Flags scanned pages that have no text layer. |
| **Structure** | Uses the PDF's embedded outline (TOC) when it has one. Otherwise it finds headings from font-size tiers, bold text and section numbering (`2.3.1 …`). Detects the document title. Builds a section tree. |
| **Chunking** | `structural` (default): chunks never cross a section boundary, split at sentence boundaries, stay within the token budget, overlap their neighbours and merge small leftovers. `semantic`: splits long sections where the topic shifts, using Gemini embeddings. `fixed`: a sliding-window baseline. Every chunk records its heading path, page range and token count. |
| **Summarization** (Gemini) | Map-reduce over the section tree: chunk summaries with key points and keywords, then section summaries built bottom-up from child summaries, then an executive summary, document type, topics and audience. Uses structured JSON output, limits concurrent requests and retries 429/5xx errors with exponential backoff. Supports summary length (brief/standard/detailed), output language and a *focus* instruction. |
| **Interfaces** | REST API with background jobs and progress, a web UI, a CLI, and Markdown/JSON/JSONL export. The JSONL export is RAG-ready and includes `contextual_text` with the heading breadcrumbs. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then set GEMINI_API_KEY (https://aistudio.google.com/apikey)
```

You don't need a key to extract and chunk with the `structural` or `fixed` strategy. Summarization and the `semantic` strategy require `GEMINI_API_KEY`.

## Run the web app

```bash
uvicorn pdfchunk.api:app --reload
# open http://127.0.0.1:8000
```

Upload a PDF, pick a strategy and summary options, and watch the progress. Then browse the **Summary**, the **Outline** (section tree with summaries) and the **Chunks** (searchable). From there you can export Markdown, JSON or JSONL.

## CLI

```bash
python -m pdfchunk report.pdf                                   # Markdown summary to stdout
python -m pdfchunk report.pdf -f json -o report.json            # full result
python -m pdfchunk report.pdf --no-summary -f jsonl -o chunks.jsonl   # chunk only, no API calls
python -m pdfchunk report.pdf -s semantic --max-tokens 600 --length detailed --focus "risks" --language English
```

## REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Server status, and whether Gemini is configured |
| `POST` | `/api/documents` | multipart: `file` (PDF) and optional `options` (JSON, see below). Returns a job with status `202` |
| `GET` | `/api/jobs` / `/api/jobs/{id}` | Job status: `stage`, `progress` (0–1), `message` and `error` |
| `GET` | `/api/documents/{id}` | Full processed result: metadata, sections, chunks, summary and stats |
| `GET` | `/api/documents/{id}/export?format=markdown\|json\|jsonl` | Download the result |
| `GET` | `/api/documents/{id}/pdf` | The original PDF |
| `DELETE` | `/api/documents/{id}` | Delete the upload and its results |

`options` (every field is optional):

```json
{
  "detect_tables": true,
  "chunking": {"strategy": "structural", "max_tokens": 800, "min_tokens": 120, "overlap_tokens": 80,
               "semantic_breakpoint_percentile": 90},
  "summarization": {"enabled": true, "length": "standard", "language": null, "focus": null}
}
```

```bash
curl -F file=@report.pdf -F 'options={"chunking":{"max_tokens":600}}' localhost:8000/api/documents
```

## Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | none | Required for summaries and semantic chunking. `GOOGLE_API_KEY` also works |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any Gemini model that supports structured JSON output |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Used by the `semantic` strategy |
| `GEMINI_CONCURRENCY` | `4` | Maximum parallel Gemini requests. Lower it if the free tier returns 429 errors |
| `GEMINI_MAX_RETRIES` | `5` | Retries for 429/5xx errors |
| `PDFCHUNK_DATA_DIR` | `data` | Uploads, jobs and results (JSON files) |
| `PDFCHUNK_MAX_UPLOAD_MB` | `100` | |

## Project layout

```
pdfchunk/
  extraction.py   PDF → ordered elements (headings w/ levels, paragraphs, lists, tables)
  chunking.py     section tree + structural / semantic / fixed strategies
  gemini.py       async Gemini client: retries, concurrency, structured output, embeddings
  summarizer.py   hierarchical map-reduce summarization
  pipeline.py     orchestration + progress reporting
  api.py          FastAPI app (serves frontend/)
  cli.py          command-line interface
  exporters.py    Markdown / JSONL export
  store.py        file-backed job & result storage
  models.py       Pydantic models
frontend/         dependency-free web UI (HTML/CSS/JS)
tests/            pytest suite + realistic sample-PDF generator
```

## Cost and Gemini calls

A document with *C* chunks and *S* multi-part sections makes about **C + S + 1** generation calls. Sections that contain a single chunk reuse that chunk's summary. The `semantic` strategy also makes one embedding call per 100 sentences, and only for sections longer than `max_tokens`. Token counts per document are shown in `stats` and in the UI.

## Tests

```bash
pytest -q
```

The tests generate a multi-page PDF with headings, a running header, page numbers, a list and a table. They check extraction, every chunking strategy, the hierarchical summarizer (with a fake LLM, so no API calls) and the API flow.

## Limitations

- Scanned (image-only) pages are detected and reported but not OCR'd. Run `ocrmypdf` on them first.
- Token counts are estimates (about 4 characters per token), which is enough for packing chunks. Gemini's exact counts appear in `stats.llm_*`.
- Jobs run in-process (at most 2 at a time). For multi-worker deployments, replace `store.py` and the asyncio task with a real queue such as Redis, RQ or Celery.
