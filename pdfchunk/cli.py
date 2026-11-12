"""Command-line interface.

    python -m pdfchunk report.pdf                       # chunk + summarize, print Markdown
    python -m pdfchunk report.pdf --no-summary -f jsonl -o chunks.jsonl
    python -m pdfchunk report.pdf --strategy semantic --max-tokens 600 -f json -o out.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .exporters import to_jsonl, to_markdown
from .models import ChunkingOptions, ProcessingOptions, SummarizationOptions
from .pipeline import process_pdf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdfchunk", description="Intelligent PDF chunking & summarization (Gemini)")
    p.add_argument("pdf", type=Path)
    p.add_argument("-s", "--strategy", choices=["structural", "fixed", "semantic"], default="structural")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--min-tokens", type=int, default=120)
    p.add_argument("--overlap", type=int, default=80, help="overlap tokens between consecutive chunks")
    p.add_argument("--no-summary", action="store_true", help="only extract and chunk (no Gemini calls)")
    p.add_argument("--length", choices=["brief", "standard", "detailed"], default="standard")
    p.add_argument("--language", help="output language for summaries (default: same as document)")
    p.add_argument("--focus", help='steer summaries, e.g. "financial risks"')
    p.add_argument("--no-tables", action="store_true", help="disable table detection (faster)")
    p.add_argument("-f", "--format", choices=["markdown", "json", "jsonl"], default="markdown")
    p.add_argument("-o", "--output", type=Path, help="write to file instead of stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 2

    options = ProcessingOptions(
        chunking=ChunkingOptions(strategy=args.strategy, max_tokens=args.max_tokens,
                                 min_tokens=args.min_tokens, overlap_tokens=args.overlap),
        summarization=SummarizationOptions(enabled=not args.no_summary, length=args.length,
                                           language=args.language, focus=args.focus),
        detect_tables=not args.no_tables,
    )

    def progress(stage: str, frac: float, msg: str) -> None:
        print(f"\r[{frac * 100:5.1f}%] {stage:<12} {msg:<60}", end="", file=sys.stderr, flush=True)

    try:
        doc = asyncio.run(process_pdf(args.pdf, options, progress=progress))
    except Exception as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    print(file=sys.stderr)
    for w in doc.warnings:
        print(f"warning: {w}", file=sys.stderr)

    if args.format == "json":
        out = doc.model_dump_json(indent=2)
    elif args.format == "jsonl":
        out = to_jsonl(doc)
    else:
        out = to_markdown(doc)

    if args.output:
        args.output.write_text(out)
        print(f"Wrote {args.output} ({doc.stats.chunk_count} chunks, {doc.stats.llm_calls} Gemini calls)",
              file=sys.stderr)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
