"""PDF -> ordered list of logical elements (headings, paragraphs, list items, tables).

Layout intelligence implemented here:
  * per-line font statistics to find the body font size and heading candidates
  * heading levels from the embedded PDF outline (TOC) when present, otherwise from a
    font-size hierarchy + bold + section numbering ("2.3.1 Results")
  * repeated header/footer and page-number removal
  * table detection (rendered as Markdown) with de-duplication of the table's text
  * de-hyphenation, list-item detection, and merging of paragraphs split across pages
  * detection of scanned (image-only) pages
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .models import DocumentMetadata, Element, ElementType, ExtractedDocument
from .textutils import fingerprint, normalize_whitespace

MARGIN_RATIO = 0.09  # top/bottom band in which headers/footers live
_PAGE_NUMBER = re.compile(r"^(page\s*)?[\divxlc]+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
_KEYWORD_HEADING = re.compile(r"^(chapter|section|part|appendix|article)\s+[\w.]+", re.IGNORECASE)
_ROMAN_HEADING = re.compile(r"^[IVXLC]+\.\s+\S")
_BULLET = re.compile(r"^\s*([•▪●◦‣⁃∙·*\-–—]|\(?[a-z0-9]{1,3}[.)])\s+")
_TERMINAL = (".", "!", "?", ":", ";", '."', ".”", ".)")


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]


@dataclass
class _Block:
    page: int
    lines: list[_Line]
    page_height: float
    kind: str = "text"  # "text" | "table"
    table_md: str | None = None
    removed: bool = False
    heading_level: int | None = None
    _bbox: tuple[float, float, float, float] | None = field(default=None, repr=False)

    @property
    def text(self) -> str:
        if self.kind == "table":
            return self.table_md or ""
        return normalize_whitespace("\n".join(ln.text for ln in self.lines))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if self._bbox:
            return self._bbox
        xs0, ys0, xs1, ys1 = zip(*(ln.bbox for ln in self.lines))
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    @property
    def size(self) -> float:
        weights: Counter[float] = Counter()
        for ln in self.lines:
            weights[ln.size] += len(ln.text)
        return weights.most_common(1)[0][0] if weights else 0.0

    @property
    def bold(self) -> bool:
        total = sum(len(ln.text) for ln in self.lines) or 1
        return sum(len(ln.text) for ln in self.lines if ln.bold) / total > 0.6


def _round_size(s: float) -> float:
    return round(s * 2) / 2


def _span_is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & 16) or "bold" in span.get("font", "").lower()


def _inside(inner: tuple, outer: tuple, tol: float = 2.0) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] - tol <= cx <= outer[2] + tol and outer[1] - tol <= cy <= outer[3] + tol


# --------------------------------------------------------------------------- page reading


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    def cell(v: str | None) -> str:
        return " ".join((v or "").split()).replace("|", "\\|")

    rows = [[cell(v) for v in r] for r in rows if any((v or "").strip() for v in r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _read_page(page: pymupdf.Page, page_no: int, detect_tables: bool) -> tuple[list[_Block], bool]:
    height = page.rect.height
    tables: list[_Block] = []
    if detect_tables:
        try:
            for t in page.find_tables().tables:
                if t.row_count >= 2 and t.col_count >= 2:
                    rows = [
                        [page.get_textbox(pymupdf.Rect(c)) if c else "" for c in row.cells]
                        for row in t.rows
                    ]
                    md = _table_to_markdown(rows)
                    if md:
                        tables.append(
                            _Block(page_no, [], height, kind="table", table_md=md, _bbox=tuple(t.bbox))
                        )
        except Exception:  # table finder is heuristic; never fail extraction because of it
            tables = []

    raw = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    blocks: list[_Block] = []
    for b in raw.get("blocks", []):
        if b.get("type") != 0:
            continue
        if any(_inside(tuple(b["bbox"]), t.bbox) for t in tables):
            continue
        lines: list[_Line] = []
        for ln in b.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in ln["spans"]).strip()
            size_w: Counter[float] = Counter()
            bold_chars = 0
            for s in spans:
                size_w[_round_size(s["size"])] += len(s["text"])
                if _span_is_bold(s):
                    bold_chars += len(s["text"].strip())
            total = sum(len(s["text"].strip()) for s in spans) or 1
            lines.append(_Line(text, size_w.most_common(1)[0][0], bold_chars / total > 0.6, tuple(ln["bbox"])))
        if not lines:
            continue
        # Split a block where the font style changes (e.g. a bold heading glued to its paragraph).
        group: list[_Line] = [lines[0]]
        for prev, cur in zip(lines, lines[1:]):
            style_change = abs(cur.size - prev.size) >= 1.0 or cur.bold != prev.bold
            if style_change:
                blocks.append(_Block(page_no, group, height))
                group = []
            group.append(cur)
        blocks.append(_Block(page_no, group, height))

    # Insert tables in reading position: right after the last body block that ends above
    # the table (margin blocks such as footers are ignored, as they may come early in
    # PyMuPDF's native order).
    def in_body(b: _Block) -> bool:
        return height * MARGIN_RATIO < b.bbox[1] < height * (1 - MARGIN_RATIO)

    for t in sorted(tables, key=lambda t: t.bbox[1]):
        above = [i for i, b in enumerate(blocks) if b.kind == "text" and in_body(b) and b.bbox[3] <= t.bbox[1] + 2]
        blocks.insert(max(above) + 1 if above else 0, t)

    scanned = not any(b.kind == "text" for b in blocks) and bool(page.get_images())
    return blocks, scanned


# --------------------------------------------------------------------------- cleanup


def _remove_headers_footers(blocks: list[_Block], page_count: int) -> int:
    def in_margin(b: _Block) -> bool:
        y0, y1 = b.bbox[1], b.bbox[3]
        return y1 < b.page_height * MARGIN_RATIO or y0 > b.page_height * (1 - MARGIN_RATIO)

    removed = 0
    pages_by_fp: dict[str, set[int]] = defaultdict(set)
    for b in blocks:
        if b.kind == "text" and in_margin(b):
            pages_by_fp[fingerprint(b.text)].add(b.page)

    if page_count < 2:
        threshold = 10**9  # cannot detect repetition on a single page
    else:
        threshold = max(2, -(-page_count * 2 // 5))  # appears on >= 40% of pages (at least 2)
    repeated = {fp for fp, pages in pages_by_fp.items() if len(pages) >= threshold}
    for b in blocks:
        if b.kind != "text" or not in_margin(b):
            continue
        txt = b.text.strip()
        if fingerprint(txt) in repeated or _PAGE_NUMBER.match(txt):
            b.removed = True
            removed += 1
    return removed


def _body_font_size(blocks: list[_Block]) -> float:
    weights: Counter[float] = Counter()
    for b in blocks:
        if b.kind == "text" and not b.removed:
            for ln in b.lines:
                weights[ln.size] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 10.0


# --------------------------------------------------------------------------- headings


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _is_heading_candidate(b: _Block, body: float) -> bool:
    text = b.text
    if b.kind != "text" or b.removed or not text or len(b.lines) > 3:
        return False
    if len(text) > 160 or len(text.split()) > 22 or not re.search(r"[A-Za-z]", text):
        return False
    if text.endswith((",", ";")) or (text.endswith(".") and not _NUMBERED_HEADING.match(text)
                                     and len(text.split()) > 6):
        return False
    if _BULLET.match(text) and not _NUMBERED_HEADING.match(text):
        return False
    bigger = b.size >= body * 1.12
    bold_short = b.bold and b.size >= body * 0.95 and len(text) <= 100
    numbered = bool(_NUMBERED_HEADING.match(text) or _KEYWORD_HEADING.match(text)
                    or _ROMAN_HEADING.match(text))
    if numbered:
        # "3 apples were ..." is not a heading: require a style signal or title-like shape
        return bigger or b.bold or (text[:1].isdigit() and not text.endswith(".") and len(text) <= 80
                                    and text.split(maxsplit=1)[-1][:1].isupper())
    return bigger or bold_short


def _assign_heading_levels(blocks: list[_Block], body: float, toc: list[list]) -> bool:
    """Mark heading blocks and their levels. Returns True if the PDF outline was used."""
    toc_by_page: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for lvl, title, page in toc:
        if title and page and page > 0:
            toc_by_page[page].append((lvl, _norm_title(title)))
    used_toc = False

    candidates = [b for b in blocks if _is_heading_candidate(b, body)]

    # 1) outline matches
    if toc_by_page:
        for b in blocks:
            if b.kind != "text" or b.removed or len(b.text) > 200:
                continue
            nt = _norm_title(b.text)
            if not nt:
                continue
            for page in (b.page, b.page - 1, b.page + 1):
                match = next(
                    (lvl for lvl, t in toc_by_page.get(page, []) if t and (nt == t or (len(t) > 6 and nt.startswith(t)))),
                    None,
                )
                if match:
                    b.heading_level = match
                    used_toc = True
                    break

    # 2) heuristics for everything not matched by the outline
    remaining = [b for b in candidates if b.heading_level is None]
    if used_toc:
        deepest = max(lvl for entries in toc_by_page.values() for lvl, _ in entries)
        for b in remaining:
            if b.size >= body * 1.12 or b.bold:
                b.heading_level = min(deepest + 1, 6)
        return True

    sizes = sorted({b.size for b in remaining if b.size >= body * 1.12}, reverse=True)
    tiers: list[float] = []
    for s in sizes:  # merge sizes within 0.75pt into one tier
        if not tiers or tiers[-1] - s > 0.75:
            tiers.append(s)

    def tier_of(size: float) -> int:
        for i, t in enumerate(tiers):
            if size >= t - 0.75:
                return i + 1
        return len(tiers) + 1

    for b in remaining:
        m = _NUMBERED_HEADING.match(b.text)
        if m and (b.bold or b.size >= body * 1.05):
            b.heading_level = min(m.group(1).count(".") + 1, 6)
        elif b.size >= body * 1.12:
            b.heading_level = min(tier_of(b.size), 6)
        else:  # bold at body size, or plain numbered
            b.heading_level = min(len(tiers) + 1, 6)
    return False


def _pop_title_block(blocks: list[_Block]) -> str | None:
    """If the biggest heading is unique and on page 1, treat it as the document title."""
    heads = [b for b in blocks if b.heading_level is not None and not b.removed]
    if len(heads) < 2:
        return None
    top = max(heads, key=lambda b: b.size)
    same = [b for b in heads if abs(b.size - top.size) < 0.75]
    others = [b for b in heads if b is not top]
    if top.page == 1 and len(same) == 1 and top.size >= max(b.size for b in others) + 2:
        top.removed = True
        return top.text
    return None


# --------------------------------------------------------------------------- assembly


def _blocks_to_elements(blocks: list[_Block]) -> list[Element]:
    elements: list[Element] = []
    for b in blocks:
        if b.removed:
            continue
        if b.kind == "table":
            elements.append(Element(type=ElementType.table, text=b.text, page=b.page, bbox=b.bbox))
            continue
        if b.heading_level is not None:
            elements.append(
                Element(type=ElementType.heading, text=b.text, page=b.page, level=b.heading_level,
                        font_size=b.size, bbox=b.bbox)
            )
            continue

        # split a block into list items when its lines start with bullets
        pieces: list[tuple[ElementType, list[str]]] = []
        for ln in b.lines:
            if _BULLET.match(ln.text):
                pieces.append((ElementType.list_item, [ln.text]))
            elif pieces:
                pieces[-1][1].append(ln.text)
            else:
                pieces.append((ElementType.paragraph, [ln.text]))
        for etype, lines in pieces:
            text = normalize_whitespace("\n".join(lines))
            if not text:
                continue
            prev = elements[-1] if elements else None
            # merge paragraphs broken across columns/pages mid-sentence
            if (
                etype == ElementType.paragraph
                and prev is not None
                and prev.type == ElementType.paragraph
                and not prev.text.rstrip().endswith(_TERMINAL)
                and text[:1].islower()
            ):
                prev.text = prev.text.rstrip("-") + ("" if prev.text.endswith("-") else " ") + text
                continue
            elements.append(Element(type=etype, text=text, page=b.page, font_size=b.size, bbox=b.bbox))
    return elements


def extract_pdf(source: str | Path | bytes, filename: str | None = None,
                detect_tables: bool = True) -> ExtractedDocument:
    if isinstance(source, (bytes, bytearray)):
        doc = pymupdf.open(stream=bytes(source), filetype="pdf")
        filename = filename or "document.pdf"
    else:
        doc = pymupdf.open(str(source))
        filename = filename or Path(source).name

    if doc.needs_pass:
        raise ValueError("PDF is password protected")
    if not doc.is_pdf:
        raise ValueError("File is not a PDF")

    try:
        all_blocks: list[_Block] = []
        scanned: list[int] = []
        for i, page in enumerate(doc, start=1):
            blocks, is_scanned = _read_page(page, i, detect_tables)
            if is_scanned:
                scanned.append(i)
            all_blocks.extend(blocks)

        removed = _remove_headers_footers(all_blocks, doc.page_count)
        body = _body_font_size(all_blocks)
        toc = doc.get_toc(simple=True)
        used_toc = _assign_heading_levels(all_blocks, body, toc)
        detected_title = None if used_toc else _pop_title_block(all_blocks)
        elements = _blocks_to_elements(all_blocks)

        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or detected_title
        if not title:
            first = next((e for e in elements if e.type == ElementType.heading), None)
            title = first.text if first else None

        warnings: list[str] = []
        if scanned:
            warnings.append(
                f"{len(scanned)} page(s) look scanned (image-only, no text layer): "
                f"{scanned[:10]}{'…' if len(scanned) > 10 else ''}. Run OCR first for full coverage."
            )
        if not elements:
            warnings.append("No extractable text found in this PDF.")

        metadata = DocumentMetadata(
            filename=filename,
            title=title,
            author=(meta.get("author") or "").strip() or None,
            subject=(meta.get("subject") or "").strip() or None,
            page_count=doc.page_count,
            has_outline=used_toc,
            scanned_pages=scanned,
            removed_header_footer_lines=removed,
            tables_found=sum(1 for b in all_blocks if b.kind == "table"),
        )
        return ExtractedDocument(metadata=metadata, elements=elements, body_font_size=body, warnings=warnings)
    finally:
        doc.close()
