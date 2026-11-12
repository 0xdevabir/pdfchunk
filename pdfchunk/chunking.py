"""Section tree construction and chunking strategies.

Strategies
----------
structural (default)
    Chunks never cross section boundaries. Within a section, paragraphs / list items /
    tables are packed greedily up to ``max_tokens``; oversized paragraphs are split at
    sentence boundaries; consecutive chunks of the same section share ``overlap_tokens``
    of trailing sentences; tiny trailing chunks are merged back. Each section's first
    chunk starts with its Markdown heading so chunks are self-describing.

fixed
    Sentence-aligned sliding window over the whole document. Baseline for comparison.

semantic
    Like structural, but long sections are split where the topic shifts: adjacent
    sentence windows are embedded (Gemini embeddings) and the section is cut at the
    largest cosine-distance jumps (above a document-wide percentile).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from .models import Chunk, ChunkingOptions, Element, ElementType, ExtractedDocument, Section
from .textutils import estimate_tokens, split_oversized, split_sentences

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class _Unit:
    text: str
    page: int
    etype: ElementType
    para_id: int  # units with the same para_id are sentences of the same paragraph
    tokens: int
    is_heading: bool = False


@dataclass
class _Run:
    """Contiguous body content belonging to one section."""

    section_id: str
    heading_path: list[str]
    units: list[_Unit]


# --------------------------------------------------------------------------- sections


def build_sections(extracted: ExtractedDocument) -> tuple[list[Section], list[_Run]]:
    title = extracted.metadata.title or extracted.metadata.filename
    root = Section(id="s0", title=title, level=0, page_start=1, page_end=max(1, extracted.metadata.page_count))
    sections: dict[str, Section] = {root.id: root}
    order: list[str] = [root.id]
    stack: list[Section] = [root]
    runs: list[_Run] = []
    para_counter = 0

    def path_of(sec: Section) -> list[str]:
        p: list[str] = []
        cur: Section | None = sec
        while cur is not None and cur.level > 0:
            p.append(cur.title)
            cur = sections.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(p))

    current = root
    current_run: _Run | None = None
    for el in extracted.elements:
        if el.type == ElementType.heading:
            level = max(1, el.level or 1)
            while stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            sec = Section(
                id=f"s{len(order)}", title=el.text, level=level, page_start=el.page,
                page_end=el.page, parent_id=parent.id,
            )
            sections[sec.id] = sec
            order.append(sec.id)
            parent.children.append(sec.id)
            stack.append(sec)
            current = sec
            para_counter += 1
            current_run = _Run(sec.id, path_of(sec), [
                _Unit("#" * min(level, 6) + " " + el.text, el.page, ElementType.heading, para_counter,
                      estimate_tokens(el.text) + 1, is_heading=True)
            ])
            runs.append(current_run)
            continue

        if current_run is None:  # preamble before the first heading
            current_run = _Run(root.id, [], [])
            runs.append(current_run)
        para_counter += 1
        for s in stack:  # extend page ranges of the whole ancestor chain
            s.page_end = max(s.page_end, el.page)
        current_run.units.extend(_element_units(el, para_counter))

    return [sections[i] for i in order], runs


def _element_units(el: Element, para_id: int) -> list[_Unit]:
    if el.type == ElementType.table:
        return [_Unit(el.text, el.page, el.type, para_id, estimate_tokens(el.text))]
    return [
        _Unit(s, el.page, el.type, para_id, estimate_tokens(s))
        for s in (split_sentences(el.text) or [el.text])
    ]


def _cap_units(units: list[_Unit], max_tokens: int) -> list[_Unit]:
    """Make sure no single unit exceeds max_tokens (huge sentences / tables)."""
    out: list[_Unit] = []
    for u in units:
        if u.tokens <= max_tokens:
            out.append(u)
            continue
        if u.etype == ElementType.table:
            rows = u.text.splitlines()
            header, body = rows[:2], rows[2:]
            buf: list[str] = []
            for r in body:
                if buf and estimate_tokens("\n".join(header + buf + [r])) > max_tokens:
                    t = "\n".join(header + buf)
                    out.append(_Unit(t, u.page, u.etype, u.para_id, estimate_tokens(t)))
                    buf = []
                buf.append(r)
            t = "\n".join(header + buf)
            out.append(_Unit(t, u.page, u.etype, u.para_id, estimate_tokens(t)))
        else:
            for piece in split_oversized(u.text, max_tokens):
                out.append(_Unit(piece, u.page, u.etype, u.para_id, estimate_tokens(piece)))
    return out


# --------------------------------------------------------------------------- packing


def _join(units: Sequence[_Unit]) -> str:
    parts: list[str] = []
    prev: _Unit | None = None
    for u in units:
        if prev is None:
            parts.append(u.text)
        elif u.para_id == prev.para_id and u.etype != ElementType.table:
            parts.append(" " + u.text)
        else:
            parts.append("\n\n" + u.text)
        prev = u
    return "".join(parts).strip()


def _overlap_tail(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    if overlap_tokens <= 0:
        return []
    tail: list[_Unit] = []
    total = 0
    for u in reversed(units):
        if u.is_heading or u.etype == ElementType.table or total + u.tokens > overlap_tokens:
            break
        tail.insert(0, u)
        total += u.tokens
    return tail


def _pack(units: list[_Unit], max_tokens: int, overlap_tokens: int) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_tokens = 0
    for u in units:
        if cur and cur_tokens + u.tokens > max_tokens:
            groups.append(cur)
            overlap = _overlap_tail(cur, overlap_tokens)
            ov_tokens = sum(x.tokens for x in overlap)
            if ov_tokens + u.tokens > max_tokens:
                overlap, ov_tokens = [], 0
            cur, cur_tokens = list(overlap), ov_tokens
        cur.append(u)
        cur_tokens += u.tokens
    if cur:
        groups.append(cur)
    return groups


def _merge_small(groups: list[list[_Unit]], min_tokens: int, max_tokens: int) -> list[list[_Unit]]:
    """Merge a too-small trailing group into its predecessor if the result stays reasonable."""
    if len(groups) < 2 or min_tokens <= 0:
        return groups
    last = groups[-1]
    if sum(u.tokens for u in last) < min_tokens:
        prev = groups[-2]
        prev_ids = {id(u) for u in prev}
        extra = [u for u in last if id(u) not in prev_ids]  # drop overlap duplicates
        if sum(u.tokens for u in prev) + sum(u.tokens for u in extra) <= int(max_tokens * 1.2):
            groups = groups[:-2] + [prev + extra]
    return groups


# --------------------------------------------------------------------------- strategies


def _structural_groups(run: _Run, opts: ChunkingOptions) -> list[list[_Unit]]:
    units = _cap_units(run.units, opts.max_tokens)
    groups = _pack(units, opts.max_tokens, opts.overlap_tokens)
    return _merge_small(groups, opts.min_tokens, opts.max_tokens)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return 1.0 - dot / (na * nb)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("inf")
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def _semantic_groups(runs: list[_Run], opts: ChunkingOptions, embed: Embedder) -> list[list[list[_Unit]]]:
    capped = [_cap_units(r.units, opts.max_tokens) for r in runs]
    needs = [i for i, units in enumerate(capped) if sum(u.tokens for u in units) > opts.max_tokens and len(units) > 2]

    # Embed a 3-sentence window around each sentence for a smoother topic signal.
    windows: list[str] = []
    spans: dict[int, tuple[int, int]] = {}
    for i in needs:
        units = capped[i]
        start = len(windows)
        for j in range(len(units)):
            windows.append(" ".join(u.text for u in units[max(0, j - 1) : j + 2])[:2000])
        spans[i] = (start, len(windows))
    vectors = await embed(windows) if windows else []

    distances: dict[int, list[float]] = {}
    for i, (a, b) in spans.items():
        vecs = vectors[a:b]
        distances[i] = [_cosine_distance(vecs[j], vecs[j + 1]) for j in range(len(vecs) - 1)]
    threshold = _percentile([d for ds in distances.values() for d in ds], opts.semantic_breakpoint_percentile)

    result: list[list[list[_Unit]]] = []
    for i, units in enumerate(capped):
        if i not in distances:
            result.append(_merge_small(_pack(units, opts.max_tokens, 0), opts.min_tokens, opts.max_tokens))
            continue
        segments: list[list[_Unit]] = [[units[0]]]
        for j, d in enumerate(distances[i]):
            nxt = units[j + 1]
            seg_tokens = sum(u.tokens for u in segments[-1])
            # split at topic shifts, but never between sentences of the same paragraph
            # unless the segment is already large
            boundary = d >= threshold and (nxt.para_id != units[j].para_id or seg_tokens > opts.max_tokens // 2)
            if boundary and seg_tokens >= opts.min_tokens:
                segments.append([])
            segments[-1].append(nxt)
        groups: list[list[_Unit]] = []
        for seg in segments:
            if groups and sum(u.tokens for u in groups[-1]) + sum(u.tokens for u in seg) <= opts.min_tokens:
                groups[-1].extend(seg)  # glue tiny topic fragments together
            else:
                groups.extend(_pack(seg, opts.max_tokens, 0))
        result.append(_merge_small(groups, opts.min_tokens, opts.max_tokens))
    return result


def _fixed_groups(runs: list[_Run], opts: ChunkingOptions) -> list[tuple[list[_Unit], str]]:
    tagged: list[tuple[_Unit, str]] = []
    for r in runs:
        for u in _cap_units(r.units, opts.max_tokens):
            tagged.append((u, r.section_id))
    groups: list[tuple[list[_Unit], str]] = []
    i = 0
    while i < len(tagged):
        cur: list[_Unit] = []
        tokens = 0
        j = i
        while j < len(tagged) and (not cur or tokens + tagged[j][0].tokens <= opts.max_tokens):
            cur.append(tagged[j][0])
            tokens += tagged[j][0].tokens
            j += 1
        groups.append((cur, tagged[i][1]))
        if j >= len(tagged):
            break
        # step back for overlap, but always advance by at least one unit
        back, ov = j, 0
        while back - 1 > i and ov + tagged[back - 1][0].tokens <= opts.overlap_tokens:
            back -= 1
            ov += tagged[back][0].tokens
        i = max(back, i + 1)
    return groups


# --------------------------------------------------------------------------- entry point


async def chunk_document(
    extracted: ExtractedDocument, opts: ChunkingOptions, embed: Embedder | None = None
) -> tuple[list[Section], list[Chunk]]:
    sections, runs = build_sections(extracted)
    by_id = {s.id: s for s in sections}
    # A heading with no body of its own (e.g. "2 Solar" directly followed by "2.1 ...") is
    # carried into the next run so it never becomes a tiny heading-only chunk.
    merged: list[_Run] = []
    pending: list[_Unit] = []
    for r in runs:
        if r.units and all(u.is_heading for u in r.units):
            pending.extend(r.units)
            continue
        if pending:
            r.units = pending + r.units
            pending = []
        if r.units:
            merged.append(r)
    if pending:  # trailing headings with no content at all
        merged.append(_Run(runs[-1].section_id, runs[-1].heading_path, pending))
    runs = merged
    path_by_section = {r.section_id: r.heading_path for r in runs}

    grouped: list[tuple[list[_Unit], str]] = []
    if opts.strategy == "fixed":
        grouped = _fixed_groups(runs, opts)
    elif opts.strategy == "semantic":
        if embed is None:
            raise ValueError("The semantic strategy needs an embedding function (set GEMINI_API_KEY).")
        for run, groups in zip(runs, await _semantic_groups(runs, opts, embed)):
            grouped.extend((g, run.section_id) for g in groups)
    else:
        for run in runs:
            grouped.extend((g, run.section_id) for g in _structural_groups(run, opts))

    chunks: list[Chunk] = []
    for idx, (units, section_id) in enumerate(grouped):
        text = _join(units)
        if not text.strip():
            continue
        chunk = Chunk(
            id=f"c{len(chunks)}",
            index=len(chunks),
            text=text,
            token_count=estimate_tokens(text),
            page_start=min(u.page for u in units),
            page_end=max(u.page for u in units),
            section_id=section_id,
            heading_path=path_by_section.get(section_id, []),
            element_types=sorted({u.etype for u in units}, key=lambda e: e.value),
        )
        chunks.append(chunk)
        by_id[section_id].chunk_ids.append(chunk.id)

    # roll token counts up the tree (children always come after their parent in `sections`)
    tokens_by_chunk = {c.id: c.token_count for c in chunks}
    for sec in reversed(sections):
        sec.token_count = sum(tokens_by_chunk[c] for c in sec.chunk_ids) + sum(
            by_id[ch].token_count for ch in sec.children
        )
    return sections, chunks
