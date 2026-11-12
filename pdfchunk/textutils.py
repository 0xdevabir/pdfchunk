"""Small text helpers: token estimation, sentence splitting, normalization."""

from __future__ import annotations

import math
import re

_WS = re.compile(r"[ \t ]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "fig", "figs", "eq", "eqs", "no", "dr", "mr", "mrs", "ms",
    "prof", "inc", "ltd", "co", "corp", "jr", "sr", "st", "al", "approx", "dept", "sec",
    "vol", "pp", "ch", "cf", "resp", "u.s", "u.k",
}
# A sentence boundary: terminal punctuation (+ optional closing quote/bracket), whitespace,
# then an uppercase letter, digit, quote or bracket.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9À-ɏ])")


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 chars/token, floor of 0.75 tokens/word).

    Exact counts depend on the model tokenizer; this is consistent and fast, which is
    what matters for packing chunks.
    """
    if not text:
        return 0
    by_chars = len(text) / 4.0
    by_words = len(text.split()) * 0.75
    return max(1, math.ceil(max(by_chars, by_words)))


def normalize_whitespace(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return " ".join(ln for ln in lines if ln)


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that respects common abbreviations and decimals."""
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    for m in _SENT_BOUNDARY.finditer(text):
        candidate = text[start : m.start()].rstrip()
        last_word = candidate.rsplit(" ", 1)[-1].rstrip(".!?\"')]").lower()
        if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
            continue  # "e.g. Foo", "J. Smith"
        pieces.append(text[start : m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split text into pieces each <= max_tokens, preferring sentence, then word boundaries."""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    out: list[str] = []
    for sentence in split_sentences(text):
        if estimate_tokens(sentence) <= max_tokens:
            out.append(sentence)
            continue
        words = sentence.split()
        buf: list[str] = []
        for w in words:
            buf.append(w)
            if estimate_tokens(" ".join(buf)) >= max_tokens:
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
    return out


def fingerprint(text: str) -> str:
    """Normalize a line for header/footer repetition detection (digits -> #)."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))
