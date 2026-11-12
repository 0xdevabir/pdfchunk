"""Generates a realistic multi-page PDF for tests and demos.

Features: title, numbered multi-level headings, body paragraphs (some spanning a page
break), a bullet list, a table, a running header and page-number footer.

    python tests/sample_pdf.py sample.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

LOREM = (
    "Solar capacity additions accelerated sharply during the year as module prices fell and "
    "permitting timelines shortened in several large markets. Utility-scale projects accounted "
    "for most new installations, while distributed rooftop systems grew steadily in residential "
    "segments. Grid operators reported increasing curtailment during midday peaks, which "
    "strengthened the business case for co-located battery storage. Analysts expect these "
    "dynamics to persist as interconnection queues are gradually cleared."
)
STORAGE = (
    "Battery storage deployments doubled compared with the previous year. Lithium iron phosphate "
    "chemistry now dominates stationary applications because of its cost and safety profile. "
    "Four-hour systems became the standard configuration for capacity contracts, and several "
    "operators began piloting long-duration technologies such as iron-air and flow batteries."
)
POLICY = (
    "Policy support remained a key driver. Tax credits were extended for ten years, and new "
    "domestic content bonuses encouraged local manufacturing. However, trade restrictions on "
    "imported components increased short-term costs for some developers."
)

STRUCTURE = [
    ("title", "Renewable Energy Market Review 2025"),
    ("p", "This report summarizes developments across solar, wind and storage markets. " + POLICY),
    ("h1", "1 Introduction"),
    ("p", LOREM),
    ("p", POLICY),
    ("h1", "2 Solar Power"),
    ("h2", "2.1 Utility-Scale Solar"),
    ("p", LOREM), ("p", LOREM), ("p", LOREM),
    ("h2", "2.2 Distributed Solar"),
    ("p", "Residential installations grew by 18 percent. Key drivers included:"),
    ("li", "- Falling installation costs for rooftop systems"),
    ("li", "- Net metering reforms in several states"),
    ("li", "- Growing demand for home backup power"),
    ("h1", "3 Energy Storage"),
    ("p", STORAGE), ("p", STORAGE),
    ("table", None),
    ("h1", "4 Outlook"),
    ("p", LOREM + " " + STORAGE), ("p", POLICY),
]


def build(path: Path | None = None) -> bytes:
    doc = pymupdf.open()
    W, H = 595, 842
    margin, y = 60, 0.0
    page = None

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=W, height=H)
        page.insert_text((margin, 35), "Renewable Energy Market Review - Confidential", fontsize=8, fontname="helv")
        page.insert_text((W / 2 - 5, H - 30), str(doc.page_count), fontsize=8, fontname="helv")
        y = 70

    scratch = pymupdf.open()
    scratch_page = scratch.new_page(width=W, height=H)

    def fits(rect, text, size, font) -> bool:
        return scratch_page.insert_textbox(rect, text, fontsize=size, fontname=font) >= 0

    def write(text: str, size: float, font: str, gap: float):
        """Write text, flowing onto new pages (splitting at word boundaries) when needed."""
        nonlocal y
        words = text.split()
        while words:
            if page is None or y > H - 90:
                new_page()
            box = pymupdf.Rect(margin, y, W - margin, H - 70)
            lo, hi = 0, len(words)
            while lo < hi:  # largest prefix of words that fits in the box
                mid = (lo + hi + 1) // 2
                if fits(box, " ".join(words[:mid]), size, font):
                    lo = mid
                else:
                    hi = mid - 1
            if lo == 0:
                new_page()
                continue
            rc = page.insert_textbox(box, " ".join(words[:lo]), fontsize=size, fontname=font)
            y += (box.height - rc) + gap
            words = words[lo:]
            if words:
                new_page()

    for kind, text in STRUCTURE:
        if kind == "title":
            write(text, 20, "hebo", 14)
        elif kind == "h1":
            write(text, 15, "hebo", 8)
        elif kind == "h2":
            write(text, 12.5, "hebo", 6)
        elif kind == "li":
            write(text, 10, "helv", 2)
        elif kind == "p":
            write(text, 10, "helv", 9)
        elif kind == "table":
            if y > H - 200:
                new_page()
            rows = [("Technology", "Capacity (GW)", "Growth"), ("Lithium iron phosphate", "42", "+110%"),
                    ("Flow batteries", "1.2", "+35%"), ("Iron-air", "0.3", "new")]
            col_x = [margin, margin + 200, margin + 330, margin + 430]
            top = y
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    page.insert_text((col_x[c] + 4, y + 14), cell, fontsize=10,
                                     fontname="hebo" if r == 0 else "helv")
                y += 20
            for r in range(len(rows) + 1):
                page.draw_line((col_x[0], top + 20 * r), (col_x[-1], top + 20 * r))
            for x in col_x:
                page.draw_line((x, top), (x, top + 20 * len(rows)))
            y += 14

    scratch.close()
    doc.set_metadata({"title": "", "author": "Test Suite"})
    data = doc.tobytes()
    doc.close()
    if path:
        Path(path).write_bytes(data)
    return data


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample.pdf")
    build(out)
    print(f"wrote {out}")
