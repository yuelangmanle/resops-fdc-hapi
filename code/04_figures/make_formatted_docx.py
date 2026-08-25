#!/usr/bin/env python3
"""Generate the journal-format Word manuscript and its PDF rendering."""
from __future__ import annotations

import subprocess
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_LINE_SPACING

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "manuscript/drafts/STANDARD_MANUSCRIPT.md"
OUT_DOCX = ROOT / "outputs/submission/STANDARD_MANUSCRIPT_JH_FORMATTED.docx"
OUT_PDF = ROOT / "outputs/submission/STANDARD_MANUSCRIPT_JH_FORMATTED.pdf"

FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(12)


def main() -> None:
    # Generate the base DOCX from the canonical Markdown source.
    subprocess.run([
        "pandoc", str(SRC), "-o", str(OUT_DOCX),
        "--standalone",
        "--resource-path=manuscript/drafts",
    ], check=True, cwd=ROOT)

    # Apply journal-style formatting.
    doc = Document(str(OUT_DOCX))
    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = FONT_SIZE
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(6)

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    doc.save(str(OUT_DOCX))

    # Convert the formatted document to PDF.
    subprocess.run([
        "soffice", "--headless", "--convert-to", "pdf",
        "--outdir", str(OUT_DOCX.parent), str(OUT_DOCX),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("saved", OUT_DOCX, OUT_PDF)


if __name__ == "__main__":
    main()
