"""Add real Word cross-reference fields to the submission DOCX.

Pass 1: wrap each figure/table caption's "Figure N"/"Table N" text in a bookmark
        (_RefFigN / _RefTabN) so it becomes a cross-reference target.
Pass 2: in body paragraphs, replace each in-text "Figure N"/"Table N" mention with
        a fldSimple REF field (live, clickable, updates on F9).

pandoc produces paragraphs where the whole text often lives in a single w:r, so we
rebuild paragraphs run-by-run, splitting at each match boundary.

Run: python3 code/04_figures/add_docx_crossrefs.py <in.docx> <out.docx>
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def new_run(text):
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def make_bookmark(name):
    bid = abs(hash(name)) % 900000 + 1000
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(bid))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(bid))
    return start, end


def make_ref_field(bookmark, display):
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), " REF " + bookmark + " \\h ")
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rStyle = OxmlElement("w:rStyle")
    rStyle.set(qn("w:val"), "Hyperlink")
    rPr.append(rStyle)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = display
    r.append(t)
    fld.append(r)
    return fld


def para_text(p_el):
    parts = []
    for r in p_el.findall(qn("w:r")):
        t = r.find(qn("w:t"))
        parts.append(t.text if t is not None else "")
    return "".join(parts)


def replace_runs(p_el, new_children):
    """Remove all w:r / w:fldSimple, keep others (pPr, bookmarkStart...), append new children after last kept node."""
    kept = []
    for child in list(p_el):
        if child.tag in (qn("w:r"), qn("w:fldSimple")):
            p_el.remove(child)
        else:
            kept.append(child)
    anchor = kept[-1] if kept else None
    for el in new_children:
        if anchor is not None:
            anchor.addnext(el)
            anchor = el
        else:
            p_el.append(el)


def process(src, dst):
    doc = Document(str(src))
    body = doc.element.body
    paras = body.findall(qn("w:p"))

    bookmarks = {}

    # ---- Pass 1: captions (wrap "Figure N" / "Table N" in bookmark) ----
    for p_el in paras:
        full = para_text(p_el).strip()
        m = re.match(r"^(Figure|Table)\s+(\d+)", full)
        if not m:
            continue
        kind = m.group(1).lower()
        num = m.group(2)
        name = ("_RefFig" if kind == "figure" else "_RefTab") + num
        bookmarks[(kind, num)] = name

        word_end = m.start() + len(m.group(1))          # end of "Figure"
        digits_end = word_end + len(m.group(2)) + 0
        # handle leading space between word and digits
        digits_start = m.start(2)
        pre = full[m.start():word_end]                  # "Figure" or "Table"
        # re-insert the space(s) that were consumed by \s+ and any around
        space_before = full[word_end:digits_start]      # " "
        digits = full[digits_start:digits_end]          # "1"
        rest = full[digits_end:]                        # ". Distribution ..."
        start, end = make_bookmark(name)
        children = [
            new_run(pre + space_before + digits),
            start,
            end,
            new_run(rest) if rest else None,
        ]
        replace_runs(p_el, [c for c in children if c is not None])

    print("captions bookmarked:", len(bookmarks))

    # ---- Pass 2: body references ----
    ref_count = 0
    for p_el in paras:
        full = para_text(p_el)
        if not full.strip():
            continue
        if re.match(r"^\s*(Figure|Table)\s+\d+[.\s]", full.strip()):
            continue  # caption already handled
        segs = []
        pos = 0
        n = 0
        for m in re.finditer(r"(Figure|Table)\s+(\d+)", full):
            kind = m.group(1).lower()
            num = m.group(2)
            bm = bookmarks.get((kind, num))
            if bm is None:
                continue
            if m.start() > pos:
                segs.append(new_run(full[pos:m.start()]))
            segs.append(make_ref_field(bm, m.group(1) + " " + num))
            pos = m.end()
            n += 1
        if n == 0:
            continue
        if pos < len(full):
            segs.append(new_run(full[pos:]))
        replace_runs(p_el, segs)
        ref_count += n

    print("in-text references replaced:", ref_count)
    doc.save(str(dst))
    print("saved:", dst)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/submission/STANDARD_MANUSCRIPT.docx"
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    process(src, dst)
