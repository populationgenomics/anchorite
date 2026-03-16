# Markdown → PDF anchor association

`prototype_md.py` is a command-line tool that takes a PDF and a cleaned
Markdown file (with `<!--page-->` page-break markers) and produces one
bounding-box anchor per Markdown segment — heading, sentence, list item, table
cell, etc.  It is the *reverse* of the main `anchorite` workflow: instead of
aligning OCR word boxes to generated Markdown, it aligns generated Markdown
segments back to the raw PDF character layer.

The result is written as one annotated PNG per page showing which region of
the PDF each segment was matched to.

---

## Prerequisites

The project is managed with [uv](https://github.com/astral-sh/uv).

```shell
# from the repo root
uv sync
```

---

## Input format

### PDF

Any selectable-text PDF.  The tool extracts per-character bounding boxes
directly from the PDF's text layer via pypdfium2, so scanned/image-only PDFs
will not work.

### Markdown

The Markdown must contain `<!--page-->` HTML comments as page-break markers,
one per page boundary in the source PDF.  The *n*-th `<!--page-->` marker
(counting from 0) corresponds to the start of PDF page *n*.

Example structure:

```markdown
<!--page-->
# Title

Author names ...

<!--page-->
## Introduction

First paragraph ...

<!--page-->
```

The markers do not need to be perfectly aligned with page boundaries — the
association algorithm uses a ±10-page search window in its first pass and
then refines using document-order constraints, so moderate inaccuracies are
tolerated.

---

## Running the tool

```shell
uv run python prototype_md.py <pdf> <markdown>
```

### Example

```shell
uv run python prototype_md.py paper.pdf paper.md
```

### Output

**Console** — summary statistics and per-page anchor listings:

```
1985 segments parsed
Phase 1 (conservative HSP): 847/1985 segments matched (42%)
1066 anchors matched

Page 0: 3 anchors
  pass=1 boxes=(BBox(top=98, ...))  '# A scalable approach ...'
  ...
Page 1: 12 anchors
  ...
```

**Images** — one PNG per page, written alongside the PDF:

```
paper.p0.md.png
paper.p1.md.png
...
```

Each PNG is a rendered page with coloured rectangles overlaid on matched
segments.  Dark fill (α ≈ 80/255) indicates a phase-1 match; medium fill
(α ≈ 50/255) indicates a phase-2 match.  Colours cycle through five hues so
adjacent anchors are visually distinct.

---

## Algorithm overview

Association runs in two phases.

### Phase 1 — conservative HSP-based page assignment

For each Markdown segment (≥ 10 alphanumeric characters), both the segment
text and the candidate page texts are reduced to their alphanumeric characters
only (no spaces).  The segment is then aligned against each page in a ±10-page
window around its `<!--page-->` marker using ungapped local alignment (HSPs),
retrieving the top-2 scores per page via
`seq_smith.top_k_ungapped_local_align_many`.

A segment is assigned to a page only when:

- **Coverage** — the best HSP covers ≥ 90 % of the segment's normalised length.
- **Cross-page uniqueness** — the winning page score is ≥ 2× the next-best
  page score, so ambiguous content (e.g. running page headers, common phrases)
  is deferred rather than misassigned.

Accepted segments are then aligned with full gapped Smith-Waterman (including
spaces) against the *residual* of their assigned page — the flat-string text
not yet claimed by any earlier segment — to obtain precise character positions
and bounding boxes.

### Phase 2 — page-constrained matching

Segments not matched in phase 1 (short segments, repeating headers, ambiguous
content) are retried using the document-order constraint.  Because Markdown is
in reading order, any unmatched segment must lie between the PDF pages of its
nearest matched neighbours.  The search is restricted to the interval
`[prev_matched_page, next_matched_page]`; no uniqueness requirement applies
within this narrow window.

---

## Using the association API directly

```python
import pathlib
from anchorite.md_association import associate, parse_markdown_segments

pdf_path = pathlib.Path("paper.pdf")
markdown  = pathlib.Path("paper.md").read_text()

# Returns one Anchor per matched segment, in Markdown order.
anchors = associate(pdf_path, markdown)

for anchor in anchors:
    print(anchor.page, anchor.boxes, anchor.text[:60])
```

`return_pass_info=True` makes `associate` return `(anchors, passes)` where
`passes` is a parallel list of integers: `1` = phase-1 match, `2` = phase-2
match.

```python
anchors, passes = associate(pdf_path, markdown, return_pass_info=True)
```

---

## Markdown preparation tips

- **Page markers are required.**  Without `<!--page-->` markers the algorithm
  has no page hints and phase 1 will search the entire document for every
  segment, which both slows it down and increases false-positive risk.
- **One marker per page boundary.**  Place each `<!--page-->` on its own line,
  separated from surrounding content by a blank line.  Markers that immediately
  follow a paragraph (no blank line) are handled automatically, but clean
  separation is preferred.
- **Tables are fine.**  GFM pipe-table rows are parsed as individual cell
  segments.  Table cells containing short values (single numbers, "NA", etc.)
  are typically not unique enough for phase 1 and are matched in phase 2.
- **Equations.**  LaTeX math environments (`$...$`, `$$...$$`) are treated as
  opaque strings; the normalisation discards all non-alphanumeric characters,
  so an equation will only match if the variable names or numbers it contains
  appear verbatim in the PDF's text layer.  Image-rendered equations (common in
  older PDFs) will not match.
