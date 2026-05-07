# Markdown → PDF anchor association

`md_association.associate` aligns generated Markdown back to the raw PDF
character layer.  It is the *reverse* of the main `align` workflow: instead
of mapping OCR word boxes onto generated Markdown, it pulls per-character
bounding boxes from the PDF directly via `pypdfium2` and finds the region
that each Markdown segment — heading, sentence, list item, table cell —
covers on the page.

This is the right entry point when the Markdown is independently
authoritative: JATS XML rendered to Markdown, hand-curated content, an LLM
rewrite that you trust.  When you have OCR word boxes from a separate engine
and want to align *those* to your Markdown, use `align` instead — see the
top-level README.

`prototype_md.py` is a command-line wrapper around `associate` that writes
one annotated PNG per page showing which region of the PDF each segment was
matched to.

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

`<!--page-->` HTML comments are used as a search-window hint when the
Markdown carries them.  The *n*-th `<!--page-->` marker (counting from 0)
corresponds to the start of PDF page *n*; phase 1 then searches a ±10-page
window around the hint rather than the full document.  The markers do not
need to be perfectly aligned with page boundaries — moderate inaccuracies
are tolerated.

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

Markers are *optional*.  Markdown produced from sources with no notion of
pagination — JATS XML, MarkItDown, hand-typed content — won't have them.
In that case every segment is treated as having an unknown source page and
phase 1 searches every page of the PDF.  The (unchanged) cross-page
uniqueness check still rejects ambiguous matches, so coverage stays sound;
only the cost goes up.

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

```text
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

```text
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

Association runs in two phases over the per-character bounding boxes that
`pypdfium2` extracts from the PDF.

### Flat-string construction

For each PDF page, anchorite walks the per-character stream and builds a
flat string with positional metadata.  Three small touches keep the flat
string aligned with what a human reader sees:

- **Word gaps.** A space is inserted between adjacent chars whose horizontal
  gap exceeds 20 % of font size.
- **Line-break spaces.** PDFium emits no whitespace at line breaks; the
  builder detects them (next char's baseline drops by ≥ 50 % of font size,
  or its *x* sits to the left of the current char) and inserts a space, so
  end-of-line `we` + start-of-line `identified` doesn't concatenate.
- **Soft-hyphen reconnection.** End-of-line `induc-` followed by `tion`
  reconnects to `induction` rather than `induc- tion`.  Triggered only
  when both surrounding glyphs are alphabetic, so numeric ranges like
  `2009-` + `2010` keep their hyphen.

### Normalisation

Both the segment text and the page flat string then go through a shared
NFKD-aware normaliser that decomposes accented letters to their base form,
expands ligatures, maps super/subscript digits to plain digits, drops
combining marks zero-width, and strips HTML tags from the Markdown side
only (the PDF side preserves literal `<` / `>`).  See the top-level README
for details — the same normaliser is used by every alignment entry point in
the package.

### Phase 1 — conservative HSP-based page assignment

For each Markdown segment with ≥ 10 normalised alphanum characters, both
sides are run through the loose normaliser (no spaces; lets letter-spaced
display headings match) and ungapped local alignment is run against each
candidate page using `seq_smith.top_k_ungapped_local_align_many` with `k=2`
— so up to *2 × #candidate-pages* HSPs come back.  Candidate pages are the
hint window if the Markdown has `<!--page-->` markers, or every page of the
PDF if it doesn't.

The HSPs are then pooled globally, sorted by score, and the best one is
accepted only when:

- **Coverage** — the best HSP covers ≥ 90 % of the segment's normalised
  length.
- **Uniqueness** — the best HSP scores ≥ 2× the second-best HSP *anywhere*
  in the pool.  The score gap is what matters; the runner-up's location is
  irrelevant — within-page ambiguity (a phrase that appears twice on the
  same page) and cross-page ambiguity (running page headers) are caught by
  the same check.

Accepted segments are then aligned with full gapped Smith-Waterman (strict
normalisation, spaces preserved) against the *residual* of their assigned
page — the flat-string text not yet claimed by any earlier segment — to
obtain precise character positions and bounding boxes.

### Phase 2 — page-constrained matching

Segments not matched in phase 1 (short segments, repeating headers,
ambiguous content) are retried using the document-order constraint.  Since
the Markdown is in reading order, any unmatched segment must lie between
the PDF pages of its nearest matched neighbours.  The search is restricted
to the interval `[prev_matched_page, next_matched_page]`; no uniqueness
requirement applies within this narrow window.  The phase-2 alignment
must still beat a 50 %-coverage filter to be accepted, which guards
against partial echoes (e.g. matching only `conflicting` from
`conflicting interpretations` when the heading isn't in the PDF at all).

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

- **Page markers are optional but recommended.**  When the Markdown has
  `<!--page-->` markers, phase 1 only searches a ±10-page window around
  each segment's hint, which keeps the cost down on long documents.
  Without markers, phase 1 falls back to searching every page; the
  uniqueness check still rejects ambiguous matches, so coverage stays
  sound — only the throughput drops.
- **One marker per page boundary.**  Place each `<!--page-->` on its own
  line, separated from surrounding content by a blank line.  Markers that
  immediately follow a paragraph (no blank line) are handled automatically,
  but clean separation is preferred.
- **HTML in segments.**  The Markdown side of the alignment treats HTML
  tags (`<sup>1</sup>`, `<a id="…">`) as zero-width, so segments emitted
  from JATS-style sources align correctly against PDF text that lacks the
  markup.  The PDF side keeps literal `<` / `>` characters intact —
  `p < 0.05` aligns as `p 0.05` (the operator collapses to space) rather
  than disappearing.
- **Tables.**  GFM pipe-table rows are parsed as individual cell segments.
  Cells containing short values (single numbers, "NA", etc.) typically
  aren't unique enough for phase 1 and are matched in phase 2.
- **Equations.**  LaTeX math environments (`$...$`, `$$...$$`) are treated
  as opaque strings; the normalisation discards all non-alphanumeric
  characters, so an equation only matches if the variable names or numbers
  it contains appear verbatim in the PDF's text layer.  Image-rendered
  equations (common in older PDFs) will not match.
- **Accents and ligatures.**  The shared NFKD-aware normaliser
  decomposes accented letters to their base form (`Töpf` → `topf`) and
  expands ligatures (`ﬁ` → `fi`) on both sides of the alignment, so
  publishers that compose differently from the rendered PDF still align
  cleanly.
