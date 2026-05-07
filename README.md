# anchorite

<img src="https://raw.githubusercontent.com/populationgenomics/anchorite/main/docs/source/_static/anchorite.svg" alt="anchorite" width="200">

**Spatial text alignment for document AI pipelines.**

`anchorite` connects generated Markdown text to physical bounding boxes on the source document pages. It bridges the gap between text-based representations (LLM-generated Markdown, OCR layout markup, JATS XML rendered to Markdown) and the precise coordinates a viewer needs to highlight quoted text on the original page.

---

## The problem

Modern document AI pipelines combine readable text with physical coordinates from a variety of sources:

1. **A generative model** (Gemini, Claude, GPT-4) that reads a page image and produces clean, well-structured Markdown.
2. **An OCR engine** (Google Document AI, Tesseract, Docling) that identifies words and their bounding boxes.
3. **Native PDF text** extracted via `pypdfium2` from publisher PDFs.
4. **JATS XML** distributed by PMC and other publishers, alongside the same paper's PDF.

Most pipelines have abundant *content* but no coordinates, or precise *coordinates* but poor structure. `anchorite` fuses them. It supports two complementary directions:

- **OCR anchors → Markdown:** align a list of OCR-derived `Anchor` objects to a Markdown string and inject coordinate spans (`align`, `annotate`).
- **Markdown → PDF char layer:** align Markdown segments back to per-character bounding boxes extracted directly from a PDF (`md_association.associate`).

Both produce the same `Anchor` data shape, so downstream resolution (`resolve`, `resolve_quote`, `quote_locates`) is identical regardless of which path produced the anchors.

---

## Installation

```shell
pip install anchorite
```

---

## Core concepts

**`Anchor`** — a fragment of text linked to a page region: a `text` string, a `page` number (0-indexed), and a tuple of `BBox`es (one per visual line the anchor covers).

**`BBox`** — a bounding box `(top, left, bottom, right)`, integer coordinates in 0–1000 normalised page space.

**`SpanAnchor`** — an anchor paired with its character range in the source Markdown: `(span: (int, int), page: int, box: BBox)`. Lit-manager-style sidecar formats that store `[{"span": [s, e], "page": p, "rect": [t, l, b, r]}, …]` map directly onto a list of `SpanAnchor`s, which `resolve_quote` consumes.

**`alignment`** — a `dict[Anchor, tuple[int, int]]` mapping each anchor to a `(start, end)` character span in the Markdown string.

---

## Workflows

### 1. Align and annotate

The most common workflow: align OCR anchors to Markdown, then inject coordinate spans.

```python
import anchorite

anchors = [
    anchorite.Anchor(text="Observations of a Nebula", page=0, box=anchorite.BBox(52, 120, 68, 880)),
    anchorite.Anchor(text="Edwin Hubble", page=0, box=anchorite.BBox(80, 340, 92, 660)),
]

markdown = "# Observations of a Nebula\n\n*Edwin Hubble*, 1929"

alignment = anchorite.align(anchors, markdown)
annotated = anchorite.annotate(markdown, alignment)
# <span data-bbox="52,120,68,880" data-page="0">Observations of a Nebula</span>
# <span data-bbox="80,340,92,660" data-page="0">Edwin Hubble</span>
```

The annotated Markdown is otherwise valid Markdown and can be rendered normally; the `<span>` tags carry coordinate metadata as HTML attributes.

### 2. Derive anchors from a PDF + Markdown directly

When the Markdown is independently authoritative (JATS XML rendered to Markdown, hand-curated content, an LLM rewrite that you trust), you can skip the OCR engine and align the Markdown segments to per-character bounding boxes that `pypdfium2` extracts straight from the PDF. `md_association.associate` handles the segmentation and two-phase alignment in one call.

```python
import pathlib
from anchorite.md_association import associate

anchors = associate(
    pathlib.Path("paper.pdf"),
    pathlib.Path("paper.md").read_text(),
)
# Returns a list[Anchor], one per matched Markdown segment, in document order.
```

Page-break markers (`<!--page-->`) in the input are used as a search-window hint for cost; they're optional. Without them, phase 1 falls back to searching every page. See `docs/source/md_association.md` for the algorithm and tunables.

### 3. Resolve quotes to coordinates

Given a list of verbatim quotes (e.g. citations extracted by an LLM), find the bounding boxes that each quote covers. Two API shapes depending on what you have on hand:

**`resolve_quote`** — when you've stored Markdown and a list of `(span, page, box)` records on disk (the typical sidecar shape):

```python
spans = [
    anchorite.SpanAnchor(span=(0, 25), page=0, box=anchorite.BBox(10, 10, 20, 20)),
    anchorite.SpanAnchor(span=(25, 44), page=1, box=anchorite.BBox(30, 30, 40, 40)),
]
located = anchorite.resolve_quote(markdown, spans, "quick brown fox jumps")
# [(0, BBox(10, 10, 20, 20))]
```

**`resolve`** — when you've stored *annotated* Markdown (with `<span data-bbox=…>` tags inline, as produced by `annotate`):

```python
locations = anchorite.resolve(annotated, quotes=["Observations of a Nebula"])
# {"Observations of a Nebula": [(0, BBox(52, 120, 68, 880))]}
```

Both use the same fuzzy iterative Smith-Waterman pipeline and the same shared normaliser used during anchor generation, so a quote that aligned cleanly at ingest aligns cleanly here too. Each quote maps to a sorted list of `(page, BBox)` pairs — one per distinct anchor the quote overlaps.

For callers that only need to know whether a quote can be grounded (LLM tool-call validation, "did the model hallucinate this?"), `quote_locates(markdown, quote) -> bool` skips the span-overlap step:

```python
if anchorite.quote_locates(markdown, quote):
    ...  # the LLM's quote actually appears in the source
```

**`PdfIndex`** — when you have raw PDF bytes and a list of quotes and want to skip the Markdown / Anchor pipeline entirely (e.g. the upstream LLM emitted citations against a PDF you already have on disk):

```python
index = anchorite.PdfIndex(pdf_bytes)
located = index.resolve(["Observations of a Nebula", "first 19 nebulae"])
# {"Observations of a Nebula": [(0, BBox(52, 120, 68, 880))],
#  "first 19 nebulae": [(2, BBox(...)), ...]}
```

Construction extracts per-character bounding boxes from every page (the expensive step); `.resolve` is then cheap and batches all quotes through a single `seq_smith.local_global_align_many` pass. Pages are 0-indexed.

You can optionally pass a Markdown transcription at construction time. The Markdown is aligned against the extracted PDF chars and used to clean up the cached flat string — chars the LLM didn't transcribe (running heads, page numbers, footnote markers) get dropped. The Markdown is then discarded; the index stays Markdown-free, but the cache is higher quality:

```python
index = anchorite.PdfIndex(pdf_bytes, markdown=llm_emitted_markdown)
```

### 4. Strip annotations for downstream validation

`strip` is the inverse of `annotate`. It removes the `<span>` tags and returns a plain-text string alongside a validation map you can use to check whether a generated quote is grounded in the source document.

```python
stripped = anchorite.strip(annotated)
# stripped.plain_text  — Markdown with tags removed
# stripped.validation_map  — list of (start, end, Anchor) in plain_text
```

### 5. Orchestrated multi-page processing

For pipelines that process multi-page documents, `process_document` handles parallelism, page-chunk assembly, and alignment in one call. You supply pre-chunked document data and implement two provider protocols.

```python
import asyncio
import anchorite
from anchorite.document import DocumentChunk
from anchorite.providers import MarkdownProvider, AnchorProvider

class MyMarkdownProvider:
    async def generate_markdown(self, chunk: DocumentChunk) -> str:
        # Call your LLM or OCR layout model here
        ...

class MyAnchorProvider:
    async def generate_anchors(self, chunk: DocumentChunk) -> list[anchorite.Anchor]:
        # Call your OCR engine here and return Anchor objects
        ...

# Chunk the document yourself (e.g. 10 pages per chunk)
chunks = list(anchorite.document.chunks("paper.pdf", page_count=10))

result = asyncio.run(anchorite.process_document(
    chunks,
    MyMarkdownProvider(),
    MyAnchorProvider(),
))

print(result.coverage_percent)   # fraction of Markdown covered by aligned anchors
annotated = result.annotate()    # AlignmentResult.annotate() calls anchorite.annotate internally
```

`process_document` runs the markdown and anchor providers concurrently across all chunks using `asyncio.gather`, then aligns the assembled full-document Markdown against the complete anchor set.

#### Provider protocols

```python
class MarkdownProvider(Protocol):
    async def generate_markdown(self, chunk: DocumentChunk) -> str: ...

class AnchorProvider(Protocol):
    async def generate_anchors(self, chunk: DocumentChunk) -> list[Anchor]: ...
```

Both are structural protocols — no inheritance required, duck typing works.

#### Document chunking

`anchorite.document.chunks(source, *, page_count, mime_type)` splits a PDF into sub-documents of `page_count` pages each. `source` can be a file path, URL, `bytes`, or a file-like object. Images (PNG, JPEG, WebP) are yielded as a single chunk unchanged.

You do not have to use `anchorite.document.chunks`. If your pipeline already produces chunks (for example, Docling's own document parser), create `DocumentChunk` objects directly:

```python
from anchorite.document import DocumentChunk

chunk = DocumentChunk(
    document_sha256="abc123...",
    start_page=0,
    end_page=10,
    data=pdf_bytes,
    mime_type="application/pdf",
)
```

---

## API reference

### `anchorite.align(anchors, markdown, *, uniqueness_threshold, min_overlap)`

Aligns a sequence of `Anchor` objects to a Markdown string. Returns `dict[Anchor, tuple[int, int]]`.

| Parameter | Default | Description |
|---|---|---|
| `uniqueness_threshold` | `0.5` | An anchor is accepted only if its best-match score exceeds this fraction of its second-best score. Higher values demand more unique matches. |
| `min_overlap` | `0.9` | Minimum fraction of the anchor's normalised length that must be covered by the alignment. |

### `anchorite.annotate(markdown, alignment)`

Injects `<span data-bbox="t,l,b,r" data-page="N">` tags into Markdown at the positions given by `alignment`. Handles overlapping and nested spans. Math blocks (`$...$`, `$$...$$`) are detected and span boundaries are snapped to their edges so LaTeX is not broken.

### `anchorite.strip(annotated_md)`

Removes `<span>` tags and returns a `StrippedMarkdown` with fields:

- `plain_text`: the Markdown with all tags removed
- `validation_map`: sorted list of `(start, end, Anchor)` tuples in `plain_text` coordinates

### `anchorite.resolve(annotated_md, quotes)`

Resolves a list of verbatim quote strings to their bounding boxes using fuzzy iterative Smith-Waterman alignment against the stripped text. Returns `dict[str, list[tuple[int, BBox]]]` mapping each quote to a list of `(page, BBox)` pairs.

### `anchorite.resolve_quote(markdown, spans, quote, *, min_score, warn_coverage, fail_coverage)`

The bbox-records variant of `resolve`. Locates `quote` in `markdown` via the same iterative SW pipeline, then returns every `SpanAnchor` whose `span` overlaps the matched region as a sorted, de-duplicated `[(page, BBox), …]` list. Suitable for callers that store Markdown and bbox records separately rather than as inline `<span>` tags.

| Parameter | Default | Description |
|---|---|---|
| `min_score` | `15` | Reject SW alignments scoring below this threshold. |
| `warn_coverage` | `0.5` | Log a warning when matched coverage falls below this fraction. |
| `fail_coverage` | `0.3` | Return `[]` when matched coverage falls below this fraction. |

### `anchorite.quote_locates(markdown, quote, *, min_score, fail_coverage)`

Boolean variant of `resolve_quote` for grounding checks. Returns `True` iff the quote aligns with sufficient confidence; no span list required.

### `anchorite.PdfIndex(pdf_data, *, markdown=None)`

A pre-extracted PDF index for batched quote-to-bbox resolution. Construction reads per-character bounding boxes from every page once; `.resolve(quotes, *, min_score, num_threads)` then aligns every quote in a single `seq_smith.local_global_align_many` call and returns `dict[str, list[tuple[int, BBox]]]`. Pages are 0-indexed. Empty / whitespace / unmatchable quotes map to `[]`.

When `markdown` is supplied at construction, it's used to clean up the cached flat string (matched-only chars in Markdown order, untranscribed runs dropped) and then discarded — the index stays Markdown-free.

| Parameter | Default | Description |
|---|---|---|
| `min_score` | `15` | Reject alignments scoring below this. |
| `num_threads` | `None` | Thread count for batched alignment; `None` defers to seq_smith's default. |

Construction is not thread-safe (PDFium isn't); serialise concurrent `PdfIndex(...)` calls in the caller. `.resolve` after construction is thread-safe.

### `anchorite.md_association.associate(pdf_path, markdown, *, min_score, return_pass_info)`

Aligns Markdown segments (sentences, headings, list items, table cells) to per-character bounding boxes extracted from a PDF via `pypdfium2`. Returns `list[Anchor]` in document order. With `return_pass_info=True` returns `(anchors, passes)`, where `passes[i]` is `1` for a phase-1 (conservative HSP) match or `2` for a phase-2 (page-constrained) match.

`<!--page-->` markers in the Markdown are an optional search-window hint; without them, phase 1 searches every page.

### `anchorite.process_document(chunks, markdown_provider, anchor_provider, *, ...)`

Orchestrates multi-chunk document alignment. Returns `AlignmentResult`.

| Parameter | Default | Description |
|---|---|---|
| `alignment_uniqueness_threshold` | `0.5` | Passed to `align`. |
| `alignment_min_overlap` | `0.9` | Passed to `align`. |
| `renumber` | `True` | Renumber `<!--table-->` and `<!--figure-->` markers across chunks before joining. |

---

## Algorithm

### Normalisation

Before any alignment, text is normalised to a reduced alphabet through a single shared pipeline used by every entry point in the package — `align`, `associate`, `resolve`, `resolve_quote`, `quote_locates`. Sharing the normaliser is what guarantees that a quote produced from a piece of Markdown will align against the same Markdown its bboxes were derived from.

Each input character runs through:

1. **NFKD compatibility decomposition.** Accented letters split into a base letter plus combining marks (`Töpf` → `T`, `o`, U+0308, `p`, `f`); ligatures expand (`ﬁ` → `fi`); superscript and subscript digits become plain digits (`²` → `2`); Mathematical Alphanumeric Symbols map to ASCII (`𝑆` → `S`).
2. **ASCII-alphanumeric filter.** Decomposed characters that aren't `[a-z0-9]` (case-folded) are dropped — combining marks, punctuation, and unmapped Unicode all fall away.
3. **Combining-mark guard.** Unicode `M*`-category characters (the residual combining marks from the previous step) are zero-width: they emit no alphanum output *and* don't trigger the punctuation-collapses-to-space branch in strict mode. This makes precomposed (`ö`) and decomposed (`o` + U+0308) input produce identical alignment bytes.
4. **Strict vs. loose alphabet.** Strict normalisation collapses non-alphanumeric runs to a single space; loose normalisation drops spaces entirely. Loose is the fallback when strict can't recover the segment text (e.g. letter-spaced display headings — `C A S E  R E P O R T` aligns to `CASEREPORT` only when spaces are dropped).
5. **HTML-tag stripping (Markdown side only).** When `strip_html=True` is set — automatically the case for every Markdown call site — `<…>` tag spans are zero-width, so `Author<sup>1</sup>` aligns as `author1` rather than `authorsup1sup`. PDF-side text never strips tags: a literal `<` or `>` extracted from the PDF is real content (`p < 0.05`, `Vol < 100`).

PDF char extraction adds two more steps before normalisation:

- **Soft-hyphen reconnection.** End-of-line `induc-` followed by start-of-line `tion` reconnects to `induction` rather than emitting `induc- tion`. Triggered only when both surrounding glyphs are alphabetic, so numeric ranges like `2009- 2010` keep the hyphen.
- **Line-break space insertion.** PDFium emits no whitespace at line breaks (the next char's *x* coordinate jumps backward instead). The flat-string builder detects line breaks (next char's baseline drops by ≥ 50 % of font size, or its *x* sits to the left of the current char) and inserts a space, so `we` + `identified` doesn't concatenate to `weidentified`.

### Document fragmentation

(The remainder of this section describes the `align` algorithm — OCR anchors → Markdown spans. The complementary algorithm, `md_association.associate` — Markdown segments → PDF char bboxes — has its own document at `docs/source/md_association.md`.)

The Markdown is split at HTML comment markers (e.g. `<!--page-->`, `<!--table: 1-->`) into contiguous fragments. Each fragment inherits a page range from its position in the assembled document, which is used to restrict which anchors can match it — anchors are only compared against fragments whose page range includes the anchor's page number.

### Iterative alignment

The core loop runs until all anchors are matched or no further progress is made.

**Pass 1 — ungapped alignment.** Each unmatched anchor is aligned against each compatible document fragment using ungapped Smith-Waterman local alignment (via `seq_smith.top_k_ungapped_local_align_many`, retrieving the top-2 scores per anchor per fragment). An anchor is promoted to a high-confidence candidate only if both conditions hold:

- *Overlap*: the best-match score covers at least `min_overlap` of the anchor's normalised length.
- *Uniqueness*: the best-match score exceeds `uniqueness_threshold` × the second-best score, ensuring the match is not ambiguous.

**Subsequent passes — gapped alignment.** The same candidate-selection logic is repeated using semi-global alignment (`seq_smith.local_global_align_many`), which allows gaps within the alignment. This recovers anchors that the LLM paraphrased or reformatted slightly.

### Span assignment

Once a set of high-confidence candidates is identified for a fragment, each candidate is assigned a precise character range within the fragment. Candidates are processed in descending alignment score order and are accepted only if:

1. At least 90% of the aligned positions are exact character matches (no-gap criterion within the assignment step).
2. The proposed range is *page-consistent*: anchors from earlier pages must map to earlier positions in the Markdown than anchors from later pages.
3. At least 90% of the proposed range is *new* coverage — not already claimed by a higher-scoring anchor in the same fragment.

The assigned range is mapped back from normalised-character coordinates to original Markdown character offsets via the `normalized_to_source` index.

### Fragment splitting

After assignment, any portion of a document fragment not covered by any accepted anchor becomes a new sub-fragment for subsequent iterations. This allows later iterations to focus on progressively smaller uncovered regions, recovering matches that were hidden by initially ambiguous context.

### Result

The final result is a `dict[Anchor, (start, end)]` giving the character span in the original Markdown for each successfully aligned anchor. Anchors that could not be matched with sufficient confidence are omitted.
