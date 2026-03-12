# anchorite

**Spatial text alignment for document AI pipelines.**

`anchorite` aligns generated Markdown text back to the physical bounding boxes that an OCR engine found on the original document pages. It bridges the gap between generative AI (which produces high-quality, readable Markdown) and traditional OCR (which provides precise coordinates) by finding where each OCR word or phrase appears in the generated output.

---

## The problem

Modern document AI pipelines often combine two sources:

1. **A generative model** (Gemini, Claude, GPT-4) that reads a page image and produces clean, well-structured Markdown.
2. **An OCR engine** (Google Document AI, Tesseract, Docling) that identifies individual words and their bounding boxes on the page.

The generative model's output is readable and accurate but has no coordinates. The OCR output has precise coordinates but poor structure. `anchorite` fuses them: it takes the Markdown as the ground truth for text content and finds the corresponding bounding box for each OCR word or phrase within it.

---

## Installation

```shell
pip install anchorite
```

---

## Core concepts

**`Anchor`** — a piece of OCR text with its location: a `text` string, a `page` number (0-indexed), and a `BBox` (bounding box in 0–1000 normalised coordinates).

**`BBox`** — a bounding box `(top, left, bottom, right)`.

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

### 2. Resolve quotes to coordinates

Given annotated Markdown and a list of verbatim quotes (e.g. extracted by an LLM), find the bounding boxes that each quote covers. Useful for grounding LLM citations.

```python
locations = anchorite.resolve(annotated, quotes=["Observations of a Nebula"])
# {"Observations of a Nebula": [(0, BBox(52, 120, 68, 880))]}
```

`resolve` uses fuzzy iterative matching so it tolerates minor transcription differences. Each quote maps to a list of `(page, BBox)` pairs — one per distinct OCR anchor the quote overlaps.

### 3. Strip annotations for downstream validation

`strip` is the inverse of `annotate`. It removes the `<span>` tags and returns a plain-text string alongside a validation map you can use to check whether a generated quote is grounded in the source document.

```python
stripped = anchorite.strip(annotated)
# stripped.plain_text  — Markdown with tags removed
# stripped.validation_map  — list of (start, end, Anchor) in plain_text
```

### 4. Orchestrated multi-page processing

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

Before any alignment, text is normalised to a reduced alphabet: letters are lowercased, all non-alphanumeric characters (punctuation, whitespace variants) are mapped to a single space, and consecutive spaces are collapsed to one. This makes the alignment robust to minor formatting differences between the OCR text and the generated Markdown (e.g. hyphenation, ligatures, smart quotes).

### Document fragmentation

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
