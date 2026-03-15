"""Prototype: align Markdown segments to PDF characters and visualise results."""

import pathlib
import sys

import pypdfium2 as pdfium
from PIL import ImageDraw

from anchorite.anchors import Anchor
from anchorite.md_association import associate, parse_markdown_segments

_COLOURS = [
    (220,  50,  50, 200),
    ( 50, 130, 220, 200),
    ( 40, 180,  80, 200),
    (200, 130,  30, 200),
    (150,  50, 200, 200),
]


# Fill alpha by pass: pass 1 = darkest, pass 3 = lightest.
_PASS_FILL_ALPHA = {1: 80, 3: 25}


def visualize_page(
    page: pdfium.PdfPage,
    anchors: list[Anchor],
    output_path: pathlib.Path,
    scale: float = 2.0,
    passes: list[int] | None = None,
) -> None:
    page_height = page.get_height()
    page_width = page.get_width()

    bitmap = page.render(scale=scale)
    img = bitmap.to_pil()
    draw = ImageDraw.Draw(img, "RGBA")

    for i, anchor in enumerate(anchors):
        r, g, b, _ = _COLOURS[i % len(_COLOURS)]
        pass_num = passes[i] if passes else 1
        fill_alpha = _PASS_FILL_ALPHA.get(pass_num, 50)
        outline = (r, g, b, 230)
        fill = (r, g, b, fill_alpha)

        # BBox is 0-1000 normalised, top-left origin.
        for box in anchor.boxes:
            x0 = box.left / 1000 * page_width * scale
            y0 = box.top / 1000 * page_height * scale
            x1 = box.right / 1000 * page_width * scale
            y1 = box.bottom / 1000 * page_height * scale
            draw.rectangle([x0, y0, x1, y1], outline=outline, fill=fill, width=2)

    img.save(output_path)


def main(pdf_path: pathlib.Path, md_path: pathlib.Path) -> None:
    markdown = md_path.read_text()
    segments = parse_markdown_segments(markdown)
    print(f"{len(segments)} segments parsed")

    anchors, passes = associate(pdf_path, markdown, return_pass_info=True)
    print(f"{len(anchors)} anchors matched\n")

    doc = pdfium.PdfDocument(pdf_path)
    num_pages = len(doc)

    for page_idx in range(num_pages):
        idx = [(a, p) for a, p in zip(anchors, passes) if a.page == page_idx]
        page_anchors = [a for a, _ in idx]
        page_passes = [p for _, p in idx]
        print(f"Page {page_idx}: {len(page_anchors)} anchors")
        for a, p in zip(page_anchors, page_passes):
            print(f"  pass={p} boxes={a.boxes}  {a.text[:80]!r}")

        output_path = pdf_path.with_stem(f"{pdf_path.stem}.p{page_idx}").with_suffix(".md.png")
        visualize_page(doc[page_idx], page_anchors, output_path, passes=page_passes)
        print(f"  -> {output_path}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <pdf> <markdown>")
        sys.exit(1)
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
