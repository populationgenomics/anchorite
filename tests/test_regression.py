import json
import os
import pathlib

import pytest

import anchorite
from anchorite.document import DocumentChunk
from anchorite.types import Anchor, BBox

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


class MockMarkdownProvider:
    def __init__(self, responses: list) -> None:
        self.responses = responses

    async def generate_markdown(self, chunk: DocumentChunk) -> str:
        idx = chunk.start_page // 10
        return str(self.responses[idx])


class MockAnchorProvider:
    def __init__(self, bboxes_data: list) -> None:
        self.bboxes_data = bboxes_data

    async def generate_anchors(self, chunk: DocumentChunk) -> list[Anchor]:
        idx = chunk.start_page // 10
        chunk_data = self.bboxes_data[idx]
        return [
            Anchor(
                text=b["text"],
                page=b["page"],
                box=BBox(**b["box"]),
            )
            for b in chunk_data
        ]


@pytest.mark.asyncio
async def test_hubble_regression() -> None:
    # Load fixtures
    with open(FIXTURES_DIR / "hubble_markdown_chunks.json") as f:
        gemini_responses = json.load(f)

    with open(FIXTURES_DIR / "hubble_anchors.json") as f:
        docai_bboxes = json.load(f)

    markdown_provider = MockMarkdownProvider(gemini_responses)
    bbox_provider = MockAnchorProvider(docai_bboxes)

    # Mock chunks so we don't need a real PDF
    mock_chunks = [
        DocumentChunk(
            document_sha256="fake",
            start_page=i * 10,
            end_page=(i + 1) * 10,
            data=b"",
            mime_type="application/pdf",
        )
        for i in range(len(gemini_responses))
    ]

    result = await anchorite.process_document(
        mock_chunks,
        markdown_provider,
        bbox_provider,
        renumber=False,
    )

    output_md = result.annotate()

    golden_path = FIXTURES_DIR / "hubble_golden.md"

    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.write_text(output_md)

    expected = golden_path.read_text()

    assert output_md == expected
