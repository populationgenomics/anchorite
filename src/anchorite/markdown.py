import collections
import re


def renumber_markers(markdown_chunks: list[str]) -> list[str]:
    """
    Renumbers <!--table--> and <!--figure--> markers across multiple chunks.

    Transforms e.g. <!--table--> into <!--table: 1-->.
    """
    counters: collections.Counter[str] = collections.Counter()

    def _renumber(match: re.Match) -> str:
        kind = match.group(1)
        counters[kind] += 1
        return f"<!--{kind}: {counters[kind]}-->"

    return [re.sub(r"<!--(table|figure)-->", _renumber, chunk_text) for chunk_text in markdown_chunks]
