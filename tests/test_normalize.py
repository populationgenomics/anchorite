"""Tests for the shared text normalisation module.

The normalisers underpin the alignment guarantee: a quote produced from
a piece of Markdown will align against the same Markdown its bboxes
were derived from precisely because every alignment site funnels
through these functions.
"""

from anchorite import normalize

# ---------------------------------------------------------------------------
# Normalisation: HTML tags must be zero-width
# ---------------------------------------------------------------------------


class TestHtmlTagsInNormalisation:
    """HTML tags surviving into Markdown (`<sup>`, `<a id="...">`, etc.) must
    not contribute alphanum bytes to the alignment string when *strip_html*
    is True — otherwise their tag names line up with letters in the PDF and
    trash the alignment.  *strip_html* defaults to False because pdfium-
    extracted PDF text contains literal ``<`` / ``>`` characters when those
    glyphs appear in the document (e.g. ``p < 0.05``); stripping them would
    silently drop real content.
    """

    def test_strict_strips_when_enabled(self) -> None:
        text = "Author<sup>1</sup>"
        plain = "Author1"
        assert (
            normalize.normalize_strict(text, strip_html=True)[0]
            == normalize.normalize_strict(plain, strip_html=True)[0]
        )

    def test_loose_strips_when_enabled(self) -> None:
        text = "Author<sup>1</sup>"
        plain = "Author1"
        assert (
            normalize.normalize_loose(text, strip_html=True)[0] == normalize.normalize_loose(plain, strip_html=True)[0]
        )

    def test_default_does_not_strip(self) -> None:
        # PDF-side default: `<` / `>` are literal characters and must contribute
        # like any other punctuation.  ``a<b`` (PDF rendering of ``a < b``)
        # normalises like ``a b``.
        norm, _ = normalize.normalize_strict("a<b")
        assert norm == normalize.normalize_strict("a b")[0]

    def test_idx_map_points_into_original_text(self) -> None:
        text = "Author<sup>1</sup>"
        norm_bytes, idx_map = normalize.normalize_loose(text, strip_html=True)
        # 7 alphanum bytes (a,u,t,h,o,r,1) plus one trailing sentinel.
        assert len(norm_bytes) == 7
        assert list(idx_map) == [0, 1, 2, 3, 4, 5, 11, len(text)]

    def test_anchor_tag_with_id_zero_width(self) -> None:
        text = '<a id="R1"></a>\n## References'
        plain = "## References"
        assert (
            normalize.normalize_strict(text, strip_html=True)[0]
            == normalize.normalize_strict(plain, strip_html=True)[0]
        )

    def test_lone_lt_gt_treated_as_punctuation(self) -> None:
        # A bare ``<`` or ``>`` (no closing ``>``) is left alone even when
        # strip_html is True — the regex requires a complete tag.
        norm_bytes, _ = normalize.normalize_strict("a < b", strip_html=True)
        assert norm_bytes == normalize.normalize_strict("a   b", strip_html=True)[0]


# ---------------------------------------------------------------------------
# Normalisation: inline Markdown link wrappers must be zero-width
# ---------------------------------------------------------------------------


class TestMarkdownLinksInNormalisation:
    """Inline Markdown links ``[text](url)`` render as just ``text`` in the
    PDF.  A naïve normalisation contributes both ``text`` *and* the URL,
    which doubles the autolink footprint (``[https://x.org](https://x.org)``
    becomes two copies of ``httpsxorg``) and silently inflates the segment
    far beyond what the PDF holds.  When *strip_html* is True the wrapper
    portions (``[``, ``](url)``) must drop out, leaving only ``text``.
    """

    def test_strict_strips_link_wrapper(self) -> None:
        text = "see [phosphosite](https://www.phosphosite.org/homeAction.action) for details"
        plain = "see phosphosite for details"
        assert (
            normalize.normalize_strict(text, strip_html=True)[0]
            == normalize.normalize_strict(plain, strip_html=True)[0]
        )

    def test_loose_strips_link_wrapper(self) -> None:
        text = "see [phosphosite](https://www.phosphosite.org/homeAction.action) for details"
        plain = "see phosphosite for details"
        assert (
            normalize.normalize_loose(text, strip_html=True)[0] == normalize.normalize_loose(plain, strip_html=True)[0]
        )

    def test_autolink_with_duplicate_text_collapses(self) -> None:
        # Link text equals the URL: the wrapper must collapse so the URL
        # contributes only once, matching what the PDF actually shows.
        text = "[https://x.org](https://x.org)"
        plain = "https://x.org"
        assert (
            normalize.normalize_loose(text, strip_html=True)[0] == normalize.normalize_loose(plain, strip_html=True)[0]
        )

    def test_citation_reference_normalises_to_marker(self) -> None:
        # Pubmed-style citation links ``[6](#R6)`` should normalise to just
        # ``6`` — the digit the PDF actually renders.
        text = "...nucleus ([6](#R6))."
        plain = "...nucleus (6)."
        assert (
            normalize.normalize_loose(text, strip_html=True)[0] == normalize.normalize_loose(plain, strip_html=True)[0]
        )

    def test_html_inside_link_text_is_also_stripped(self) -> None:
        # ``[<sup>1</sup>](#R1)`` should normalise to ``1`` — both the link
        # wrapper and the surviving HTML tag are zero-width.
        text = "see [<sup>1</sup>](#R1) above"
        plain = "see 1 above"
        assert (
            normalize.normalize_loose(text, strip_html=True)[0] == normalize.normalize_loose(plain, strip_html=True)[0]
        )

    def test_default_does_not_strip_link(self) -> None:
        # PDF-side default: ``[`` / ``]`` / ``(`` / ``)`` are literal glyphs
        # and contribute as ordinary punctuation when *strip_html* is False.
        text = "[a](b)"
        norm_with = normalize.normalize_loose(text, strip_html=True)[0]
        norm_without = normalize.normalize_loose(text, strip_html=False)[0]
        assert norm_with != norm_without


# ---------------------------------------------------------------------------
# Normalisation: NFKD compatibility decomposition
# ---------------------------------------------------------------------------


class TestNfkdNormalisation:
    """Each input character is NFKD-decomposed before classification, so
    accented letters keep their base form, ligatures expand, superscript
    digits become plain digits, and Mathematical Alphanumeric Symbols map
    back to ASCII.  Without NFKD these all dropped out as non-ASCII and
    shrank the alignable sequence — short segments hit the phase-1
    minimum-length cutoff and never matched.
    """

    def test_accented_letters_keep_base(self) -> None:
        # Töpf used to normalise to 'tpf' (ö dropped); now keeps 'topf'.
        assert normalize.normalize_loose("Töpf")[0] == normalize.normalize_loose("Topf")[0]
        assert normalize.normalize_loose("Müller")[0] == normalize.normalize_loose("Muller")[0]
        assert normalize.normalize_loose("naïve")[0] == normalize.normalize_loose("naive")[0]
        assert normalize.normalize_loose("Göngör")[0] == normalize.normalize_loose("Gongor")[0]

    def test_ligatures_expand(self) -> None:
        # ﬁnal (single ligature glyph) used to drop entirely; now becomes 'final'.
        assert normalize.normalize_loose("ﬁnal")[0] == normalize.normalize_loose("final")[0]
        assert normalize.normalize_loose("eﬃcient")[0] == normalize.normalize_loose("efficient")[0]

    def test_superscript_digits_become_plain(self) -> None:
        # ⁶¹RNRKRKAEPY⁷⁰ used to lose the superscripts entirely.
        assert normalize.normalize_loose("⁶¹RNRKRKAEPY⁷⁰")[0] == normalize.normalize_loose("61RNRKRKAEPY70")[0]
        assert normalize.normalize_loose("H²O")[0] == normalize.normalize_loose("H2O")[0]

    def test_math_alphanumeric_symbols(self) -> None:
        # Mathematical Italic capital S (𝑆) should normalise to 's', etc.
        assert normalize.normalize_loose("𝑆ensitivity")[0] == normalize.normalize_loose("Sensitivity")[0]

    def test_idx_map_for_one_to_many_decomposition(self) -> None:
        # 'ﬁnal' has 4 chars; under NFKD ``ﬁ`` decomposes to ``fi`` (two
        # bytes), so the normalised output has 5 bytes — but both
        # decomposed letters of ``ﬁ`` map back to its single original
        # offset 0.
        norm, idx_map = normalize.normalize_loose("ﬁnal")
        assert len(norm) == 5
        # Original offsets for f, i, n, a, l: ﬁ is at 0; n, a, l at 1, 2, 3.
        assert list(idx_map) == [0, 0, 1, 2, 3, 4]

    def test_combining_marks_dropped(self) -> None:
        # NFKD decomposes 'ö' to 'o' + U+0308 combining diaeresis.  The
        # combining mark must not contribute an output byte.
        norm, idx_map = normalize.normalize_loose("ö")
        assert len(norm) == 1  # just 'o', no combining mark
        assert list(idx_map) == [0, 1]

    def test_strict_mode_emits_space_for_pure_punctuation(self) -> None:
        # NFKD doesn't decompose dashes/quotes; they still collapse to a
        # space in strict mode — preserve the existing behaviour.
        norm, _ = normalize.normalize_strict("a — b")
        assert norm == normalize.normalize_strict("a   b")[0]

    def test_decomposed_input_normalises_like_precomposed_strict(self) -> None:
        # ``Töpf`` arriving as decomposed [T, o, U+0308, p, f] must yield the
        # same byte sequence as the precomposed form.  Without the combining-
        # mark guard, strict mode would emit a space for U+0308 and produce
        # ``to pf`` instead of ``topf``.
        decomposed = "T" + "o" + "̈" + "p" + "f"
        precomposed = "Töpf"
        assert normalize.normalize_strict(decomposed)[0] == normalize.normalize_strict(precomposed)[0]

    def test_decomposed_input_normalises_like_precomposed_loose(self) -> None:
        decomposed = "T" + "o" + "̈" + "p" + "f"
        precomposed = "Töpf"
        assert normalize.normalize_loose(decomposed)[0] == normalize.normalize_loose(precomposed)[0]

    def test_combining_mark_omits_idx_map_entry(self) -> None:
        # The combining mark must produce no normalised byte and no idx_map
        # entry — only the base letters land in the output.
        decomposed = "öp"  # o + combining diaeresis + p (3 chars)
        norm, idx_map = normalize.normalize_loose(decomposed)
        assert len(norm) == 2
        # Original positions: o=0, p=2 (combining mark at 1 is skipped).
        assert list(idx_map) == [0, 2, len(decomposed)]
