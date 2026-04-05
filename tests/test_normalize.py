"""Tests for extractmark.normalize -- 6-step normalization pipeline."""

from extractmark.normalize import (
    normalize,
    _step1_strip_tokens,
    _step2_normalize_unicode,
    _step3_collapse_whitespace,
    _step4_normalize_tables,
    _step5_normalize_formulas,
    _step6_strip_headers_footers,
)


class TestStep1StripTokens:
    def test_strip_im_tokens(self):
        assert _step1_strip_tokens("<|im_start|>Hello<|im_end|>") == "Hello"

    def test_strip_eos_tokens(self):
        assert _step1_strip_tokens("<s>Hello</s>") == "Hello"

    def test_strip_role_tokens(self):
        assert _step1_strip_tokens("<|assistant|>response<|endoftext|>") == "response"

    def test_strip_markdown_fences(self):
        assert _step1_strip_tokens("```markdown\n# Title\n```") == "# Title"

    def test_strip_grounding_token(self):
        assert _step1_strip_tokens("<|grounding|>text") == "text"

    def test_no_tokens(self):
        assert _step1_strip_tokens("plain text") == "plain text"

    def test_whitespace_after_strip(self):
        """After stripping tokens, leading/trailing whitespace is trimmed."""
        assert _step1_strip_tokens("  <s> hello </s>  ") == "hello"


class TestStep2NormalizeUnicode:
    def test_nfc_normalization(self):
        # é composed vs decomposed
        composed = "\u00e9"   # é (single codepoint)
        decomposed = "e\u0301"  # e + combining accent
        assert _step2_normalize_unicode(decomposed) == composed

    def test_plain_ascii(self):
        assert _step2_normalize_unicode("hello") == "hello"

    def test_preserves_cjk(self):
        text = "中文文本"
        assert _step2_normalize_unicode(text) == text


class TestStep3CollapseWhitespace:
    def test_collapse_blank_lines(self):
        text = "a\n\n\n\nb"
        assert _step3_collapse_whitespace(text) == "a\n\nb"

    def test_collapse_spaces(self):
        assert _step3_collapse_whitespace("a   b   c") == "a b c"

    def test_collapse_tabs(self):
        assert _step3_collapse_whitespace("a\t\tb") == "a b"

    def test_strip_whitespace_only_lines(self):
        text = "a\n   \nb"
        assert _step3_collapse_whitespace(text) == "a\n\nb"

    def test_preserves_single_newlines(self):
        text = "a\nb\nc"
        assert _step3_collapse_whitespace(text) == "a\nb\nc"


class TestStep4NormalizeTables:
    def test_normalize_pipe_table_spacing(self):
        text = "|  a  |  b  |"
        assert _step4_normalize_tables(text) == "| a | b |"

    def test_multiline_table(self):
        text = "|a|b|\n|---|---|\n|1|2|"
        result = _step4_normalize_tables(text)
        lines = result.splitlines()
        assert lines[0] == "| a | b |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 1 | 2 |"

    def test_non_table_lines_preserved(self):
        text = "Hello world\n|a|b|\nGoodbye"
        result = _step4_normalize_tables(text)
        lines = result.splitlines()
        assert lines[0] == "Hello world"
        assert lines[1] == "| a | b |"
        assert lines[2] == "Goodbye"

    def test_no_table(self):
        text = "No tables here."
        assert _step4_normalize_tables(text) == text


class TestStep5NormalizeFormulas:
    def test_alpha(self):
        assert _step5_normalize_formulas(r"\alpha") == "\u03b1"

    def test_pi(self):
        assert _step5_normalize_formulas(r"\pi") == "\u03c0"

    def test_leq_geq(self):
        text = r"x \leq 5 and y \geq 10"
        result = _step5_normalize_formulas(text)
        assert "\u2264" in result
        assert "\u2265" in result

    def test_strip_dollar_signs(self):
        assert _step5_normalize_formulas("$x + y$") == "x + y"

    def test_preserve_multiline_dollar(self):
        """Dollar signs spanning multiple lines should not be stripped."""
        text = "$long\nformula$"
        assert _step5_normalize_formulas(text) == text

    def test_preserve_long_formulas(self):
        """Formulas longer than 50 chars inside $ should not be stripped."""
        text = "$" + "x" * 60 + "$"
        assert _step5_normalize_formulas(text) == text

    def test_multiple_replacements(self):
        text = r"\alpha + \beta = \gamma"
        result = _step5_normalize_formulas(text)
        assert "\u03b1" in result
        assert "\u03b2" in result
        assert "\u03b3" in result


class TestStep6StripHeadersFooters:
    def test_strip_page_number(self):
        text = "Content\n42\nMore content"
        result = _step6_strip_headers_footers(text)
        assert "42" not in result
        assert "Content" in result

    def test_strip_page_of_pattern(self):
        text = "Content\nPage 3 of 10\nMore"
        result = _step6_strip_headers_footers(text)
        assert "Page 3 of 10" not in result

    def test_strip_dashed_page_number(self):
        text = "Content\n— 5 —\nMore"
        result = _step6_strip_headers_footers(text)
        assert "— 5 —" not in result

    def test_preserve_numbers_in_text(self):
        """Numbers embedded in text should NOT be stripped."""
        text = "There are 42 items in the list."
        assert _step6_strip_headers_footers(text) == text

    def test_preserve_large_numbers(self):
        """5+ digit numbers are not page numbers."""
        text = "12345"
        assert "12345" in _step6_strip_headers_footers(text)


class TestNormalizePipeline:
    def test_full_pipeline(self):
        text = "<s><|assistant|>```markdown\n# Title\n\n\n\n|  a |  b  |\n\n$\\alpha$\n\n42\n```</s>"
        result = normalize(text)
        assert "<s>" not in result
        assert "<|assistant|>" not in result
        assert "```" not in result
        assert "\u03b1" in result  # alpha converted
        assert "| a | b |" in result  # table normalized

    def test_empty_string(self):
        assert normalize("") == ""

    def test_deterministic(self):
        text = "<|im_start|>Test $\\pi$ content\n\n\n\nPage 1 of 5<|im_end|>"
        r1 = normalize(text)
        r2 = normalize(text)
        assert r1 == r2

    def test_plain_text_passthrough(self):
        text = "Simple plain text without any special tokens."
        result = normalize(text)
        assert result == text
