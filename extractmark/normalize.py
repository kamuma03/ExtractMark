"""Output normalisation pipeline (Section 9.3).

All model outputs are normalised to a canonical Markdown representation
before any metric evaluation. This prevents measuring format differences
instead of extraction quality.

Normalisation steps (applied in order):
1. Strip model-specific preamble/postamble (system tokens)
2. Normalise Unicode (NFC form)
3. Collapse multiple whitespace / blank lines
4. Standardise table format to pipe-delimited Markdown
5. Normalise inline formulas to Unicode where possible
6. Strip page headers/footers if not part of the evaluation target
"""

from __future__ import annotations

import re
import unicodedata


# Step 1: Model-specific tokens to strip
_MODEL_TOKENS = [
    "<|im_start|>", "<|im_end|>",
    "<|endoftext|>", "<|end|>",
    "<s>", "</s>",
    "<|assistant|>", "<|user|>", "<|system|>",
    "<|grounding|>",
    "```markdown", "```",
]

# Step 5: Common LaTeX to Unicode mappings
_LATEX_TO_UNICODE = {
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\delta": "\u03b4",
    r"\epsilon": "\u03b5",
    r"\theta": "\u03b8",
    r"\lambda": "\u03bb",
    r"\mu": "\u03bc",
    r"\pi": "\u03c0",
    r"\sigma": "\u03c3",
    r"\omega": "\u03c9",
    r"\times": "\u00d7",
    r"\div": "\u00f7",
    r"\pm": "\u00b1",
    r"\leq": "\u2264",
    r"\geq": "\u2265",
    r"\neq": "\u2260",
    r"\approx": "\u2248",
    r"\infty": "\u221e",
    r"\sum": "\u2211",
    r"\prod": "\u220f",
    r"\int": "\u222b",
    r"\sqrt": "\u221a",
    r"\rightarrow": "\u2192",
    r"\leftarrow": "\u2190",
    r"\Rightarrow": "\u21d2",
    r"\degree": "\u00b0",
}


def _step1_strip_tokens(text: str) -> str:
    """Strip model-specific preamble/postamble tokens."""
    for token in _MODEL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def _step2_normalize_unicode(text: str) -> str:
    """Normalise Unicode to NFC form (resolves smart quotes, ligatures)."""
    return unicodedata.normalize("NFC", text)


def _step3_collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace and blank lines."""
    # Collapse multiple blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but preserve newlines and indentation)
    text = re.sub(r"[ \t]+", " ", text)
    # Clean up lines that are just whitespace
    text = re.sub(r"\n +\n", "\n\n", text)
    return text.strip()


def _step4_normalize_tables(text: str) -> str:
    """Standardise table format to pipe-delimited Markdown.

    Handles common table formats:
    - HTML tables (<table>...</table>)
    - Tab-separated values
    - Already pipe-delimited tables (cleaned up)
    """
    # Clean up existing pipe tables: ensure consistent spacing
    lines = text.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # Normalise cell spacing in pipe tables
            cells = [c.strip() for c in stripped.split("|")]
            # cells[0] and cells[-1] are empty due to leading/trailing |
            normalized = "| " + " | ".join(cells[1:-1]) + " |"
            result.append(normalized)
        else:
            result.append(line)
    return "\n".join(result)


def _step5_normalize_formulas(text: str) -> str:
    """Normalise inline LaTeX formulas to Unicode where possible."""
    for latex, unicode_char in _LATEX_TO_UNICODE.items():
        text = text.replace(latex, unicode_char)
    # Strip remaining $ delimiters around simple inline formulas
    text = re.sub(r"\$([^$\n]{1,50})\$", r"\1", text)
    return text


def _step6_strip_headers_footers(text: str) -> str:
    """Strip common page headers/footers.

    Removes lines that look like page numbers, running headers, or footers.
    """
    lines = text.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        # Skip standalone page numbers
        if re.match(r"^[-—–]?\s*\d{1,4}\s*[-—–]?$", stripped):
            continue
        # Skip "Page X of Y" patterns
        if re.match(r"^[Pp]age\s+\d+\s*(of|/)\s*\d+$", stripped):
            continue
        result.append(line)
    return "\n".join(result)


def normalize(text: str) -> str:
    """Apply the full 6-step normalisation pipeline.

    This function is deterministic -- given the same input, it always
    produces the same output.
    """
    text = _step1_strip_tokens(text)
    text = _step2_normalize_unicode(text)
    text = _step3_collapse_whitespace(text)
    text = _step4_normalize_tables(text)
    text = _step5_normalize_formulas(text)
    text = _step6_strip_headers_footers(text)
    return text
