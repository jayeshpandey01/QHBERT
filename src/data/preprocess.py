"""Text cleaning shared by local dev and the Kaggle embedding-extraction notebook."""

import re

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
