import re

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

_CODE_BLOCK = re.compile(r"(```.*?```)", re.DOTALL)
_MATH_BLOCK = re.compile(r"(\$\$.*?\$\$)", re.DOTALL)


def segment(text: str) -> list[str]:
    if not text.strip():
        return []
    if _CODE_BLOCK.search(text) or _MATH_BLOCK.search(text):
        return _segment_with_protected_blocks(text)
    return [s for s in sent_tokenize(text) if s.strip()]


def _segment_with_protected_blocks(text: str) -> list[str]:
    # re.split with a capturing group keeps delimiters in the result list,
    # alternating: [plain, block, plain, block, ...]
    parts = _CODE_BLOCK.split(text)
    segments: list[str] = []
    for part in parts:
        if _CODE_BLOCK.fullmatch(part):
            segments.append(part)
            continue
        # Further split on display math within plain-text parts
        sub_parts = _MATH_BLOCK.split(part)
        for sub in sub_parts:
            if _MATH_BLOCK.fullmatch(sub):
                segments.append(sub)
            elif sub.strip():
                segments.extend(s for s in sent_tokenize(sub) if s.strip())
    return segments
