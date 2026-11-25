"""Extract sentence candidates from original Steam review structure."""
from __future__ import annotations

import re
from itertools import islice


LONG_REVIEW_CHARACTERS = 250
LIST_MARKER = re.compile(
    r"^(?:[-*+•▪‣]\s+|\d{1,3}[.)]\s+|\d{1,3}、\s*"
    r"|[（(](?:\d{1,3}|[一二三四五六七八九十]{1,3})[）)]\s*)"
)
CHECKBOX = re.compile(r"\[(?:\s+|\s*[xX✓✔]\s*)\]|[☐☑☒□✓✔🗹]")
UNCHECKED = {"☐", "□"}
HEADING = re.compile(
    r"^\s*(?:-{2,}\s*\{\s*(?P<steam>[^{}]{1,65})\s*\}\s*-{2,}"
    r"|\[(?P<tag>h[1-3]|b)\](?P<bbcode>.+?)\[/(?P=tag)\]"
    r"|\#{1,6}\s+(?P<markdown>.+?))\s*$",
    re.IGNORECASE,
)
PLAIN_HEADING = re.compile(r"^([^:：.!?。！？]{1,55})[:：]\s*$")
OVERALL_RATING = re.compile(r"(?:(?:my |overall )?rating|评分|总评|总分|\?\s*/\s*10)", re.IGNORECASE)
SENTENCE_END = re.compile(r"[。！？!?；;]+|\.(?=\s)")
ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|mr|mrs|dr|vs|etc|no)\.$|\b[A-Z]\.$", re.IGNORECASE)
CONTRAST = re.compile(
    r"(?:[,，]\s*)?(?P<connector>\b(?:but|however|whereas)\b|但是|不过|然而|而且)"
    r"|[,，]\s*(?P<comma_connector>但|却|就是|另外|同时|\band\b)", re.IGNORECASE
)
DEPENDENT = re.compile(r"^[,，\s]*(?:not\b|never\b|also\b|only if\b|unless\b|而已|罢了)", re.IGNORECASE)
HAN = re.compile(r"[\u3400-\u9fff]")
WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def compact(text: str) -> str:
    return " ".join(text.split())


def without_list_marker(text: str) -> str:
    return LIST_MARKER.sub("", text.strip(), count=1)


def has_checkbox_items(text: str) -> bool:
    return any(CHECKBOX.match(without_list_marker(line)) for line in text.splitlines())


def split_contrast(sentence: str):
    start = 0
    for match in CONTRAST.finditer(sentence):
        left = sentence[start:match.start()].strip()
        right = sentence[match.end():].strip()
        has_left = len(HAN.findall(left)) >= 2 or len(WORDS.findall(left)) >= 2
        has_right = len(HAN.findall(right)) >= 4 or len(WORDS.findall(right)) >= 3
        if not (has_left and has_right) or DEPENDENT.match(right) or re.search(r"\bnot (?:just|only)\b", left, re.I):
            continue
        yield left
        start = match.start("connector") if match["connector"] else match.start("comma_connector")
    if tail := sentence[start:].strip():
        yield tail


def split_prose(text: str):
    start = 0
    for end in SENTENCE_END.finditer(text):
        if end[0] == "." and ABBREVIATION.search(text[start:end.end()]):
            continue
        sentence = text[start:end.start()].strip()
        if sentence:
            yield from split_contrast(sentence)
        start = end.end()
    if tail := text[start:].strip():
        yield from split_contrast(tail)


def extract_units(text: str) -> list[str]:
    lines = text.splitlines()
    # Each block keeps its text chunks in source order. A heading belongs to
    # checked options only. Unchecked options may consume indented continuation.
    blocks: list[tuple[str, list[str]]] = []
    heading = ""
    paragraph: list[str] | None = None
    continuation: list[str] | None = None
    for index, raw in enumerate(lines):
        if not raw.strip():
            paragraph = continuation = None
            continue
        line = without_list_marker(raw)
        decorated = HEADING.fullmatch(line)
        plain = PLAIN_HEADING.fullmatch(line)
        if decorated or plain:
            next_line = next((following for following in islice(lines, index + 1, None) if following.strip()), "")
            title = (decorated["steam"] or decorated["bbcode"] or decorated["markdown"]) if decorated else plain[1]
            if CHECKBOX.match(without_list_marker(next_line)):
                heading = compact(title)
                paragraph = continuation = None
                continue
            if decorated:
                line = compact(title) + ":"
                paragraph = None
        if CHECKBOX.match(line):
            markers = list(CHECKBOX.finditer(line))
            for pos, marker in enumerate(markers):
                end = markers[pos + 1].start() if pos + 1 < len(markers) else len(line)
                continuation = [line[marker.end():end].strip()]
                checked = marker[0] not in UNCHECKED and not (marker[0].startswith("[") and not marker[0][1:-1].strip())
                if checked and not OVERALL_RATING.fullmatch(heading):
                    blocks.append((heading or " ", continuation))
            paragraph = None
            continue
        if continuation is not None and (raw.startswith("\t") or raw.startswith("  ")):
            continuation.append(line)
            continue
        heading = ""
        continuation = None
        if paragraph is None or LIST_MARKER.match(raw.lstrip()):
            paragraph = []
            blocks.append(("", paragraph))
        paragraph.append(line)
        if LIST_MARKER.match(raw.lstrip()):
            paragraph = None

    units = []
    for header, chunks in blocks:
        content = compact(" ".join(chunks))
        if not content:
            continue
        if header:
            units.append(f"{header}: {content}" if header.strip() else content)
        else:
            units.extend(split_prose(content))
    return units
