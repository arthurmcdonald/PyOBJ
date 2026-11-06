"""
OBJ3 lexical analysis.

OBJ3 tokens are separated by white space, except that ``( ) , [ ] { } _``
are always tokens on their own (this is what makes ``_+_`` the token list
``_ + _``).  ``---`` and ``***`` start a comment that runs to end of line;
``--->`` and ``***>`` introduce a comment that is echoed.
"""
from __future__ import annotations

from dataclasses import dataclass

SPECIAL = set("()[]{},_")
WS = set(" \t\r\n\f")


@dataclass(slots=True)
class Token:
    text: str
    start: int
    end: int

    def __repr__(self):
        return self.text


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(source)
    while i < n:
        char = source[i]
        if char in WS:
            i += 1
            continue
        if char in SPECIAL:
            tokens.append(Token(char, i, i + 1))
            i += 1
            continue
        word_end = i
        while word_end < n and source[word_end] not in WS and source[word_end] not in SPECIAL:
            word_end += 1
        text = source[i:word_end]
        if text in ("---", "***"):  # comment to end of line
            line_break = source.find("\n", word_end)
            i = n if line_break < 0 else line_break + 1
            continue
        if text in ("--->", "***>"):  # echoed comment
            line_break = source.find("\n", word_end)
            line_break = n if line_break < 0 else line_break
            tokens.append(Token(text, i, line_break))
            i = line_break
            continue
        tokens.append(Token(text, i, word_end))
        i = word_end
    return tokens


def line_end(source: str, pos: int) -> int:
    line_break = source.find("\n", pos)
    return len(source) if line_break < 0 else line_break
