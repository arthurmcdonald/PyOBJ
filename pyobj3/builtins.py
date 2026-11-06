"""
Built-in sorts (numbers, identifiers).

In the Lisp original these were declared with ``bsort Name (is_token create
print is_value)``, with the four hook functions written in Lisp.  Here the
hooks are native Python and ``bsort Name .`` simply looks the sort up in the
registry below; extra built-in sorts can be added with ``register_bsort``.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Callable

_INT = re.compile(r"^-?[0-9]+$")
_FLOAT = re.compile(r"^-?(?:[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+|\.[0-9]+)$")


class BuiltinSort:
    def __init__(self, name: str, kind: str, token_ok: Callable[[str], bool],
                 parse: Callable[[str], object], value_ok: Callable[[object], bool],
                 show: Callable[[object], str] = str):
        self.name = name
        self.kind = kind # 'int' | 'rat' | 'float' | 'id'
        self.token_ok = token_ok
        self.parse = parse
        self.value_ok = value_ok
        self.show = show

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)

def _is_rat(value):
    return _is_int(value) or isinstance(value, Fraction)

def _show_rat(value):
    return str(value.numerator) + "/" + str(value.denominator) if isinstance(value, Fraction) else str(value)

def _show_float(value):
    return repr(float(value))

REGISTRY: dict[str, BuiltinSort] = {}

def register_bsort(builtin_sort: BuiltinSort):
    REGISTRY[builtin_sort.name] = builtin_sort


def _int_tok(extra_check=lambda v: True, allow_negative=True):
    return lambda t: bool(_INT.match(t)) and (allow_negative or not t.startswith("-")) and extra_check(int(t))


register_bsort(BuiltinSort("Zero", "int", _int_tok(lambda v: v == 0, False), int,
                           lambda v: _is_int(v) and v == 0))
register_bsort(BuiltinSort("NzNat", "int", _int_tok(lambda v: v > 0, False), int,
                           lambda v: _is_int(v) and v > 0))
register_bsort(BuiltinSort("Nat", "int", _int_tok(lambda v: v >= 0, False), int,
                           lambda v: _is_int(v) and v >= 0))
register_bsort(BuiltinSort("NzInt", "int", _int_tok(lambda v: v != 0), int,
                           lambda v: _is_int(v) and v != 0))
register_bsort(BuiltinSort("Int", "int", _int_tok(), int, _is_int))
register_bsort(BuiltinSort("NzRat", "rat", lambda t: False, Fraction,
                           lambda v: _is_rat(v) and v != 0, _show_rat))
register_bsort(BuiltinSort("Rat", "rat", lambda t: False, Fraction, _is_rat, _show_rat))
register_bsort(BuiltinSort("Float", "float", lambda t: bool(_FLOAT.match(t)), float,
                           lambda v: isinstance(v, float), _show_float))
register_bsort(BuiltinSort("Id", "id", lambda t: len(t) > 1 and t[0] == "'", str,
                           lambda v: isinstance(v, str), str))


def value_kind(value) -> str:
    if isinstance(value, bool):
        raise TypeError("bool is not a literal value")
    if isinstance(value, int):
        return "int"
    if isinstance(value, Fraction):
        return "rat"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "id"
    raise TypeError(type(value))


def _smallest_sorts(flat, candidates):
    """The candidates that have no strictly smaller candidate below them."""
    return [sort for sort in candidates
            if not any(flat.order.lt(other, sort) for other in candidates)]


def make_literal_from_token(module, token: str):
    """Return a Lit for ``token`` if some built-in sort of ``module`` accepts it."""
    from .core import Literal
    flat = module.flat()
    candidates = [sort for sort in flat.bsorts if sort.builtin.token_ok(token)]
    if not candidates:
        return None
    # "3" is accepted by NzNat, Nat, NzInt, Int, ...; give it the smallest sort.
    sort = _smallest_sorts(flat, candidates)[0]
    return Literal(sort.builtin.parse(token), sort, sort.builtin.kind)


def make_literal_from_value(module, value):
    """Return the Lit for a Python value, or None if no sort of ``module`` fits."""
    from .core import Literal
    if isinstance(value, Fraction) and value.denominator == 1:
        value = int(value)
    try:
        kind = value_kind(value)
    except TypeError:
        return None
    flat = module.flat()
    # An int may also be a rational; every other kind needs a sort of its own kind.
    kinds = ("int", "rat") if kind == "int" else (kind,)
    candidates = [sort for sort in flat.bsorts
                  if sort.builtin.value_ok(value) and sort.builtin.kind in kinds]
    if not candidates:
        return None
    smallest = _smallest_sorts(flat, candidates)
    # Prefer a sort of the value's own kind, then the oldest declaration.
    smallest.sort(key=lambda sort: (sort.builtin.kind != kind, sort.uid))
    return Literal(value, smallest[0], kind)
