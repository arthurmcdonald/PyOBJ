"""
Term printing, following OBJ3's mixfix rules (term_print.lsp).
"""
from __future__ import annotations

from .core import Term, Var, Literal, App

SYNTAX = set("()[]{},_")


def _need_space(left: str, right: str) -> bool:
    """Whether two adjacent tokens need a blank between them (none next to ( ) [ ] { } , _)."""
    return left[-1] not in SYNTAX and right[0] not in SYNTAX


def _need_space_bars(left: str, right: str) -> bool:
    """Like _need_space, but "_" stands for an argument that has not been printed yet."""
    if left == "_":
        return right == "_" or right[0] not in SYNTAX
    if right == "_":
        return left[-1] not in SYNTAX
    return _need_space(left, right)


def term_to_str(term: Term, module=None, show_retracts: bool = False,
                show_var_sorts: bool = False) -> str:
    pieces: list[str] = []
    _write_term(term, 127, module, show_retracts, show_var_sorts, pieces)
    return "".join(pieces)


def _literal_text(literal: Literal) -> str:
    return literal.sort.builtin.show(literal.value) if literal.sort.builtin else str(literal.value)


def _write_term(term, precedence_limit, module, show_retracts, show_var_sorts, pieces):
    """
    Append the text of ``term`` to ``pieces``.

    ``precedence_limit`` is the largest operator precedence allowed without
    parentheses at this position (127 = anything), taken from the enclosing
    operator's gather pattern.
    """
    term_type = type(term)
    if term_type is Var:
        pieces.append(term.name)
        if show_var_sorts:
            pieces.append(":" + term.sort.name)
        return
    if term_type is Literal:
        pieces.append(_literal_text(term))
        return
    op = term.op
    if op.kind == "retract":
        if show_retracts:
            pieces.append("r:%s>%s(" % (op.retract[0].name, op.retract[1].name))
            _write_term(term.args[0], 127, module, show_retracts, show_var_sorts, pieces)
            pieces.append(")")
        else:
            _write_term(term.args[0], precedence_limit, module, show_retracts, show_var_sorts, pieces)
        return
    if op.standard:
        # Ordinary prefix form: f(a,b)
        prev = None
        for token in op.name:
            if prev is not None and _need_space(prev, token):
                pieces.append(" ")
            pieces.append(token)
            prev = token
        if term.args:
            pieces.append("(")
            for k, arg in enumerate(term.args):
                if k:
                    pieces.append(",")
                _write_term(arg, 127, module, show_retracts, show_var_sorts, pieces)
            pieces.append(")")
        return
    if not term.args and len(op.name) == 1 and op.name[0] != "_":
        pieces.append(op.name[0])
        return
    # Mixfix: parenthesize when the surrounding position only allows tighter
    # (lower-precedence) operators than this one.
    needs_parens = precedence_limit <= op.prec
    if needs_parens:
        pieces.append("(")
    # An associative operator nested in itself prints without parentheses:
    # a + b + c, not a + (b + c).
    theory = module.flat().fam_theory.get(op.fam) if module is not None else None
    assoc = op.assoc or (theory is not None and theory.assoc)
    prev = None
    args = list(term.args)
    for item in op.form:
        if item[0] == "a":
            if prev is not None and _need_space_bars(prev, "_"):
                pieces.append(" ")
            prev = "_"
            arg = args.pop(0) if args else None
            if arg is None:
                break
            arg_limit = item[1]
            if assoc and type(arg) is App and arg.op.fam == op.fam:
                arg_limit = 127
            _write_term(arg, arg_limit, module, show_retracts, show_var_sorts, pieces)
        else:
            token = item[1]
            if prev is not None and _need_space_bars(prev, token):
                pieces.append(" ")
            prev = token
            pieces.append(token)
    if needs_parens:
        pieces.append(")")
