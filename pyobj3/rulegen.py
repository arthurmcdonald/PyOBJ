"""
Rule generation: which equations become rewrite rules.

The Lisp ``rule_gen`` package (ext_rule_gen.lsp) "compiled" a module's
equations: it attached each equation to its top operator, copied it onto the
lower-sorted versions of that operator (specialization), and generated the
extra rules needed for associative / AC / identity operators (``X + lhs = X +
rhs``, ``lhs + X = rhs + X``, ``X + lhs + Y = X + rhs + Y``).

This port does not materialise those copies.  The engine keys rules by
operator family (so every rank of an overloaded operator sees the rule;
the sort test happens when matching) and matches A / AC / identity patterns
with extension directly (``Engine.match_top``).  What remains of rule
generation is deciding which equations are usable rules at all.
"""
from __future__ import annotations

from .core import Var, term_vars


def is_bad_rule(equation) -> bool:
    """
    True for equations that can never be used as rewrite rules: the
    left-hand side is a bare variable, or (for ordinary equations) the right
    side or condition mentions a variable that does not occur on the left.
    """
    if type(equation.lhs) is Var:
        return True
    if equation.py is not None: # built-in (bq) equations compute their result in Python
        return False
    lhs_vars = set(term_vars(equation.lhs))
    if equation.rhs is not None and not set(term_vars(equation.rhs)) <= lhs_vars:
        return True
    if equation.cond is not None and not set(term_vars(equation.cond)) <= lhs_vars:
        return True
    return False


def bad_rules(module):
    return [equation for equation in module.flat().eqs if is_bad_rule(equation)]


WARNING = ("Warning: Module contains rule with LHS that is a variable or\n"
           "a RHS or condition with a variable not occurring in the LHS.\n"
           "These rules will be ignored during reduction.")
