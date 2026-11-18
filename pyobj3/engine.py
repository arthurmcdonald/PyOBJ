"""
Matching modulo equational theories and the rewriting (reduction) engine.

Matching supports, per operator family: free, commutative (C), associative
(A), associative-commutative (AC), idempotent (I) and identity (Z)
operators, and *extension* for A/AC operators at the top of a left-hand
side (so ``eq s(N) + s(M) = ...`` applies inside ``s(a) + s(b) + c``).

Reduction follows OBJ3's evaluation strategies: each operator has an
E-strategy -- a list of argument positions to evaluate and ``0`` meaning
"try the equations at the top".  The default is "innermost": all arguments
left to right, then 0.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Iterator, Optional

from .rulegen import is_bad_rule
from .core import (Term, Var, Literal, App, Operator, Equation, subst)
from .builtins import make_literal_from_value
from .errors import RuleFailure


def _skey(term):
    """Sort key giving terms a deterministic, hashable-by-structure order
    (used to put the arguments of commutative operators in canonical order)."""
    term_type = type(term)
    if term_type is Var:
        return (0, term.name, term.sort.uid)
    if term_type is Literal:
        return (1, term.kind, repr(term.value))
    return (2, term.op.fam, tuple(_skey(arg) for arg in term.args))


class _Fail(Exception):
    """Raised inside the matcher to abandon a branch of the search."""


class Engine:
    """Rewrite engine for one module: matching, normalization and memoization."""

    def __init__(self, module):
        self.module = module
        flat = module.flat()
        self.flat = flat
        self.order = flat.order
        self.theories = flat.fam_theory
        self.rules: dict[int, list[Equation]] = {}
        # Index rules by the operator family at the top of the left-hand side;
        # equations whose lhs is a variable (or otherwise unusable) are skipped.
        for equation in flat.eqs:
            if type(equation.lhs) is App and not is_bad_rule(equation):
                self.rules.setdefault(equation.lhs.op.fam, []).append(equation)
        self.normal_forms: set = set()
        self.memo: dict = {}
        self.canon_cache: dict = {}
        self.rewrite_count = 0
        true_op = module.find_const("true", "Bool")
        false_op = module.find_const("false", "Bool")
        self.true = App(true_op, ()) if true_op else None
        self.false = App(false_op, ()) if false_op else None
        self.tracer = None          # trace.Tracer when tracing
        self.context: list = []     # trace whole: builders of the enclosing terms
        self.context_kind = "term"

    # Term building
    def make_term(self, op: Operator, args) -> App:
        args = tuple(args)
        if op.kind is not None or not args:
            return App(op, args)
        lowest, _ = self.module.lowest_rank(op, tuple(self.module.sort_of(arg) for arg in args))
        return App(lowest, args)

    def theory(self, op: Operator):
        return self.theories.get(op.fam) or _PLAIN

    def sort_ok(self, term: Term, sort) -> bool:
        return self.order.leq(self.module.sort_of(term), sort)

    def identity_of(self, op: Operator) -> Optional[Term]:
        return self.theory(op).identity

    # Canonical forms
    def items(self, term: Term, op: Operator) -> list[Term]:
        """Flatten nested applications of ``op``'s family into a list of operands."""
        flattened: list[Term] = []
        family = op.fam
        pending = [term]
        while pending:
            node = pending.pop()
            if type(node) is App and node.op.fam == family and len(node.args) == 2:
                pending.append(node.args[1])
                pending.append(node.args[0])
            else:
                flattened.append(node)
        return flattened

    def rebuild(self, op: Operator, items: list[Term]) -> Term:
        """Right-nested application of a binary op to ``items``."""
        if not items:
            identity = self.identity_of(op)
            if identity is None:
                raise _Fail()
            return identity
        result = items[-1]
        for item in reversed(items[:-1]):
            result = self.make_term(op, (item, result))
        return result

    def canon(self, term: Term) -> Term:
        """Canonical representative of ``term`` modulo the declared theories:
        A operators are flattened and right-nested, C operands sorted, I
        duplicates dropped and identities removed.  Two terms equal modulo
        these axioms have the same canonical form."""
        term_type = type(term)
        if term_type is not App:
            return term
        cached = self.canon_cache.get(term)
        if cached is not None:
            return cached
        args = tuple(self.canon(arg) for arg in term.args)
        theory = self.theory(term.op)
        op = term.op
        if theory.plain or not args:
            result = term if all(arg is other_arg for arg, other_arg in zip(args, term.args)) else App(op, args)
        else:
            if theory.assoc:
                items: list[Term] = []
                for arg in args:
                    if type(arg) is App and arg.op.fam == op.fam:
                        items.extend(self.items(arg, op))
                    else:
                        items.append(arg)
            else:
                items = list(args)
            if theory.identity is not None:
                identity_canon = self.canon(theory.identity)
                items = [item for item in items if item != identity_canon]
            if theory.comm:
                items.sort(key=_skey)
            if theory.idem:
                # Sorted (C) or not (A only), equal items are adjacent, so
                # dropping adjacent repeats removes all duplicates.
                deduped: list[Term] = []
                for item in items:
                    if not deduped or deduped[-1] != item:
                        deduped.append(item)
                items = deduped
            if not items:
                result = self.canon(theory.identity)
            elif len(items) == 1:
                result = items[0]
            elif theory.assoc:
                nested = items[-1]
                for item in reversed(items[:-1]):
                    nested = App(op, (item, nested))
                result = nested
            else:
                result = App(op, tuple(items))
        self.canon_cache[term] = result
        return result

    def eq_mod(self, term_a: Term, term_b: Term) -> bool:
        return term_a == term_b or self.canon(term_a) == self.canon(term_b)

    # Matching
    def match(self, pattern: Term, subject: Term, substitution: dict) -> Iterator[dict]:
        """Generate every extension of ``substitution`` that makes ``pattern``
        equal to ``subject`` modulo the theories.  Matching is lazy: callers
        take the first solution, or backtrack into the rest (for conditions)."""
        pattern_type = type(pattern)
        if pattern_type is Var:
            bound = substitution.get(pattern)
            if bound is not None:
                if self.eq_mod(bound, subject):
                    yield substitution
                return
            if self.sort_ok(subject, pattern.sort):
                extended = dict(substitution)
                extended[pattern] = subject
                yield extended
            return
        if pattern_type is Literal:
            if type(subject) is Literal and subject == pattern:
                yield substitution
            return
        # Application pattern: the strategy depends on the operator's theory.
        op = pattern.op
        if op.kind == "retract":
            if self.sort_ok(subject, op.coarity):
                yield from self.match(pattern.args[0], subject, substitution)
            return
        theory = self.theory(op)
        headed = type(subject) is App and subject.op.fam == op.fam and len(subject.args) == len(pattern.args)
        if not headed:
            # modulo identity a subject x is also  x op e  (and  e op x)
            if not theory.zmatch or len(pattern.args) != 2:
                return
            if type(subject) is App and subject.op.kind == "retract":
                return
            if theory.assoc:
                subject_args = self.subject_items(subject, op, theory)
                yield from (found for found, _ in self._match_assoc(pattern, subject_args, substitution, False))
            else:
                identity = theory.identity
                for pair in ((subject, identity), (identity, subject)):
                    yield from self._match_seq(pattern.args, pair, substitution)
            return
        if theory.assoc:
            subject_args = self.subject_items(subject, op, theory)
            yield from (found for found, _ in self._match_assoc(pattern, subject_args, substitution, False))
        elif theory.comm and len(pattern.args) == 2:
            for pattern_args in ((pattern.args[0], pattern.args[1]), (pattern.args[1], pattern.args[0])):
                for found in self._match_seq(pattern_args, subject.args, substitution):
                    yield found
                if pattern.args[0] == pattern.args[1]:
                    break
        else:
            yield from self._match_seq(pattern.args, subject.args, substitution)

    def subject_items(self, subject: Term, op: Operator, theory) -> list[Term]:
        """Flattened argument list of an A/AC subject; a subject that is not
        headed by ``op`` is a one-element list (modulo identity)."""
        if type(subject) is App and subject.op.fam == op.fam:
            items = self.items(subject, op)
        else:
            items = [subject]
        if theory.zmatch:
            identity_canon = self.canon(theory.identity)
            items = [item for item in items if self.canon(item) != identity_canon]
        if theory.idem:
            items = self._dedupe(items, theory.comm)
        return items

    def _match_seq(self, patterns, subjects, substitution, k=0):
        if k == len(patterns):
            yield substitution
            return
        for found in self.match(patterns[k], subjects[k], substitution):
            yield from self._match_seq(patterns, subjects, found, k + 1)

    # Associative / AC matching on flattened argument lists
    def _match_assoc(self, pattern: App, subject_args: list, substitution, extension: bool):
        """Yield (substitution, leftover) pairs.  ``leftover`` is None, or, when
        ``extension`` is set, the subject items the pattern did not consume
        (a list for AC, a (before, after) pair for A).  ``subject_args`` is the
        flattened operand list of the subject."""
        op = pattern.op
        theory = self.theory(op)
        pattern_args = self.items(pattern, op)
        if theory.comm:
            yield from self._match_ac(op, theory, pattern_args, subject_args, substitution, extension)
        else:
            yield from self._match_a(op, theory, pattern_args, subject_args, substitution, extension)

    @staticmethod
    def _dedupe(items, commutative):
        kept = []
        for item in items:
            if commutative:
                if item not in kept:
                    kept.append(item)
            elif not kept or kept[-1] != item:
                kept.append(item)
        return kept

    def _match_a(self, op, theory, pattern_args, subject_args, substitution, extension):
        """Associative (non-commutative) matching: a variable takes a
        contiguous run of subject items; non-variables match one item each."""
        pattern_len, subject_len = len(pattern_args), len(subject_args)
        match_identity = theory.zmatch
        # min_needed[k]: fewest subject items the patterns from k onward must
        # consume; lets a variable stop before starving the rest.
        min_needed = [0] * (pattern_len + 1)
        for k in range(pattern_len - 1, -1, -1):
            min_needed[k] = min_needed[k + 1] + (0 if (match_identity and type(pattern_args[k]) is Var) else 1)
        start_positions = range(subject_len + 1) if extension else [0]

        def rec(pattern_index, subject_index, current):
            if pattern_index == pattern_len:
                if subject_index == subject_len or extension:
                    yield current, subject_index
                return
            pattern_item = pattern_args[pattern_index]
            if type(pattern_item) is Var:
                bound = current.get(pattern_item)
                if bound is not None:
                    bound_items = self.items(bound, op) if (type(bound) is App and bound.op.fam == op.fam) else [bound]
                    if match_identity and self.eq_mod(bound, theory.identity):
                        bound_items = []
                    if subject_index + len(bound_items) <= subject_len and all(self.eq_mod(item_x, item_y) for item_x, item_y in zip(bound_items, subject_args[subject_index:subject_index + len(bound_items)])):
                        yield from rec(pattern_index + 1, subject_index + len(bound_items), current)
                    return
                min_size = 0 if match_identity else 1
                for k in range(min_size, subject_len - subject_index - min_needed[pattern_index + 1] + 1):
                    if k == 0:
                        value = theory.identity
                    elif k == 1:
                        value = subject_args[subject_index]
                    else:
                        value = self.rebuild(op, subject_args[subject_index:subject_index + k])
                    if self.sort_ok(value, pattern_item.sort):
                        extended = dict(current)
                        extended[pattern_item] = value
                        yield from rec(pattern_index + 1, subject_index + k, extended)
            else:
                if subject_index < subject_len:
                    for extended in self.match(pattern_item, subject_args[subject_index], current):
                        yield from rec(pattern_index + 1, subject_index + 1, extended)

        for start in start_positions:
            for current, end in rec(0, start, substitution):
                if extension:
                    if start + (subject_len - end) < subject_len and (start or end < subject_len):
                        # extension: (items before, items after) the match
                        yield current, (subject_args[:start], subject_args[end:])
                else:
                    yield current, None
            if not extension:
                break

    def plug(self, op, leftover, replacement):
        """Put ``replacement`` back among the unconsumed items ``leftover`` of a
        match with extension: a (before, after) pair for A, a list for AC."""
        if type(leftover) is tuple:
            return self.rebuild(op, list(leftover[0]) + [replacement] + list(leftover[1]))
        return self.rebuild(op, leftover + [replacement])

    def _match_ac(self, op, theory, pattern_args, subject_args, substitution, extension):
        """Associative-commutative matching over a multiset of subject items:
        non-variable patterns pick an item each, then variables take
        sub-multisets of whatever is left."""
        match_identity = theory.zmatch
        nonvariables = [pattern_item for pattern_item in pattern_args if type(pattern_item) is not Var]
        variables = [pattern_item for pattern_item in pattern_args if type(pattern_item) is Var]
        # Non-variables first: they constrain the search the most, so the
        # variables are left to share out the remaining items.
        ordered_patterns = nonvariables + variables

        def rec(pattern_index, remaining, current):
            if pattern_index == len(ordered_patterns):
                if not remaining:
                    yield current, None
                elif extension and len(remaining) < len(subject_args):
                    yield current, list(remaining)  # extension: match consumed at least one item
                return
            pattern_item = ordered_patterns[pattern_index]
            if type(pattern_item) is not Var:
                tried_items = []   # skip duplicate items: same result, avoids blowup
                for k, item in enumerate(remaining):
                    if item in tried_items:
                        continue
                    tried_items.append(item)
                    for extended in self.match(pattern_item, item, current):
                        yield from rec(pattern_index + 1, remaining[:k] + remaining[k + 1:], extended)
                return
            bound = current.get(pattern_item)
            if bound is not None:
                bound_items = self.items(bound, op) if (type(bound) is App and bound.op.fam == op.fam) else [bound]
                if match_identity and self.eq_mod(bound, theory.identity):
                    bound_items = []
                still_needed = list(remaining)
                for needed in bound_items:
                    for k, item in enumerate(still_needed):
                        if self.eq_mod(item, needed):
                            del still_needed[k]
                            break
                    else:
                        return
                yield from rec(pattern_index + 1, still_needed, current)
                return
            # A fresh variable: try every sub-multiset size.  When it is the last
            # variable and there is no extension it must take everything left.
            remaining_count = len(remaining)
            later_variables = sum(1 for later_item in ordered_patterns[pattern_index + 1:] if type(later_item) is Var and later_item not in current and later_item != pattern_item)
            min_size = 0 if match_identity else 1
            sizes = [remaining_count] if (later_variables == 0 and not extension) else range(min_size, remaining_count + 1)
            if later_variables == 0 and not extension and remaining_count < min_size:
                return
            for size in sizes:
                if size == 0:
                    value = theory.identity
                    if self.sort_ok(value, pattern_item.sort):
                        extended = dict(current)
                        extended[pattern_item] = value
                        yield from rec(pattern_index + 1, list(remaining), extended)
                    continue
                seen_selections = set()   # equal-looking selections give equal results
                for combination in itertools.combinations(range(remaining_count), size):
                    chosen = [remaining[k] for k in combination]
                    key = tuple(sorted(map(_skey, chosen)))
                    if key in seen_selections:
                        continue
                    seen_selections.add(key)
                    value = chosen[0] if size == 1 else self.rebuild(op, chosen)
                    if not self.sort_ok(value, pattern_item.sort):
                        continue
                    leftover = [remaining[k] for k in range(remaining_count) if k not in combination]
                    extended = dict(current)
                    extended[pattern_item] = value
                    yield from rec(pattern_index + 1, leftover, extended)

        yield from rec(0, list(subject_args), substitution)

    def match_top(self, lhs: App, subject: App):
        """Match a rule's left-hand side at the root of ``subject`` (with
        extension for A/AC roots).  Yields (substitution, leftover or None)."""
        op = lhs.op
        if lhs.op.kind == "retract":
            for found in self.match(lhs, subject, {}):
                yield found, None
            return
        if type(subject) is not App or subject.op.fam != op.fam or len(subject.args) != len(lhs.args):
            return
        theory = self.theory(op)
        if theory.assoc and len(lhs.args) == 2:
            subject_args = self.subject_items(subject, op, theory)
            for found, leftover in self._match_assoc(lhs, subject_args, {}, False):
                yield found, None
            for found, leftover in self._match_assoc(lhs, subject_args, {}, True):
                if leftover:
                    yield found, leftover
        else:
            for found in self.match(lhs, subject, {}):
                yield found, None

    # Reduction
    def reduce(self, term: Term) -> Term:
        self.rewrite_count = 0
        return self.normalize(term)

    def strategy(self, op: Operator, arg_count: int) -> list[int]:
        if op.strat is not None:
            return op.strat
        # Any member of the operator family may carry the declared strategy.
        for member in self.flat.fam_members.get(op.fam, ()):
            if member.strat is not None:
                return member.strat
        return list(range(1, arg_count + 1)) + [0]

    def normalize(self, term: Term) -> Term:
        """Rewrite ``term`` to normal form following the E-strategy of each
        operator.  Results are cached: ``normal_forms`` for terms known to be
        irreducible, ``memo`` for operators declared ``memo``."""
        if type(term) is not App:
            return term
        if term in self.normal_forms:
            return term
        memoized = term.op.memo
        memo_keys = []      # every intermediate term that must map to the final result
        if memoized:
            cached = self.memo.get(term)
            if cached is not None:
                return cached
            memo_keys.append(term)
        while True:
            op = term.op
            theory = self.theory(op)
            changed = False
            for strategy_item in self.strategy(op, len(term.args)):
                if strategy_item == 0:
                    if op.memo and term not in memo_keys:
                        cached = self.memo.get(term)
                        if cached is not None:
                            term = cached
                            for key in memo_keys:
                                self.memo[key] = term
                            return term
                        memo_keys.append(term)
                    rewritten = self.apply_top(term)
                    if rewritten is not None:
                        term = rewritten
                        changed = True
                        break
                else:
                    # For an associative operator the flattened operands are all
                    # evaluated together, on the first strategy entry.
                    if theory.assoc:
                        if strategy_item != 1:
                            continue
                        arg_indexes = range(len(term.args))
                    else:
                        arg_indexes = (strategy_item - 1,) if strategy_item - 1 < len(term.args) else ()
                    new_args = list(term.args)
                    modified = False
                    trace_whole = self.tracer is not None and self.tracer.whole
                    for index in arg_indexes:
                        arg = new_args[index]
                        if trace_whole:
                            # For "trace whole": remember how to rebuild the
                            # enclosing term around this argument.
                            def with_arg(replacement, args=new_args, position=index, op=op):
                                return self.make_term(op, args[:position] + [replacement] + args[position + 1:])
                            self.context.append(with_arg)
                            try:
                                normalized_arg = self.normalize(arg)
                            finally:
                                self.context.pop()
                        else:
                            normalized_arg = self.normalize(arg)
                        if normalized_arg is not arg:
                            new_args[index] = normalized_arg
                            modified = True
                    if modified:
                        term = self.make_term(op, new_args)
            if not changed:
                break
            if type(term) is not App:
                break
            if term in self.normal_forms:
                break
        if type(term) is App:
            self.normal_forms.add(term)
        for key in memo_keys:
            self.memo[key] = term
        return term

    # Single step at the top
    def apply_top(self, term: App) -> Optional[Term]:
        op = term.op
        kind = op.kind
        if kind is not None:
            return self._apply_special(term)
        theory = self.theory(op)
        if not theory.plain:
            implicit = self._implicit_rules(term, theory)
            if implicit is not None:
                self.rewrite_count += 1
                return implicit
        rules = self.rules.get(op.fam)
        if not rules:
            return None
        tracer = self.tracer
        for equation in rules:
            trial_number = None
            for substitution, leftover in self.match_top(equation.lhs, term):
                if tracer is not None:
                    if equation.cond is None:
                        tracer.rule(equation, substitution)
                    else:
                        trial_number = tracer.trial(equation, substitution)
                result = self._fire(equation, substitution)
                if result is None:
                    if tracer is not None:
                        if equation.cond is None:
                            tracer.builtin_failed()
                        else:
                            tracer.failure(trial_number)
                    continue
                if leftover:
                    try:
                        result = self.plug(op, leftover, result)
                    except _Fail:
                        continue
                if result is term or (leftover and self.canon(result) == self.canon(term)):
                    continue  # no progress (e.g. variables matched the identity)
                self.rewrite_count += 1
                if tracer is not None:
                    if equation.cond is not None:
                        tracer.success(trial_number)
                    tracer.replace(term, result, self._whole(term), self._whole(result), self.context_kind)
                return result
            else:
                if tracer is not None and trial_number is not None:
                    tracer.exhausted(trial_number)
        return None

    def _whole(self, term):
        """Rebuild the enclosing term around ``term`` (for ``trace whole``)."""
        for wrap in reversed(self.context):
            term = wrap(term)
        return term

    def _fire(self, equation: Equation, substitution: dict) -> Optional[Term]:
        if equation.cond is not None:
            if self.tracer is not None and self.tracer.whole:
                saved_context = (self.context, self.context_kind)
                self.context, self.context_kind = [], "cond"
                try:
                    condition = self.normalize(subst(equation.cond, substitution))
                finally:
                    self.context, self.context_kind = saved_context
            else:
                condition = self.normalize(subst(equation.cond, substitution))
            if self.true is None or condition != self.true:
                return None
        if equation.py is not None:
            return self._run_py(equation, substitution)
        return subst(equation.rhs, substitution)

    def _run_py(self, equation: Equation, substitution: dict) -> Optional[Term]:
        """Evaluate a Python-hook right-hand side (``eq ... = py: expr``).
        Variables bound to literals appear as plain Python values; failure
        (arithmetic error, ``fail()``) simply means the rule does not apply."""
        namespace = dict(_PY_ENV)
        namespace["equal"] = self.eq_mod
        namespace["similar"] = lambda first, second: first == second
        namespace["sort_of"] = self.module.sort_of
        for variable in equation.vars.values():
            bound = substitution.get(variable)
            if bound is None:
                continue
            if type(bound) is Literal:
                namespace[variable.name] = bound.value
            elif variable.sort.builtin is not None:
                return None
            else:
                namespace[variable.name] = bound
        try:
            value = eval(equation.py, {"__builtins__": _SAFE_BUILTINS}, namespace)
        except (RuleFailure, ZeroDivisionError, OverflowError, ValueError, TypeError):
            return None
        return self.from_python(value)

    def from_python(self, value) -> Optional[Term]:
        if value is None:
            return None
        if isinstance(value, Term):
            return value
        if isinstance(value, bool):
            return self.true if value else self.false
        return make_literal_from_value(self.module, value)

    def _apply_special(self, term: App) -> Optional[Term]:
        op = term.op
        kind = op.kind
        if kind == "retract":
            inner = term.args[0]
            if self.sort_ok(inner, op.coarity):
                return inner
            return None
        if self.true is None:
            return None
        if kind == "ite":
            condition = term.args[0]
            if condition == self.true:
                self.rewrite_count += 1
                return term.args[1]
            if condition == self.false:
                self.rewrite_count += 1
                return term.args[2]
            return None
        first, second = term.args
        if kind == "eq":
            self.rewrite_count += 1
            return self.true if self.eq_mod(first, second) else self.false
        if kind == "neq":
            self.rewrite_count += 1
            return self.false if self.eq_mod(first, second) else self.true
        if kind == "same":
            self.rewrite_count += 1
            return self.true if first == second else self.false
        if kind == "nsame":
            self.rewrite_count += 1
            return self.false if first == second else self.true
        return None

    def _implicit_rules(self, term: App, theory) -> Optional[Term]:
        """Identity-rules (idr:) and idempotence, applied as built-in rules."""
        op = term.op
        if len(term.args) != 2:
            return None
        current_items = self.items(term, op) if theory.assoc else list(term.args)
        reduced = current_items
        if theory.identity is not None:
            identity_canon = self.canon(theory.identity)
            reduced = [item for item in reduced if self.canon(item) != identity_canon]
        if theory.idem:
            reduced = self._dedupe(reduced, theory.comm)
        if len(reduced) == len(current_items):
            return None
        if not reduced:
            return theory.identity
        if len(reduced) == 1:
            return reduced[0]
        if theory.assoc:
            return self.rebuild(op, reduced)
        return self.make_term(op, reduced)


class _PlainTheory:
    """Stand-in for operators with no equational attributes."""
    assoc = comm = idem = False
    identity = None
    id_rules = False
    plain = True


_PLAIN = _PlainTheory()


def _fail(message="rule failure"):
    raise RuleFailure(message)


_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "int": int, "float": float, "str": str,
    "len": len, "round": round, "pow": pow, "divmod": divmod, "bool": bool,
    "True": True, "False": False, "None": None, "sum": sum,
}
_PY_ENV = {
    "Fraction": Fraction, "math": math, "fail": _fail,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "atan": math.atan, "pi": math.pi,
    "trunc": math.trunc,
}


def _quo(dividend, divisor):
    """Lisp ``truncate``: integer division rounding toward zero."""
    quotient = abs(dividend) // abs(divisor)
    return quotient if (dividend < 0) == (divisor < 0) else -quotient


def _rem(dividend, divisor):
    """Lisp ``rem``: result has the sign of the dividend."""
    return dividend - divisor * _quo(dividend, divisor)


_PY_ENV.update({"quo": _quo, "rem": _rem})
