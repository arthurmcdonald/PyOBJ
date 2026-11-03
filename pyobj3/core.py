"""
Core data model: sorts, operators and terms.
"""
from __future__ import annotations

import itertools
from typing import Optional

_uid = itertools.count(1)

# Operator families: overloaded declarations are merged with a union-find over their uids.
_parent: dict[int, int] = {}
EPOCH = 0  # increased whenever two families merge (invalidates cached hashes)


def _find(uid: int) -> int:
    root = uid
    while _parent.get(root, root) != root:
        root = _parent[root]
    while _parent.get(uid, uid) != root:  # path compression
        _parent[uid], uid = root, _parent[uid]
    return root


def union_ops(first: "Operator", second: "Operator") -> bool:
    global EPOCH
    root_first, root_second = _find(first.uid), _find(second.uid)
    if root_first == root_second:
        return False
    _parent[root_second] = root_first
    EPOCH += 1
    return True


# Sorts
class Sort:
    __slots__ = ("name", "module_name", "uid", "builtin")

    def __init__(self, name: str, module_name: str = "", builtin=None):
        self.name = name
        self.module_name = module_name
        self.uid = next(_uid)
        self.builtin = builtin  # BuiltinSort or None

    def __repr__(self):
        return self.name


# Operators
class Operator:
    """One declaration of an operator (one rank).

    ``name`` is the token tuple, e.g. ``("_", "+", "_")`` or ``("f",)``.
    ``form`` is the parse/print pattern: a list of ``("t", token)`` and
    ``("a", precedence_limit, sort)`` items.
    """

    def __init__(self, name, arity, coarity, module_name="", *,
                 prec=0, gather=None, form=None, standard=True,
                 assoc=False, comm=False, idem=False,
                 identity=None, id_rules=False, strat=None, memo=False,
                 kind=None, constr=False):
        self.uid = next(_uid)
        self.name = tuple(name)
        self.arity = tuple(arity)
        self.coarity = coarity
        self.module_name = module_name
        self.prec = prec              # precedence (lower binds tighter)
        self.gather = gather          # per-argument precedence limits, e.g. ["e", "E"]
        self.form = form              # parse/print pattern, see class docstring
        self.standard = standard      # printed as f(a, b) rather than mixfix
        self.assoc = assoc
        self.comm = comm
        self.idem = idem
        self.identity = identity      # Term or None (filled in lazily)
        self.identity_src = None      # raw tokens, parsed when module is closed
        self.id_rules = id_rules
        self.strat = strat            # evaluation strategy (0 = rewrite here); None = innermost
        self.memo = memo              # remember normal forms of terms headed by this operator
        self.kind = kind              # None | 'ite' | 'eq' | 'neq' | 'same' | 'nsame' | 'retract'
        self.constr = constr          # declared as a constructor
        self.retract = None           # (from_sort, to_sort) when kind == 'retract'

    @property
    def fam(self) -> int:
        """Id of this operator's overload family (the union-find root)."""
        return _find(self.uid)

    @property
    def is_constant(self):
        return not self.arity

    @property
    def text(self):
        return " ".join(self.name)

    def __repr__(self):
        return "Op(%s : %s -> %s)" % ("".join(self.name),
                                     " ".join(s.name for s in self.arity),
                                     self.coarity.name)


_retracts: dict[tuple[int, int], Operator] = {}


def retract_op(from_sort: Sort, to_sort: Sort) -> Operator:
    key = (from_sort.uid, to_sort.uid)
    retract = _retracts.get(key)
    if retract is None:
        retract = Operator(("r:%s>%s" % (from_sort.name, to_sort.name),), (from_sort,), to_sort,
                           standard=True, kind="retract", strat=[1, 0])
        retract.retract = (from_sort, to_sort)
        _retracts[key] = retract
    return retract


# Terms
class Term:
    __slots__ = ()
    is_var = False


class Var(Term):
    __slots__ = ("name", "sort", "_h")
    is_var = True

    def __init__(self, name: str, sort: Sort):
        self.name = name
        self.sort = sort
        self._h = hash(("V", name, sort.uid))

    def __eq__(self, other):
        return self is other or (type(other) is Var and other.name == self.name
                             and other.sort is self.sort)

    def __hash__(self):
        return self._h

    def __repr__(self):
        return "%s:%s" % (self.name, self.sort.name)


class Literal(Term):
    """Built-in constant (number, quoted id, ...).  ``kind`` identifies the
    Python-level type family so ``1`` and ``1.0`` stay different."""
    __slots__ = ("value", "sort", "kind", "_h")

    def __init__(self, value, sort: Sort, kind: str):
        self.value = value
        self.sort = sort
        self.kind = kind
        self._h = hash(("L", kind, value))

    def __eq__(self, other):
        return self is other or (type(other) is Literal and other.kind == self.kind
                             and other.value == self.value)

    def __hash__(self):
        return self._h

    def __repr__(self):
        return repr(self.value)


class App(Term):
    __slots__ = ("op", "args", "_h", "_he")

    def __init__(self, op: Operator, args=()):
        self.op = op
        self.args = tuple(args)
        self._he = -1
        self._h = 0

    def __hash__(self):
        if self._he != EPOCH:
            self._h = hash((self.op.fam, self.args))
            self._he = EPOCH
        return self._h

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not App:
            return False
        return (self.op is other.op or self.op.fam == other.op.fam) and self.args == other.args

    def __repr__(self):
        if not self.args:
            return "".join(self.op.name)
        return "%s(%s)" % ("".join(self.op.name), ", ".join(map(repr, self.args)))


def term_vars(term: Term, found: Optional[dict] = None) -> dict:
    """Collect the variables of ``term`` (name -> Var), adding to ``found`` if given."""
    if found is None:
        found = {}
    pending = [term]
    while pending:
        node = pending.pop()
        if type(node) is Var:
            found.setdefault(node.name, node)
        elif type(node) is App:
            pending.extend(node.args)
    return found


def subst(term: Term, substitution: dict) -> Term:
    """Apply substitution (Var -> Term).  Structure-sharing."""
    term_type = type(term)
    if term_type is Var:
        return substitution.get(term, term)
    if term_type is App:
        if not term.args:
            return term
        new_args = tuple(subst(arg, substitution) for arg in term.args)
        if all(new_arg is old_arg for new_arg, old_arg in zip(new_args, term.args)):
            return term
        return App(term.op, new_args)
    return term


class Equation:
    """lhs = rhs [if cond].  ``py`` holds compiled Python code for built-in
    (``bq``) equations; the right-hand side is then computed natively."""
    __slots__ = ("lhs", "rhs", "cond", "py", "py_src", "label", "module_name",
                 "vars", "src")

    def __init__(self, lhs, rhs, cond=None, py=None, py_src=None,
                 label=None, module_name="", src=""):
        self.lhs = lhs
        self.rhs = rhs
        self.cond = cond
        self.py = py
        self.py_src = py_src
        self.label = label
        self.module_name = module_name
        self.vars = term_vars(lhs)
        self.src = src
