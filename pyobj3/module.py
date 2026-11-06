"""
Modules, their sort orders, and the flattened view used for parsing and rewriting.
"""
from __future__ import annotations

from typing import Optional

from .core import (Sort, Operator, Term, Var, Literal, App, Equation,
                   union_ops)
from .errors import ModuleError


class SortOrder:
    """
    The subsort relation of a module, closed under reflexivity and transitivity.

    Also records the connected components of the sort graph: two sorts in the
    same component form a "kind", and a term of one may be turned into the
    other with a retract (this is how partial operations are handled).
    """

    def __init__(self, sorts: list[Sort], pairs: list[tuple[Sort, Sort]]):
        self.sorts = sorts
        below: dict[Sort, set[Sort]] = {s: set() for s in sorts} # strictly smaller
        above: dict[Sort, set[Sort]] = {s: set() for s in sorts} # strictly greater
        direct_up: dict[Sort, set[Sort]] = {s: set() for s in sorts}
        for lower, upper in pairs:
            if lower is not upper and lower in direct_up and upper in direct_up:
                direct_up[lower].add(upper)
        # Transitive closure: walk upward from every sort.
        for s in sorts:
            seen: set[Sort] = set()
            stack = list(direct_up[s])
            while stack:
                reached = stack.pop()
                if reached in seen:
                    continue
                seen.add(reached)
                stack.extend(direct_up[reached])
            above[s] = seen
            for reached in seen:
                below[reached].add(s)
        self.above, self.below = above, below
        # Connected components of the undirected subsort graph.
        component_of: dict[Sort, int] = {}
        component_id = 0
        neighbours: dict[Sort, set[Sort]] = {s: set() for s in sorts}
        for lower, upper_sorts in direct_up.items():
            for upper in upper_sorts:
                neighbours[lower].add(upper)
                neighbours[upper].add(lower)
        for s in sorts:
            if s in component_of:
                continue
            component_id += 1
            stack = [s]
            while stack:
                reached = stack.pop()
                if reached in component_of:
                    continue
                component_of[reached] = component_id
                stack.extend(neighbours[reached])
        self.component_of = component_of

    def leq(self, sort_a: Sort, sort_b: Sort) -> bool:
        if sort_a is sort_b or sort_b.name == "Universal":
            return True
        return sort_b in self.above.get(sort_a, ())

    def lt(self, sort_a: Sort, sort_b: Sort) -> bool:
        return sort_a is not sort_b and self.leq(sort_a, sort_b)

    def same_component(self, sort_a: Sort, sort_b: Sort) -> bool:
        if sort_a is sort_b:
            return True
        component_a, component_b = self.component_of.get(sort_a), self.component_of.get(sort_b)
        return component_a is not None and component_a == component_b

    def lub(self, sort_a: Sort, sort_b: Sort) -> Sort:
        """Least upper bound; when not unique pick a minimal common
        supersort (as the original does for if_then_else_fi)."""
        if self.leq(sort_a, sort_b):
            return sort_b
        if self.leq(sort_b, sort_a):
            return sort_a
        common = [s for s in (self.above.get(sort_a, set()) & self.above.get(sort_b, set()))]
        if not common:
            return sort_a
        mins = [s for s in common if not any(self.lt(other, s) for other in common)]
        mins.sort(key=lambda s: s.uid)
        return mins[0]


class Module:
    """
    An OBJ3 object or theory: its own declarations plus the modules it imports.

    Only the module's own sorts, operators, variables and equations are
    stored here; ``flat()`` merges them with everything imported.
    """

    def __init__(self, name: str, kind: str = "obj"):
        self.name = name
        self.kind = kind # 'obj' | 'th'
        self.imports: list[tuple[str, "Module"]] = []
        self.sorts: list[Sort] = [] # declared in this module (not imported)
        self.subsorts: list[tuple[Sort, Sort]] = []
        self.ops: list[Operator] = []
        self.vars: dict[str, Var] = {}
        self.eqs: list[Equation] = []
        self.bsorts: list = [] # sorts backed by Python values (Nat, Id, ...)
        self.params: list = [] # formal parameters (modexp.ParamInfo)
        self.psort = None # explicit principal sort
        self.lets: dict[str, Term] = {}
        self._flat = None
        self._sortcache: dict = {} # term -> least sort
        self._rank_cache: dict = {} # (operator family, argument sorts) -> least sort
        self.engine = None # rewrite engine, built on first use

    def invalidate(self):
        """Drop everything derived from the declarations (call after changing them)."""
        self._flat = None
        self._sortcache = {}
        self._rank_cache = {}
        self.engine = None

    def add_import(self, mode: str, module: "Module"):
        if module is self:
            raise ModuleError("module cannot import itself")
        if all(existing is not module for _, existing in self.imports):
            self.imports.append((mode, module))
        self.invalidate()

    def closure(self) -> list["Module"]:
        """All modules this one depends on, imports first, self last."""
        ordered: list[Module] = []
        seen: set[int] = set()

        def visit(module: "Module"):
            if id(module) in seen:
                return
            seen.add(id(module))
            for _, imported in module.imports:
                visit(imported)
            ordered.append(module)
        visit(self)
        return ordered

    def flat(self) -> "Flat":
        if self._flat is None:
            self._flat = Flat(self)
        return self._flat

    @property
    def order(self) -> SortOrder:
        return self.flat().order

    def find_sorts(self, name: str) -> list[Sort]:
        return self.flat().sort_by_name.get(name, [])

    def find_sort(self, name: str) -> Sort:
        # A sort can be written qualified as Sort.MODULE.
        if "." in name and name not in self.flat().sort_by_name:
            sort_name, _, module_name = name.rpartition(".")
            for sort in self.flat().sort_by_name.get(sort_name, []):
                if sort.module_name == module_name:
                    return sort
        matches = self.flat().sort_by_name.get(name, [])
        if not matches:
            # Sorts of a parameter are stored as Sort.Param (Elt.X); accept the
            # bare name when it is unambiguous.
            prefix = name + "."
            matches = [sort for name_in_flat, sorts in self.flat().sort_by_name.items()
                       if name_in_flat.startswith(prefix) for sort in sorts]
            if len(matches) > 1:
                raise ModuleError("ambiguous sort %s in module %s" % (name, self.name))
        if not matches:
            raise ModuleError("unknown sort %s in module %s" % (name, self.name))
        return matches[-1]  # most recently declared wins

    def leq(self, sort_a: Sort, sort_b: Sort) -> bool:
        return self.flat().order.leq(sort_a, sort_b)

    def sort_of(self, term: Term) -> Sort:
        term_type = type(term)
        if term_type is Var or term_type is Literal:
            return term.sort
        cache = self._sortcache
        sort = cache.get(term)
        if sort is not None:
            return sort
        sort = self._compute_sort(term)
        cache[term] = sort
        return sort

    def _compute_sort(self, term: App) -> Sort:
        op = term.op
        flat = self.flat()
        if op.kind == "ite":
            then_sort = self.sort_of(term.args[1])
            else_sort = self.sort_of(term.args[2])
            return flat.order.lub(then_sort, else_sort)
        if op.kind is not None:
            return op.coarity
        if not term.args:
            return op.coarity
        arg_sorts = tuple(self.sort_of(then_sort) for then_sort in term.args)
        key = (op.fam, arg_sorts)
        sort = self._rank_cache.get(key)
        if sort is None:
            sort = self.lowest_rank(op, arg_sorts)[1]
            self._rank_cache[key] = sort
        return sort

    def lowest_rank(self, op: Operator, arg_sorts) -> tuple[Operator, Sort]:
        """The least-sorted declaration of ``op``'s family applicable to
        arguments of the given sorts."""
        flat = self.flat()
        order = flat.order
        members = flat.fam_members.get(op.fam) or [op]
        best = None
        for member in members:
            if len(member.arity) != len(arg_sorts):
                continue
            if all(order.leq(arg_sort, declared_sort) for arg_sort, declared_sort in zip(arg_sorts, member.arity)):
                if best is None or order.lt(member.coarity, best.coarity):
                    best = member
        if best is not None:
            return best, best.coarity
        # The arguments fit no declaration: take any declaration whose argument
        # sorts lie in the same components (the term is a candidate for retracts).
        for member in members:
            if len(member.arity) == len(arg_sorts) and all(
                    order.same_component(arg_sort, declared_sort) for arg_sort, declared_sort in zip(arg_sorts, member.arity)):
                return member, member.coarity
        return op, op.coarity

    def find_const(self, name: str, sort_name: Optional[str] = None) -> Optional[Operator]:
        for op in self.flat().ops:
            if op.name == (name,) and not op.arity and (
                    sort_name is None or op.coarity.name == sort_name):
                return op
        return None

    def __repr__(self):
        return "Module(%s)" % self.name


class Flat:
    """The flattened (import-closed) view of a module."""

    def __init__(self, module: Module):
        closure_modules = module.closure()
        self.modules = closure_modules
        self.sorts: list[Sort] = []
        pairs: list[tuple[Sort, Sort]] = []
        self.ops: list[Operator] = []
        self.vars: dict[str, Var] = {}
        self.eqs: list[Equation] = []
        self.bsorts: list = []
        for member_module in closure_modules:
            self.sorts.extend(sort for sort in member_module.sorts if sort not in self.sorts)
            pairs.extend(member_module.subsorts)
            self.ops.extend(op for op in member_module.ops if op not in self.ops)
            self.vars.update(member_module.vars)
            self.eqs.extend(member_module.eqs)
            self.bsorts.extend(second for second in member_module.bsorts if second not in self.bsorts)
        self.order = SortOrder(self.sorts, pairs)
        self.sort_by_name: dict[str, list[Sort]] = {}
        for sort in self.sorts:
            self.sort_by_name.setdefault(sort.name, []).append(sort)
        # Group overloaded declarations into families: same name and number of
        # arguments, with argument sorts in matching connected components.  A
        # family is the unit rules and theories attach to (so a rule for
        # _+_ on Nat also applies to _+_ declared on a subsort).
        groups: dict[tuple, list[Operator]] = {}
        for op in self.ops:
            groups.setdefault((op.name, len(op.arity)), []).append(op)
        for group in groups.values():
            for i, first in enumerate(group):
                for second in group[i + 1:]:
                    if all(self.order.same_component(sort_x, sort_y)
                           for sort_x, sort_y in zip(first.arity, second.arity)):
                        union_ops(first, second)
        self.fam_members: dict[int, list[Operator]] = {}
        for op in self.ops:
            self.fam_members.setdefault(op.fam, []).append(op)
        # One combined theory (assoc/comm/idem/identity) per family.
        self.fam_theory: dict[int, "Theory"] = {}
        for family, members in self.fam_members.items():
            self.fam_theory[family] = Theory(members)


class Theory:
    """
    Equational attributes of an operator family (union over members)
    """
    __slots__ = ("assoc", "comm", "idem", "identity", "id_rules", "zmatch")

    def __init__(self, members):
        self.assoc = any(o.assoc for o in members)
        self.comm = any(o.comm for o in members)
        self.idem = any(o.idem for o in members)
        self.identity = None
        self.id_rules = False
        for o in members:
            if o.identity is not None:
                self.identity = o.identity
                self.id_rules = o.id_rules
                break
        # With id: the identity takes part in matching (a pattern x * y can match a
        # lone subject a, with y bound to the identity);
        # with idr: it only contributes the rules  e * x = x  and  x * e = x.
        self.zmatch = self.identity is not None and not self.id_rules

    @property
    def plain(self):
        return not (self.assoc or self.comm or self.idem or self.identity is not None)
