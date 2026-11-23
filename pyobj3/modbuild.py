"""
Processing of module items: imports, sorts, operators, variables and
equations.  Also computes the precedence/gather/pattern of mixfix operators
following the rules of the original (``mod_eval$$compute_op_parsing_attr``).
"""
from __future__ import annotations

from typing import Optional

from .core import Sort, Operator, Var, App, Equation, Term, term_vars
from .module import Module
from .builtins import REGISTRY
from .errors import ObjError, ModuleError, ParseError
from .rulegen import is_bad_rule, WARNING as RULEGEN_WARNING
from .parser import Parser

MAX_PREC = 127
DEFAULT_PREC = 41
DEFAULT_UNARY_PREC = 15

EQ_KEYWORDS = {"eq", "ceq", "cq", "bq", "beq", "cbq", "cbeq"}
IMPORT_KEYWORDS = {
    "pr": "protecting", "protecting": "protecting",
    "ex": "extending", "extending": "extending",
    "inc": "including", "including": "including",
    "us": "using", "using": "using",
}


def split_depth0(tokens: list[str], sep: str) -> list[list[str]]:
    """Split at ``sep`` wherever it is not inside ( ) or [ ]."""
    parts, current, depth = [], [], 0
    for token in tokens:
        if token in ("(", "["):
            depth += 1
        elif token in (")", "]"):
            depth -= 1
        if token == sep and depth == 0:
            parts.append(current)
            current = []
        else:
            current.append(token)
    parts.append(current)
    return parts


def find_depth0(tokens: list[str], sep: str, start: int = 0) -> int:
    """Index of the first ``sep`` outside brackets at or after ``start``, or -1."""
    depth = 0
    for i in range(start, len(tokens)):
        token = tokens[i]
        if token in ("(", "["):
            depth += 1
        elif token in (")", "]"):
            depth -= 1
        elif token == sep and depth == 0:
            return i
    return -1


def find_cond_if(tokens: list[str], start: int) -> int:
    """Index of the 'if' that introduces a condition (not an if-then-else-fi)."""
    open_ifs: list[int] = []
    for i in range(start, len(tokens)):
        token = tokens[i]
        if token == "if":
            open_ifs.append(i)
        elif token == "fi" and open_ifs:
            open_ifs.pop()
    return open_ifs[0] if open_ifs else -1


def unit_at(tokens: list[str], i: int) -> tuple[list[str], int]:
    """Parenthesised group or single token starting at i."""
    if i < len(tokens) and tokens[i] == "(":
        depth = 0
        for j in range(i, len(tokens)):
            if tokens[j] == "(":
                depth += 1
            elif tokens[j] == ")":
                depth -= 1
                if depth == 0:
                    return tokens[i + 1:j], j + 1
        raise ObjError("unbalanced parentheses in attribute")
    if i < len(tokens):
        return [tokens[i]], i + 1
    raise ObjError("missing value in attribute")


class OpAttrs:
    """Attributes collected from an operator's ``[ ... ]`` list."""

    def __init__(self):
        self.assoc = self.comm = self.idem = False
        self.identity_src: Optional[list[str]] = None
        self.id_rules = False
        self.prec: Optional[int] = None
        self.gather: Optional[list[str]] = None
        self.strat: Optional[list[int]] = None
        self.memo = False
        self.kind: Optional[str] = None
        self.constr = False


def parse_attrs(tokens: list[str]) -> OpAttrs:
    attrs = OpAttrs()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if token in ("assoc", "associative"):
            attrs.assoc = True
        elif token in ("comm", "commutative"):
            attrs.comm = True
        elif token in ("idem", "idempotent"):
            attrs.idem = True
        elif token in ("id:", "identity:", "idr:", "identity-rules:"):
            value, i = unit_at(tokens, i)
            attrs.identity_src = value
            attrs.id_rules = token in ("idr:", "identity-rules:")
        elif token == "prec" or token == "precedence":
            attrs.prec = int(tokens[i])
            i += 1
        elif token.isdigit():
            attrs.prec = int(token)
        elif token in ("gather", "gathering"):
            value, i = unit_at(tokens, i)
            attrs.gather = value
        elif token in ("strat", "strategy"):
            value, i = unit_at(tokens, i)
            try:
                attrs.strat = [int(word) for word in value]
            except ValueError:
                raise ObjError("bad strategy: %s" % " ".join(value))
        elif token == "(":  # bare strategy list (older syntax)
            value, i = unit_at(tokens, i - 1)
            try:
                attrs.strat = [int(word) for word in value]
            except ValueError:
                pass
        elif token == "memo":
            attrs.memo = True
        elif token in ("poly", "polymorphic"):
            value, i = unit_at(tokens, i)
            name = "".join(value)
            if "if_resolver" in name or name == "ite":
                attrs.kind = "ite"
        elif token == "constr":
            attrs.constr = True
        elif token in ("intrinsic", "nonstrict", "metadata"):
            if token == "metadata":
                _, i = unit_at(tokens, i)
        # unknown attributes are ignored (as in the original)
    return attrs


def make_form(name: tuple[str, ...], arity: tuple[Sort, ...], attrs: OpAttrs):
    """
    Work out ``(prec, gather, form, standard)`` for an operator declaration.

    ``form`` is the mixfix pattern for the parser: ``("t", token)`` for a
    literal token and ``("a", limit, sort)`` for an argument slot, where
    ``limit`` is the highest precedence an argument there may have.  The
    defaults follow the original's ``compute_op_parsing_attr``: prefix
    ``f(_,_)`` operators bind tightest, a prefix operator ``op _`` defaults
    to 15, other mixfix to 41, and gather ``E`` accepts equal precedence while
    ``e`` demands strictly lower (which makes ``assoc`` operators nest to the right).
    """
    standard = "_" not in name
    if attrs.prec is not None:
        prec = attrs.prec
    elif standard:
        prec = 0
    elif name[0] != "_" and name[-1] != "_":
        prec = 0
    elif name[-1] == "_" and "_" not in name[:-1]:
        prec = DEFAULT_UNARY_PREC
    else:
        prec = DEFAULT_PREC
    lower_prec = prec - 1 if prec else 0
    if attrs.gather is not None:
        gather = list(attrs.gather)
    elif standard:
        gather = ["&"] * len(arity)
    elif name[-1] == "_" and "_" not in name[:-1]:
        gather = ["E"]
    elif attrs.assoc:
        gather = ["e", "E"]
    else:
        gather = ["E"] * len(arity)
    if standard:
        pattern: list[str] = list(name)
        if arity:
            pattern.append("(")
            for k in range(len(arity)):
                if k:
                    pattern.append(",")
                pattern.append("_")
            pattern.append(")")
    else:
        pattern = list(name)
    form = []
    arg_index = 0
    gather_index = 0
    for k, piece in enumerate(pattern):
        if piece == "_":
            if arg_index >= len(arity):
                raise ObjError("too many _'s in operator pattern %s" % " ".join(name))
            gather_spec = gather[gather_index] if gather_index < len(gather) else "E"
            previous = pattern[k - 1] if k > 0 else None
            following = pattern[k + 1] if k + 1 < len(pattern) else None
            # An argument between two literal tokens (as in "( _ )") is delimited,
            # so any precedence is allowed there.
            if gather_spec == "&":
                limit = MAX_PREC
            elif previous is not None and previous != "_" and following is not None and following != "_":
                limit = MAX_PREC
            elif gather_spec == "e":
                limit = lower_prec
            elif gather_spec == "E":
                limit = prec
            else:
                limit = 0
            form.append(("a", limit, arity[arg_index]))
            arg_index += 1
            gather_index += 1
        else:
            form.append(("t", piece))
    if arg_index != len(arity):
        raise ObjError("operator pattern %s has too few argument positions"
                       % " ".join(name))
    return prec, gather, form, standard


class Builder:
    """
    Adds module items (sorts, operators, equations, ...) to a module.
    Module expressions are resolved through the interpreter.
    """

    def __init__(self, interp):
        self.interp = interp

    # Dispatch on the item keyword
    def add_item(self, module: Module, keyword: str, tokens: list[str], source: str = "",
                 token_objects=None):
        if keyword in IMPORT_KEYWORDS:
            return self.add_import(module, IMPORT_KEYWORDS[keyword], tokens)
        if keyword in ("sort", "sorts"):
            return self.add_sorts(module, tokens)
        if keyword in ("subsort", "subsorts"):
            return self.add_subsorts(module, tokens)
        if keyword == "bsort":
            return self.add_bsort(module, tokens)
        if keyword in ("op", "ops"):
            return self.add_ops(module, keyword, tokens)
        if keyword in ("var", "vars"):
            return self.add_vars(module, tokens)
        if keyword in EQ_KEYWORDS:
            return self.add_equation(module, keyword, tokens, source, token_objects)
        if keyword == "let":
            return self.add_let(module, tokens)
        if keyword in ("psort", "principal-sort"):
            return self.set_psort(module, tokens)
        if keyword == "as":
            return self.add_sort_constraint(module, tokens)
        if keyword == "op-as":
            return self.add_op_as(module, tokens)
        if keyword in ("dfn", "define"):
            return self.add_dfn(module, tokens)
        if keyword == "vars-of":
            return self.add_vars_of(module, tokens)
        raise ObjError("unsupported module item: %s" % keyword)

    # Sort constraints
    def add_sort_constraint(self, module, tokens):
        """``as S : T if C .`` -- OBJ3 parses and checks sort constraints but
        does not support them ("general sort constraint not currently
        handled (ignored)").  Same here."""
        say = self.interp.say
        if ":" not in tokens or "if" not in tokens:
            raise ObjError("syntax: as SORT : TERM if CONDITION .")
        colon = tokens.index(":")
        if_pos = find_depth0(tokens, "if", colon)
        sort_name, term_tokens, condition_tokens = tokens[0], tokens[colon + 1:if_pos], tokens[if_pos + 1:]
        try:
            sort = module.find_sort(sort_name)
        except ModuleError:
            sort = None
            say("Unknown sort in sort constraint")
        parser = Parser(module)
        try:
            term_parses = [parse.term for parse in parser.parse(term_tokens)]
        except ParseError:
            term_parses = []
        try:
            condition_parses = [parse.term for parse in Parser(module).parse(condition_tokens)]
        except ParseError:
            condition_parses = []
        if not term_parses:
            say("No parse for term in sort constraint")
        if len(condition_parses) != 1:
            say("No single parse for condition in sort constraint")
        if len(term_parses) > 1:
            say("Term in sort constraint is ambiguous")
        elif len(term_parses) == 1:
            if sort is not None and not module.order.leq(sort, module.sort_of(term_parses[0])):
                say("Specified sort and sort of term incompatible")
            if len(condition_parses) == 1 and not set(term_vars(condition_parses[0])) <= set(term_vars(term_parses[0])):
                say("Condition variables not subset of those in term")
        say("Error: general sort constraint not currently handled (ignored)")
        say("    as %s : %s if %s ." % (sort_name, " ".join(term_tokens), " ".join(condition_tokens)))

    def add_op_as(self, module, tokens):
        """``op-as f : A -> B for T if C [attrs] .`` -- treated as a plain
        operator declaration (with the original's warning)."""
        try:
            colon = tokens.index(":")
            for_pos = tokens.index("for")
        except ValueError:
            raise ObjError("syntax: op-as OP : SORTS -> SORT for TERM if COND [attrs] .")
        if_pos = find_depth0(tokens, "if", for_pos)
        attrs_pos = find_depth0(tokens, "[", if_pos if if_pos >= 0 else for_pos)
        declaration = tokens[:for_pos] + (tokens[attrs_pos:] if attrs_pos >= 0 else [])
        self.interp.say("Warning: operator assertion being treated simply as a "
                        "declaration\nfor operator: " + " ".join(tokens[:colon]))
        self.add_ops(module, "op", declaration)

    # Psort / dfn / vars-of
    def set_psort(self, module, tokens):
        if len(tokens) != 1:
            raise ObjError("psort: one sort name expected")
        module.psort = module.find_sort(tokens[0])

    def add_dfn(self, module, tokens):
        """``dfn NAME is MODEXP .`` -- import MODEXP with its principal sort
        renamed to NAME."""
        from . import modexp
        if len(tokens) < 3 or tokens[1] != "is":
            raise ObjError("syntax: dfn NAME is MODEXP .")
        name = tokens[0]
        base = self.interp.eval_modexp(tokens[2:])
        principal = modexp.principal_sort(base)
        if principal is None:
            raise ObjError("In defined/dfn, module has no principal sort: " + base.name)
        renamed = modexp.rename_module(base, [("sort", principal.name, name)],
                                       "%s * (sort %s to %s)" % (base.name, principal.name, name))
        module.add_import("protecting", renamed)

    def add_vars_of(self, module, tokens):
        """``vars-of [MODEXP] .`` -- copy the variable declarations of a module
        whose sorts belong to this module."""
        if tokens:
            source_module = self.interp.eval_modexp(tokens)
        elif self.interp.last_module is not None:
            source_module = self.interp.last_module
        else:
            raise ObjError("Error: no last module")
        own_sorts = set(module.sorts)
        existing_vars = module.flat().vars
        for var in source_module.flat().vars.values():
            if var.sort not in own_sorts:
                continue
            existing = existing_vars.get(var.name)
            if existing is not None and existing.sort is not var.sort:
                self.interp.say("Warning: imported variable discarded due to conflict\n"
                                "with existing variable: " + var.name)
            elif existing is None:
                module.vars[var.name] = Var(var.name, var.sort)
        module.invalidate()

    # Imports
    def add_import(self, module, mode, tokens):
        imported = self.interp.eval_modexp(tokens)
        module.add_import(mode, imported)

    # Sorts
    def add_sorts(self, module, tokens):
        for name in tokens:
            if name == ".":
                continue
            already_declared = [s for s in module.sorts if s.name == name]
            if already_declared:
                continue
            module.sorts.append(Sort(name, module.name))
        module.invalidate()

    def add_bsort(self, module, tokens):
        if not tokens:
            raise ObjError("bsort needs a name")
        name = tokens[0]
        builtin = REGISTRY.get(name)
        if builtin is None:
            raise ObjError("unknown built-in sort %s (known: %s)"
                           % (name, ", ".join(sorted(REGISTRY))))
        sort = Sort(name, module.name, builtin=builtin)
        module.sorts.append(sort)
        module.bsorts.append(sort)
        module.invalidate()

    def add_subsorts(self, module, tokens):
        groups = split_depth0(tokens, "<")
        if len(groups) < 2:
            raise ObjError("subsort declaration needs '<'")
        names = [[token for token in group] for group in groups]
        sorts = [[self._sort(module, name) for name in group] for group in names]
        for lower_group, upper_group in zip(sorts, sorts[1:]):
            for lower in lower_group:
                for upper in upper_group:
                    module.subsorts.append((lower, upper))
        module.invalidate()

    def _sort(self, mod: Module, name: str) -> Sort:
        try:
            return mod.find_sort(name)
        except ModuleError:
            raise ObjError("unknown sort %s" % name)

    # Ops
    def add_ops(self, module, keyword, tokens):
        colon = find_depth0(tokens, ":")
        if colon < 0:
            raise ObjError("operator declaration needs ':'")
        head = tokens[:colon]
        after_colon = tokens[colon + 1:]
        if keyword == "op":
            names = [self._strip_parens(head)]
        else:
            names = self._names(head)
        # attributes
        attrs = OpAttrs()
        if "[" in after_colon:
            attrs_start = after_colon.index("[")
            attrs_end = len(after_colon) - 1 - after_colon[::-1].index("]") if "]" in after_colon else len(after_colon)
            attrs = parse_attrs(after_colon[attrs_start + 1:attrs_end])
            after_colon = after_colon[:attrs_start]
        arrow = find_depth0(after_colon, "->")
        if arrow < 0:
            raise ObjError("operator declaration needs '->'")
        arity = tuple(self._sort(module, n) for n in after_colon[:arrow])
        if len(after_colon[arrow + 1:]) != 1:
            raise ObjError("operator needs exactly one result sort")
        coarity = self._sort(module, after_colon[arrow + 1])
        for name in names:
            self.make_op(module, tuple(name), arity, coarity, attrs)
        module.invalidate()

    @staticmethod
    def _strip_parens(tokens):
        tokens = list(tokens)
        if len(tokens) >= 2 and tokens[0] == "(" and tokens[-1] == ")":
            depth = 0
            balanced = True
            for i, token in enumerate(tokens):
                depth += (token == "(") - (token == ")")
                if depth == 0 and i < len(tokens) - 1:
                    balanced = False
                    break
            if balanced:
                return tokens[1:-1]
        return tokens

    @staticmethod
    def _names(tokens):
        names, i = [], 0
        while i < len(tokens):
            if tokens[i] == "(":
                unit, i = unit_at(tokens, i)
                names.append(unit)
            else:
                names.append([tokens[i]])
                i += 1
        return names

    def make_op(self, mod, name, arity, coarity, attrs: OpAttrs) -> Operator:
        if not name:
            raise ObjError("empty operator name")
        prec, gather, form, standard = make_form(name, arity, attrs)
        if attrs.assoc or attrs.comm:
            if len(arity) != 2:
                self.interp.say("Warning: associative/commutative operator %s has bad rank"
                                % " ".join(name))
        op = Operator(name, arity, coarity, mod.name, prec=prec, gather=gather,
                      form=form, standard=standard, assoc=attrs.assoc,
                      comm=attrs.comm, idem=attrs.idem, id_rules=attrs.id_rules,
                      strat=attrs.strat, memo=attrs.memo, kind=attrs.kind,
                      constr=attrs.constr)
        op.identity_src = attrs.identity_src
        if attrs.kind == "ite" and op.strat is None:
            op.strat = [1, 0]
        mod.ops.append(op)
        return op

    # Vars
    def add_vars(self, module, tokens):
        colon = find_depth0(tokens, ":")
        if colon < 0 or colon + 2 != len(tokens):
            raise ObjError("variable declaration should be: var X Y : Sort .")
        sort = self._sort(module, tokens[colon + 1])
        for name in tokens[:colon]:
            module.vars[name] = Var(name, sort)
        module.invalidate()

    # Equations
    def parse_term(self, mod, toks, what="term", extra_vars=None) -> Term:
        return Parser(mod, extra_vars).parse_one(toks, what)

    def add_equation(self, module, keyword, tokens, source, token_objects):
        equation = self.make_equation(module, keyword, tokens, source, token_objects)
        module.eqs.append(equation)
        module.invalidate()

    def make_equation(self, module, keyword, tokens, source="", token_objects=None) -> Equation:
        """Build an Equation from ``lhs = rhs`` (plus ``if cond`` for the
        conditional keywords).  For ``bq`` forms the right side is a Python
        expression, located through ``token_objects`` positions in ``source``."""
        label = None  # labels precede the keyword ([ lbl ] eq ...) and are skipped
        builtin = keyword in ("bq", "beq", "cbq", "cbeq")
        conditional = keyword in ("ceq", "cq", "cbq", "cbeq")
        equals_pos = find_depth0(tokens, "=")
        if equals_pos < 0:
            raise ObjError("equation needs '='")
        lhs_tokens = tokens[:equals_pos]
        rest = tokens[equals_pos + 1:]
        cond_tokens = None
        if conditional:
            if builtin:
                # The right side is Python source, which may contain its own
                # "if ... else"; the condition starts at the LAST top-level "if".
                if_pos = -1
                depth = 0
                for i, token in enumerate(rest):
                    depth += (token == "(") - (token == ")")
                    if token == "if" and depth == 0:
                        if_pos = i
            else:
                if_pos = find_cond_if(rest, 0)
            if if_pos < 0:
                raise ObjError("conditional equation needs 'if'")
            cond_tokens = rest[if_pos + 1:]
            rhs_tokens = rest[:if_pos]
            rhs_objects = token_objects[equals_pos + 1:equals_pos + 1 + if_pos] if token_objects else None
        else:
            rhs_tokens = rest
            rhs_objects = token_objects[equals_pos + 1:] if token_objects else None
        if not lhs_tokens or not rhs_tokens:
            raise ObjError("equation has an empty side")
        parser = Parser(module)
        lhs = self._parse_side(parser, lhs_tokens, "left-hand side", None)
        cond = None
        if cond_tokens:
            cond = self._parse_bool(parser, module, cond_tokens)
        if builtin:
            if not rhs_objects:
                raise ObjError("builtin equation needs a Python expression")
            # Take the Python text straight from the source, so spacing and
            # punctuation reach compile() exactly as written.
            python_source = source[rhs_objects[0].start:rhs_objects[-1].end]
            try:
                code = compile(python_source.strip(), "<bq %s>" % " ".join(lhs_tokens), "eval")
            except SyntaxError as error:
                raise ObjError("bad Python expression in builtin equation: %s (%s)"
                               % (python_source, error.msg))
            return Equation(lhs, None, cond, py=code, py_src=python_source.strip(),
                            label=label, module_name=module.name)
        rhs = self._parse_side(parser, rhs_tokens, "right-hand side", module.sort_of(lhs))
        return Equation(lhs, rhs, cond, label=label, module_name=module.name)

    def _parse_bool(self, parser, module, tokens):
        candidates = parser.parse(tokens)
        bool_sorts = module.find_sorts("Bool")
        if bool_sorts:
            candidates = [candidate for candidate in candidates if module.order.same_component(module.sort_of(candidate.term), bool_sorts[-1])] or candidates
        candidates = parser.select(candidates)
        if not candidates:
            raise ParseError("no parse for condition: %s" % " ".join(tokens))
        if len(candidates) > 1:
            raise ParseError("ambiguous condition: %s" % " ".join(tokens))
        return candidates[0].term

    def _parse_side(self, parser, tokens, what, ref_sort):
        """Parse one side of an equation.  When the left side's sort is known,
        right-hand parses of an unrelated sort are discarded before reporting
        ambiguity."""
        candidates = parser.parse(tokens)
        if ref_sort is not None:
            same = [candidate for candidate in candidates if parser.order.same_component(parser.module.sort_of(candidate.term), ref_sort)]
            candidates = parser.select(same) if same else candidates
        if not candidates:
            raise ParseError("no parse for %s: %s" % (what, " ".join(tokens)))
        if len(candidates) > 1:
            from .printer import term_to_str
            raise ParseError("ambiguous %s: %s (e.g. %s | %s)" % (
                what, " ".join(tokens), term_to_str(candidates[0].term, parser.module),
                term_to_str(candidates[1].term, parser.module)))
        return candidates[0].term

    # Let
    def add_let(self, module, tokens):
        equals_pos = find_depth0(tokens, "=")
        if equals_pos != 1:
            raise ObjError("let syntax: let NAME = TERM .")
        name = tokens[0]
        term = self.parse_term(module, tokens[equals_pos + 1:], "let-term")
        sort = module.sort_of(term)
        op = Operator((name,), (), sort, module.name, prec=0, form=[("t", name)],
                      standard=False)
        module.ops.append(op)
        module.eqs.append(Equation(App(op, ()), term, None, module_name=module.name))
        module.invalidate()

    # Finish
    def finalize(self, module: Module):
        """Resolve identity expressions once the full signature is known."""
        module.invalidate()
        changed = False
        if module.kind == "th" and any(is_bad_rule(equation) for member in module.closure() for equation in member.eqs):
            self.interp.say(RULEGEN_WARNING)
        for op in module.ops:
            if op.identity_src is not None and op.identity is None:
                try:
                    identity = Parser(module).parse_one(op.identity_src, "identity")
                except ObjError as error:
                    self.interp.say("Warning: bad identity for operator %s: %s"
                                    % (" ".join(op.name), error))
                    op.identity_src = None
                    continue
                if term_vars(identity):
                    self.interp.say("Warning: identity expression contains a variable, ignored")
                    op.identity_src = None
                    continue
                op.identity = identity
                changed = True
        if changed:
            module.invalidate()
