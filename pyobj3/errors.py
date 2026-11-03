"""
Exception types for the OBJ3 interpreter.
"""

class ObjError(Exception):
    """Any user-visible error (printed, then control returns to top level)."""


class ParseError(ObjError):
    pass


class ModuleError(ObjError):
    pass


class RuleFailure(Exception):
    """Raised by a built-in equation to say 'this rule does not apply'."""
