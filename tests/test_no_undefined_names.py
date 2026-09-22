"""Static guard: no undefined names (F821) in function bodies anywhere under aive/.

Regression for the 2026-09 bug where ``RestoreRunner._build_trt`` referenced
``torch`` without importing it locally. The package keeps heavy imports lazy
(function-local), so every TRT build attempt raised ``NameError: name 'torch' is
not defined`` at runtime and silently fell back to PyTorch-CUDA — invisible to
the non-GPU test suite because the code path only runs on a GPU host.

A full pyflakes pass would catch this class of bug, but this check stays
dependency-free: a small AST walker that resolves names loaded inside function /
lambda bodies against (a) true module-level globals, (b) builtins, and (c) names
bound anywhere within the same function subtree. It is deliberately conservative
(over-approximates what is "known"), so it may miss bugs but must not produce
false positives — any real finding here is a latent runtime NameError.

Only function/lambda bodies are checked: module-level and class-body code runs at
import time, where an undefined name crashes loudly instead of hiding.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG_DIR = REPO / "aive"

_DUNDER_ALLOWLIST = {
    "__name__", "__file__", "__doc__", "__all__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__annotations__", "__path__", "__class__",
}


def _target_names(t: ast.expr) -> set[str]:
    out: set[str] = set()
    if isinstance(t, ast.Name):
        out.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            out |= _target_names(e)
    return out


def _module_globals(tree: ast.Module) -> set[str]:
    """Names bound at true module scope (top-level statements only).

    Recurses into compound statements (if/try/for/with) but NOT into function or
    class bodies — a name imported inside one function is not visible to another.
    """
    names: set[str] = set()

    def rec(stmts: list[ast.stmt]) -> None:
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(st.name)  # the definition itself binds; do not descend
                continue
            if isinstance(st, ast.Import):
                for a in st.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(st, ast.ImportFrom):
                for a in st.names:
                    if a.name != "*":
                        names.add(a.asname or a.name)
            elif isinstance(st, ast.Assign):
                for tgt in st.targets:
                    names.update(_target_names(tgt))
            elif isinstance(st, ast.AnnAssign):
                if st.target is not None:
                    names.update(_target_names(st.target))
            elif isinstance(st, ast.AugAssign):
                if isinstance(st.target, ast.Name):
                    names.add(st.target.id)
            elif isinstance(st, (ast.For, ast.AsyncFor)):
                names.update(_target_names(st.target))
                rec(st.body + st.orelse)
            elif isinstance(st, ast.While):
                rec(st.body + st.orelse)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                for item in st.items:
                    if item.optional_vars is not None:
                        names.update(_target_names(item.optional_vars))
                rec(st.body)
            elif isinstance(st, ast.If):
                rec(st.body + st.orelse)
            elif isinstance(st, ast.Try):
                rec(st.body + st.orelse + st.finalbody)
                for h in st.handlers:
                    if h.name:
                        names.add(h.name)
                    rec(h.body)
            elif isinstance(st, (ast.Global, ast.Nonlocal)):
                names.update(st.names)

    rec(tree.body)
    return names


def _bound_in_subtree(node: ast.AST) -> set[str]:
    """All names bound anywhere inside ``node``'s subtree (locals of any depth)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(n.name)
            for a in n.args.args + n.args.posonlyargs + n.args.kwonlyargs:
                out.add(a.arg)
            if n.args.vararg:
                out.add(n.args.vararg.arg)
            if n.args.kwarg:
                out.add(n.args.kwarg.arg)
        elif isinstance(n, ast.ClassDef):
            out.add(n.name)
        elif isinstance(n, ast.Lambda):
            for a in n.args.args + n.args.posonlyargs + n.args.kwonlyargs:
                out.add(a.arg)
            if n.args.vararg:
                out.add(n.args.vararg.arg)
            if n.args.kwarg:
                out.add(n.args.kwarg.arg)
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    out.add(a.asname or a.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _iter_direct(node: ast.AST):
    """Yield nodes in ``node``'s subtree, stopping at nested function/lambda boundaries.

    Loads inside a nested scope are checked by that scope's own pass (with its own
    bindings), so they must not be judged here — otherwise closure locals would be
    false positives. Decorators / default values of skipped defs are simply not
    checked (conservative: may miss, never falsely flag).
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FUNC_NODES):
            # Surface the scope node itself (callers dispatch visit_scope on it)
            # but never descend into its body — that is checked with its own
            # bindings by the nested-scope recursion.
            yield child
            continue
        yield child
        yield from _iter_direct(child)


def check_source(src: str, filename: str = "<string>") -> list[str]:
    """Return a list of 'file:line name' problems (empty when clean)."""
    tree = ast.parse(src, filename=filename)

    # Star imports make resolution unsound — skip the file rather than guess.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            return []

    known_base = set(dir(builtins)) | _DUNDER_ALLOWLIST | _module_globals(tree)
    problems: list[str] = []

    def visit_scope(scope_node: ast.AST, outer_known: set[str]) -> None:
        # Everything visible inside this scope: outside context + anything bound
        # anywhere in its own subtree (over-approximation: nested-function locals
        # count too — that can only hide bugs, never invent them).
        known = outer_known | _bound_in_subtree(scope_node)
        for n in _iter_direct(scope_node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in known:
                problems.append(f"{filename}:{n.lineno} {n.id}")
        # Nested scopes inherit this scope's visibility (closures).
        for child in _iter_direct(scope_node):
            if isinstance(child, _FUNC_NODES):
                visit_scope(child, known)

    def visit_all(node: ast.AST, outer_known: set[str]) -> None:
        for child in _iter_direct(node):
            if isinstance(child, _FUNC_NODES):
                visit_scope(child, outer_known)

    # Module-level and class-body code runs at import time (loud failures); only
    # function/lambda bodies hide runtime NameErrors. Class-level names are NOT
    # visible to methods via bare loads (no implicit class scope), so methods get
    # the plain module base.
    visit_all(tree, known_base)

    return problems


def test_checker_catches_missing_local_import():
    """The exact 2026-09 bug shape: torch imported in one method, used in another."""
    src = (
        "import os\n"
        "\n"
        "class R:\n"
        "    def __init__(self):\n"
        "        import torch\n"
        "        self.torch = torch\n"
        "\n"
        "    def build(self):\n"
        "        x = torch.zeros(1)\n"
        "        return x\n"
    )
    problems = check_source(src)
    assert any(p.endswith(" torch") for p in problems), f"expected 'torch' flagged, got {problems}"


def test_checker_accepts_local_and_module_imports():
    src = (
        "import math\n"
        "\n"
        "def f(a):\n"
        "    import numpy as np\n"
        "    b = a + 1\n"
        "    return math.sqrt(b) + np.zeros(3)\n"
    )
    assert check_source(src) == []


# Documented exceptions: latent undefined names in code that is provably
# unreachable in this app AND must stay byte-identical to its upstream source.
# Remove an entry if/when the file is touched or the branch becomes reachable.
_KNOWN_DEAD_CODE = {
    # Vendored feinanshan/M2M_VFI pwcnet.py: `pyramid` is a global in the
    # upstream script, absent here. The 'up(pyramid)' branch can only trigger if
    # a Basic strType contains an up(...) part — no M2M network config does.
    ("aive/stages/interpolate/_m2m/pwcnet.py", "pyramid"),
}


def test_no_undefined_names_in_aive():
    """Every function body under aive/ must reference only resolvable names."""
    files = sorted(PKG_DIR.rglob("*.py"))
    assert files, f"no python files found under {PKG_DIR}"
    all_problems: list[str] = []
    for path in files:
        if "__pycache__" in path.parts:
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
            raise AssertionError(f"cannot read {path}: {exc}") from exc
        rel = str(path.relative_to(REPO)).replace("\\", "/")
        try:
            all_problems += check_source(src, filename=rel)
        except SyntaxError as exc:  # pragma: no cover - syntax errors fail elsewhere too
            raise AssertionError(f"syntax error in {path}: {exc}") from exc
    def _key(p: str) -> tuple[str, str]:
        filepart, name = p.rsplit(" ", 1)
        return (filepart.rsplit(":", 1)[0], name)  # strip ':line'

    unexpected = [p for p in all_problems if _key(p) not in _KNOWN_DEAD_CODE]
    assert not unexpected, (
        "undefined names (latent runtime NameErrors) found:\n" + "\n".join(unexpected[:20])
    )
