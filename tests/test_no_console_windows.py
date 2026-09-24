"""Static guard: no direct subprocess.run/Popen or os.system/os.popen under aive/.

Regression for the console-window flood bug: the app is a GUI-subsystem build
(``console=False`` in packaging/aive.spec), so on Windows every child *console*
program (ffmpeg/ffprobe/nvidia-smi/python) allocates its own visible console —
dozens of windows per job. All spawns must therefore go through
``aive.stages.ingest.run_hidden`` / ``popen_hidden``, which OR in
``subprocess.CREATE_NO_WINDOW`` on Windows only (the flag is rejected on POSIX,
so the helper is a no-op elsewhere).

The guard walks every .py under aive/ and flags:

* direct ``subprocess.run(...)`` / ``subprocess.Popen(...)`` calls — except
  inside the two wrapper definitions themselves in stages/ingest.py;
* ``os.system(...)`` / ``os.popen(...)`` anywhere (they always pop a window).

Dependency-free AST walk, self-tested below before being applied to the package
(same pattern as test_no_undefined_names.py).
"""
from __future__ import annotations

import ast
import subprocess as _sp
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG_DIR = REPO / "aive"

# The only legal home for direct subprocess.run/Popen: the wrapper definitions.
_ALLOWED_WRAPPERS = {"aive/stages/ingest.py": {"run_hidden", "popen_hidden"}}


def _spawn_name(func: ast.expr) -> str | None:
    """Return 'subprocess.<attr>' / 'os.<attr>' if `func` is a direct spawn, else None."""
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = func.value.id
        if base == "subprocess" and func.attr in ("run", "Popen"):
            return f"subprocess.{func.attr}"
        if base == "os" and func.attr in ("system", "popen"):
            return f"os.{func.attr}"
    return None


def check_source(
    src: str,
    filename: str = "<src>",
    allowed_wrappers: set[str] | None = None,
) -> list[str]:
    """Flag direct spawn calls in `src`.

    ``allowed_wrappers`` names function definitions whose bodies may contain
    direct subprocess.run/Popen (used for the ingest.py wrappers). os.system /
    os.popen are never allowed.
    """
    tree = ast.parse(src)
    problems: list[str] = []
    allowed_lines: set[int] = set()
    if allowed_wrappers:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed_wrappers:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and _spawn_name(sub.func) is not None:
                        allowed_lines.add(sub.lineno)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        what = _spawn_name(node.func)
        if what is None or (what.startswith("subprocess.") and node.lineno in allowed_lines):
            continue
        hint = (
            "use aive.stages.ingest.run_hidden/popen_hidden"
            if what.startswith("subprocess.")
            else "it always pops a console window on Windows"
        )
        problems.append(f"{filename}:{node.lineno} {what}(...) — {hint}")
    return problems


def _violations_in_package() -> list[str]:
    out: list[str] = []
    for path in sorted(PKG_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(REPO)).replace("\\", "/")
        src = path.read_text(encoding="utf-8")
        out += check_source(src, filename=rel, allowed_wrappers=_ALLOWED_WRAPPERS.get(rel))
    return out


# ---------------------------------------------------------------------------
# Self-tests: the checker must catch violations and pass clean code.
# ---------------------------------------------------------------------------

def test_checker_flags_direct_subprocess_run():
    src = "import subprocess\nsubprocess.run(['echo', 'hi'])\n"
    problems = check_source(src)
    assert any("subprocess.run" in p for p in problems), f"expected subprocess.run flagged, got {problems}"


def test_checker_flags_os_system_and_popen():
    src = "import os\nos.system('dir')\nos.popen('dir')\n"
    problems = check_source(src)
    assert sum(1 for p in problems if "os.system" in p or "os.popen" in p) == 2, problems


def test_checker_allows_only_wrapper_bodies():
    src = (
        "import subprocess\n"
        "\n"
        "def run_hidden(cmd, **kw):\n"
        "    return subprocess.run(cmd, **kw)\n"
        "\n"
        "def bad(cmd):\n"
        "    return subprocess.Popen(cmd)\n"
    )
    problems = check_source(src, allowed_wrappers={"run_hidden"})
    assert any("subprocess.Popen" in p for p in problems), f"non-wrapper call must be flagged: {problems}"
    assert not any("subprocess.run" in p for p in problems), f"wrapper's own call must pass: {problems}"


def test_checker_accepts_wrapper_usage():
    src = (
        "from aive.stages.ingest import run_hidden\n"
        "\n"
        "def spawn(cmd):\n"
        "    return run_hidden(cmd, capture_output=True)\n"
    )
    assert check_source(src) == []


# ---------------------------------------------------------------------------
# _merge_hidden platform behavior (unit).
# ---------------------------------------------------------------------------

from aive.stages import ingest  # noqa: E402 - kept after the guard so a broken package fails loudly first

_HAS_CNW = hasattr(_sp, "CREATE_NO_WINDOW")


@pytest.mark.skipif(not _HAS_CNW, reason="subprocess.CREATE_NO_WINDOW is Windows-only")
def test_merge_hidden_sets_create_no_window_on_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    kw: dict = {}
    ingest._merge_hidden(kw)
    assert kw["creationflags"] == _sp.CREATE_NO_WINDOW


@pytest.mark.skipif(not _HAS_CNW, reason="subprocess.CREATE_NO_WINDOW is Windows-only")
def test_merge_hidden_preserves_existing_flags_via_or(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    existing = 0x10000000  # e.g. CREATE_UNICODE_ENVIRONMENT — must survive the OR
    kw = {"creationflags": existing}
    ingest._merge_hidden(kw)
    assert kw["creationflags"] == (existing | _sp.CREATE_NO_WINDOW)


def test_merge_hidden_noop_on_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    kw: dict = {}
    ingest._merge_hidden(kw)
    assert "creationflags" not in kw  # creationflags is rejected on POSIX


# ---------------------------------------------------------------------------
# End-to-end wrapper round-trips (cheap: two python -c spawns).
# ---------------------------------------------------------------------------

def test_run_hidden_roundtrip():
    out = ingest.run_hidden([sys.executable, "-c", "print('hi')"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "hi"


def test_popen_hidden_roundtrip():
    proc = ingest.popen_hidden([sys.executable, "-c", "pass"])
    assert proc.wait(timeout=30) == 0


# ---------------------------------------------------------------------------
# Package-wide guard.
# ---------------------------------------------------------------------------

def test_no_direct_spawns_in_aive():
    files = sorted(PKG_DIR.rglob("*.py"))
    assert files, f"no python files found under {PKG_DIR}"
    problems = _violations_in_package()
    assert not problems, "direct console-spawning calls found:\n" + "\n".join(problems[:20])
