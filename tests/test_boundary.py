"""The split between the model and the code, enforced.

app/core is deterministic: it never imports app/model, so no decision can
depend on a model call. app/model may read core's types to build prompts, but
never the code that decides (the procedure, the verifier, the guard).
app/agent.py is the only module that imports both.
"""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
DECIDES = {"procedure", "verify", "guard"}


def _imports(path: Path) -> set[str]:
    """Absolute module names a file imports, with relative imports resolved."""
    pkg = path.relative_to(APP.parent).with_suffix("").parts[:-1]
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = list(pkg[: len(pkg) - node.level + 1]) if node.level else []
            mod = ".".join(base + ([node.module] if node.module else []))
            out.add(mod)
            out |= {f"{mod}.{a.name}" for a in node.names}
    return out


def _files(sub: str) -> list[Path]:
    return sorted((APP / sub).rglob("*.py"))


def test_core_never_imports_the_model():
    for f in _files("core"):
        bad = {m for m in _imports(f) if m == "app.model" or m.startswith("app.model.") or m == "app.agent"}
        assert not bad, f"{f.relative_to(APP)} imports {bad}"


def test_model_never_imports_the_code_that_decides():
    for f in _files("model"):
        bad = {m for m in _imports(f) if m.startswith("app.core.") and m.split(".")[2] in DECIDES}
        assert not bad, f"{f.relative_to(APP)} imports {bad}"


def test_only_the_agent_joins_both_sides():
    for f in APP.glob("*.py"):
        mods = _imports(f)
        both = any(m.startswith("app.core") for m in mods) and any(m.startswith("app.model") for m in mods)
        if both:
            assert f.name in {"agent.py", "server.py"}, f"{f.name} joins core and model"


def test_the_boundary_check_sees_relative_imports():
    imports = _imports(APP / "core" / "procedure.py")
    assert "app.core.verify" in imports and "app.config" in imports
