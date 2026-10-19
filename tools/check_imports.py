"""
tools/check_imports.py
----------------------
Static import-resolution check. Catches missing local modules without
importing anything and without needing the project environment.

Needs only the standard library. Run it before any pipeline attempt:

    python -X utf8 tools/check_imports.py

Exit code 0 = every local import resolves.
Exit code 1 = at least one import cannot be resolved (see the printed list).

Classification:
  LOCAL ok      - a project module/package that exists on disk
  LOCAL MISSING - a project module/package that does NOT exist  <- real defect
  NOT INSTALLED - a third-party package absent from the *current* interpreter.
                  This is an ENVIRONMENT fact, not a code defect: compare it
                  against pyproject.toml / uv.lock before drawing a conclusion.

Known standing result (2026-10-18): train.py imports `data.roco_dataset`,
which does not exist. See docs/KNOWN_ISSUES.md AFR-001.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

SKIP_DIRS = {".venv", ".git", "biomedclip", "__pycache__", "docs"}


def declared_dependencies(root: pathlib.Path) -> set[str]:
    """Top-level distribution names named in pyproject.toml, best effort.

    Used only to label a missing third-party import as 'declared' or not; it is
    a text scan, not a dependency resolver.
    """
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return set()
    names: set[str] = set()
    in_deps = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("]"):
                in_deps = False
                continue
            token = stripped.strip('",').split("[")[0]
            for sep in ("==", ">=", "<=", "~=", ">", "<"):
                token = token.split(sep)[0]
            if token:
                names.add(token.strip().lower().replace("-", "_"))
    return names


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    packages = {
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "__init__.py").is_file()
    }
    top_modules = {p.stem for p in root.glob("*.py")}
    declared = declared_dependencies(root)
    stdlib = set(sys.stdlib_module_names)

    missing_local: list[tuple[str, int, str]] = []
    not_installed: list[tuple[str, int, str]] = []
    rows: list[str] = []

    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            rows.append(f"  [SYNTAX ERROR        ] {rel}: {exc}")
            missing_local.append((str(rel), getattr(exc, "lineno", 0) or 0, "<syntax>"))
            continue

        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]

            for module in modules:
                head = module.split(".")[0]
                if head in stdlib:
                    continue

                target = root.joinpath(*module.split("."))
                on_disk = (
                    target.with_suffix(".py").is_file()
                    or (target / "__init__.py").is_file()
                )

                if head in packages or head in top_modules:
                    kind = "LOCAL ok" if on_disk else "LOCAL MISSING"
                elif on_disk:
                    kind = "LOCAL ok"
                else:
                    try:
                        installed = importlib.util.find_spec(head) is not None
                    except (ImportError, ValueError):
                        installed = False
                    if installed:
                        continue
                    if head.lower() in declared:
                        # A declared third-party package that this interpreter
                        # lacks: an environment gap, not a code defect.
                        kind = "NOT INSTALLED (declared)"
                    else:
                        # Not stdlib, not installed, not on disk and not
                        # declared as a dependency -> it can only be a project
                        # module that was never written. This is the case that
                        # must fail the check (e.g. train.py -> data.roco_dataset).
                        kind = "LOCAL MISSING"

                rows.append(f"  [{kind:22s}] {str(rel):30s}:{node.lineno:<4d} {module}")
                if kind == "LOCAL MISSING":
                    missing_local.append((str(rel), node.lineno, module))
                elif kind.startswith("NOT INSTALLED"):
                    not_installed.append((str(rel), node.lineno, module))

    print("=== project imports needing attention ===")
    print("\n".join(rows) if rows else "  (none)")

    if not_installed:
        print("\nNOT INSTALLED in this interpreter (environment, not code):")
        for file, line, module in not_installed:
            print(f"  {file}:{line}  ->  {module}")
        print("  If these are declared in pyproject.toml, run `uv sync` and re-check.")

    if missing_local:
        print("\n### MISSING LOCAL MODULES (real defects) ###")
        for file, line, module in missing_local:
            print(f"  {file}:{line}  ->  {module}")
        return 1

    print("\nAll local imports resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
