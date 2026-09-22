#!/usr/bin/env python3
"""Deterministic dependency-claim checks.

Evidence-matrix row C014: the README "Built With" list must name dependencies
the repo actually pins, and every top-level third-party import in
``icl_annotation_solver.py`` must be covered by ``requirements.txt``.
Stdlib-only, no network. Exit 0 iff both directions hold.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE = REPO_ROOT / "icl_annotation_solver.py"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# Distribution names the README's "Built With" list presents as core, pinned
# dependencies (C014).
BUILT_WITH_REQUIREMENTS = [
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "cubiczan-resilience",
]

# Import module name -> distribution name, where the two differ.
MODULE_TO_DIST = {"cubiczan_resilience": "cubiczan-resilience"}


def declared_requirements(text: str) -> set:
    """Lowercased distribution names from a requirements.txt."""
    names = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(" @")[0]  # drop PEP 508 URL (incl. '#subdirectory=')
        name = name.split(";")[0]  # drop environment markers
        for op in (">", "<", "=", "!", "[", " "):
            name = name.split(op)[0]
        if name.strip():
            names.add(name.strip().lower())
    return names


def third_party_imports(path: Path) -> set:
    """Top-level, non-stdlib module names imported by the engine file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods = set()
    for node in tree.body:  # top-level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    return {m for m in mods if m not in stdlib}


def main() -> int:
    failures = []
    declared = declared_requirements(REQUIREMENTS.read_text(encoding="utf-8"))

    for dist in BUILT_WITH_REQUIREMENTS:
        if dist not in declared:
            failures.append(f'"Built With" names {dist!r} but requirements.txt does not pin it')

    imported = third_party_imports(ENGINE)
    for module in sorted(imported):
        dist = MODULE_TO_DIST.get(module, module.replace("_", "-")).lower()
        if dist not in declared:
            failures.append(
                f"icl_annotation_solver.py imports {module!r} but requirements.txt "
                f"has no {dist!r} entry"
            )

    if failures:
        print("DECLARED DEPENDENCIES: FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"DECLARED DEPENDENCIES: OK "
        f"({len(BUILT_WITH_REQUIREMENTS)} Built With deps pinned, "
        f"{len(imported)} third-party engine imports covered)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
