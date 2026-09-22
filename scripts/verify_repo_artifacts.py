#!/usr/bin/env python3
"""Deterministic artifact checks for README-stated repository content.

Evidence-matrix rows C011/C012: every file the README's "Repository Structure"
block and "Demo Video" section name must exist in the committed tree.
Stdlib-only, no network. Exit 0 iff every referenced path exists.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths the README's "Repository Structure" block lists (must match the block).
README_STRUCTURE_FILES = [
    "icl_annotation_solver.py",
    "generate_report.py",
    "kaggle_notebook.ipynb",
    "tests/test_engine.py",
    "scripts",
    "evidence/matrix.yaml",
    "tools/verify_evidence_matrix.py",
    "assets/demo.mp4",
    "demo-video.mp4",
    "logo.png",
    "Technical_Report.pdf",
    "README.md",
]

# Media the README's "Demo Video" section references.
DEMO_MEDIA = ["demo-video.mp4", "assets/demo.mp4"]


def main() -> int:
    failures = []
    for rel in README_STRUCTURE_FILES:
        if not (REPO_ROOT / rel).exists():
            failures.append(f"missing: {rel}")
    for rel in DEMO_MEDIA:
        path = REPO_ROOT / rel
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"demo media missing or empty: {rel}")

    if failures:
        print("REPO ARTIFACTS: FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"REPO ARTIFACTS: OK "
        f"({len(README_STRUCTURE_FILES)} structure entries, "
        f"{len(DEMO_MEDIA)} demo media present)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
