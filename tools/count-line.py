#!/usr/bin/env python3
"""
Print a tree of the project and count lines of C++ code (.cpp, .hpp),
excluding comments and empty lines.

Usage
-----
    python count_loc_tree.py           # tree for current directory (.)
    python count_loc_tree.py path/to/project
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple


# ------------------------------------------------------------
# Comment stripping
# ------------------------------------------------------------

def strip_comments(line: str, in_block_comment: bool) -> Tuple[str, bool]:
    """
    Remove // and /* */ comments from a single line.

    - Keeps code that appears before // or outside /* */
    - Tracks whether we are inside a multi-line comment.

    Returns (code_without_comments, updated_in_block_comment).
    """
    i = 0
    n = len(line)
    result_chars = []

    while i < n:
        if not in_block_comment:
            # line comment //
            if i + 1 < n and line[i] == "/" and line[i + 1] == "/":
                # rest of line is comment
                break
            # start block comment /*
            elif i + 1 < n and line[i] == "/" and line[i + 1] == "*":
                in_block_comment = True
                i += 2
                continue
            else:
                result_chars.append(line[i])
                i += 1
        else:
            # inside block comment, look for */
            if i + 1 < n and line[i] == "*" and line[i + 1] == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1

    return "".join(result_chars), in_block_comment


# ------------------------------------------------------------
# LOC counting
# ------------------------------------------------------------

def count_loc_in_file(path: Path) -> int:
    """
    Count lines of code in a single .cpp/.hpp file.

    - Excludes empty lines and pure comments.
    - Counts lines that still contain code after removing comments.
    """
    loc = 0
    in_block_comment = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped_line, in_block_comment = strip_comments(line, in_block_comment)
            if stripped_line.strip():  # non-empty after removing comments
                loc += 1

    return loc


# ------------------------------------------------------------
# Tree printing
# ------------------------------------------------------------

def print_tree_with_loc(root: Path) -> int:
    """
    Print a tree-like view starting at 'root', annotating .cpp/.hpp files
    with their LOC. Returns the total LOC over all .cpp/.hpp files.
    """
    total_loc = 0
    extensions = {".cpp", ".hpp"}

    # Directories we might want to skip from the tree (optional)
    exclude_dirs = {".git", "__pycache__", ".vscode", ".idea"}

    def _walk(dir_path: Path, prefix: str) -> None:
        nonlocal total_loc

        # List entries: dirs first, then files, alphabetically
        entries = sorted(
            dir_path.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower())
        )

        # Filter out unwanted directories
        filtered = []
        for e in entries:
            if e.is_dir() and e.name in exclude_dirs:
                continue
            filtered.append(e)
        entries = filtered

        for idx, entry in enumerate(entries):
            is_last = (idx == len(entries) - 1)
            connector = "└── " if is_last else "├── "

            if entry.is_dir():
                print(f"{prefix}{connector}{entry.name}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                _walk(entry, new_prefix)
            else:
                if entry.suffix in extensions:
                    loc = count_loc_in_file(entry)
                    total_loc += loc
                    print(f"{prefix}{connector}{entry.name}  // {loc} LOC")
                else:
                    # Non-C++ files are printed, but without LOC
                    print(f"{prefix}{connector}{entry.name}")

    # Print root name similar to `tree .`
    print(root.name if root.name else str(root))
    _walk(root, "")

    return total_loc


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print project tree and count C++ LOC (.cpp, .hpp), excluding comments and empty lines."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.path)
    root = input_path.resolve()

    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    # For nicer header, show "." when scanning current directory
    header = "." if args.path in (".", "./") else str(root)
    print(header)

    total_loc = print_tree_with_loc(root)

    print("\n---------------------------------------------")
    print(f"Total C++ LOC (.cpp/.hpp, no comments/blank): {total_loc}")
    print("---------------------------------------------")


if __name__ == "__main__":
    main()
