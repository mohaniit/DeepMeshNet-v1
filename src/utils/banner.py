"""Project banner utilities for DeepMeshNet-v1."""

from __future__ import annotations


PROJECT_NAME = "DeepMeshNet-v1"
VERSION = "1.0.0"
TAGLINE = "Engineering Knowledge First. AI Second."


def print_banner(module_name: str | None = None) -> None:
    """Print a standard DeepMeshNet-v1 console banner."""
    line = "=" * 72
    print(line)
    print(f"{PROJECT_NAME} {VERSION}")
    print(TAGLINE)
    if module_name:
        print("-" * 72)
        print(module_name)
    print(line)