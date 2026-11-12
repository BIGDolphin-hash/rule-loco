"""CLI shared by the standalone module and the project's main entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..errors import ProInnovationError
from .generator import generate_templates


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images", nargs="+", required=True, help="normal reference image paths")
    parser.add_argument("--uts", type=Path, required=True, help="UTS JSON file")


def run(args: argparse.Namespace) -> int:
    result = generate_templates(args.images, args.uts)
    print(json.dumps(result.as_dict(), ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Images + UTS -> CQT + NRT using the configured VLM")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ProInnovationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
