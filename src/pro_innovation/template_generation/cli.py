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
    description = parser.add_mutually_exclusive_group()
    description.add_argument("--description", help="authoritative product description")
    description.add_argument("--description-file", type=Path, help="UTF-8 product description file")


def run(args: argparse.Namespace) -> int:
    description = args.description_file if args.description_file is not None else args.description
    result = generate_templates(args.images, args.uts, description=description)
    print(json.dumps(result.as_dict(), ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Description + images + UTS -> CQT + NRT")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ProInnovationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
