"""Command-line template utilities for the Rule-LOCO source release."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
from .templates import parse_template, serialize_template, to_generic_template


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rule-LOCO template utilities; use scripts/global_patch_ordered for GPO-JM")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a rule template")
    validate.add_argument("template")
    validate.add_argument("--allow-missing-values", action="store_true")
    generic = commands.add_parser("make-generic", help="derive a generic template from standard rules")
    generic.add_argument("template")
    generic.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rules = parse_template(Path(args.template).read_text(encoding="utf-8"), allow_missing_values=getattr(args, "allow_missing_values", False))
    if args.command == "make-generic":
        Path(args.output).write_text(serialize_template(to_generic_template(rules), include_values=False), encoding="utf-8")
    else:
        print(f"Valid template: {len(rules)} rules")
    return 0
