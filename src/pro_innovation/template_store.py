"""Validated persistence for class standard and generic templates."""

from __future__ import annotations

from pathlib import Path
import re

from .errors import TemplateError
from .models import Rule
from .templates import parse_template, serialize_template, to_generic_template

_CLASS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class TemplateRepository:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _validate_class_name(class_name: str) -> None:
        if not _CLASS_RE.fullmatch(class_name):
            raise TemplateError(
                "class name may contain letters, numbers, underscores, and hyphens only"
            )

    def save_standard(
        self, class_name: str, template_text: str, *, overwrite: bool = False
    ) -> tuple[Path, Path]:
        self._validate_class_name(class_name)
        standard = parse_template(template_text, allow_missing_values=False)
        generic = to_generic_template(standard)
        self.root.mkdir(parents=True, exist_ok=True)
        standard_path = self.root / f"{class_name}.standard.rules"
        generic_path = self.root / f"{class_name}.generic.rules"
        existing = [path for path in (standard_path, generic_path) if path.exists()]
        if existing and not overwrite:
            raise TemplateError(
                "refusing to overwrite existing templates: "
                + ", ".join(str(path) for path in existing)
            )
        standard_path.write_text(
            serialize_template(standard) + "\n", encoding="utf-8"
        )
        generic_path.write_text(
            serialize_template(generic) + "\n", encoding="utf-8"
        )
        return standard_path, generic_path

    def load(self, class_name: str) -> tuple[tuple[Rule, ...], tuple[Rule, ...]]:
        self._validate_class_name(class_name)
        standard_path = self.root / f"{class_name}.standard.rules"
        generic_path = self.root / f"{class_name}.generic.rules"
        standard = parse_template(
            standard_path.read_text(encoding="utf-8"), allow_missing_values=False
        )
        generic = parse_template(
            generic_path.read_text(encoding="utf-8"), allow_missing_values=True
        )
        return generic, standard

