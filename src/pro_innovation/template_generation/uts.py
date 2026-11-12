"""Machine-readable, category-agnostic UTS and generated-rule validation."""

from __future__ import annotations

import json
from math import hypot, isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..attributes import validate_color_candidates
from ..errors import TemplateError
from ..models import Rule, TaskType
from ..templates import serialize_rule

SPATIAL_RELATIONS = frozenset({
    "left_of", "right_of", "above", "below", "upper_left_of", "upper_right_of",
    "lower_left_of", "lower_right_of", "inside", "contains", "at_left",
    "at_right", "at_top", "at_lower", "at_center", "at_middle",
    "self_left_right_symmetric", "self_top_bottom_symmetric",
})
_PROPERTIES = {
    "COUNT": {"count"}, "LENGTH": {"length"}, "AREA": {"area"},
    "SPATIAL_COMBINATION": {"spatial.position", "spatial.self_symmetry"},
    "ATTRIBUTE_COMBINATION": {
        "attribute.count", "attribute.color", "attribute.shape", "attribute.type",
        "attribute.material", "attribute.texture", "attribute.state",
    },
}
_OPERATORS = {
    "COUNT": {"EQ"}, "LENGTH": {"RANGE"}, "AREA": {"RANGE"},
    "SPATIAL_COMBINATION": {"EQ", "GE"}, "ATTRIBUTE_COMBINATION": {"EQ"},
}

UTSInput = str | Path | Mapping[str, Any]


def _string_list(value: Any, field: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(set(value)) != len(value)):
        raise TemplateError(f"UTS {field} must be a nonempty list of unique strings")
    return value


def load_uts(value: UTSInput) -> dict[str, Any]:
    """Accept a JSON mapping, JSON text, or a Path to a JSON UTS file."""
    try:
        if isinstance(value, Path):
            value = value.read_text(encoding="utf-8")
        data = json.loads(value) if isinstance(value, str) else json.loads(
            json.dumps(dict(value), allow_nan=False)
        )
    except (TypeError, ValueError, OSError) as exc:
        raise TemplateError("UTS must be valid JSON or a readable JSON Path") from exc
    if not isinstance(data, dict) or set(data) != {"version", "tasks"}:
        raise TemplateError("UTS must contain exactly version and tasks")
    if type(data["version"]) is not int or data["version"] != 1:
        raise TemplateError("UTS version must be 1")
    if not isinstance(data["tasks"], list) or not data["tasks"]:
        raise TemplateError("UTS tasks must be a nonempty list")
    seen = set()
    for spec in data["tasks"]:
        if not isinstance(spec, dict) or not {"task", "properties", "operators"} <= set(spec):
            raise TemplateError("each UTS task requires task, properties and operators")
        task = spec["task"]
        if not isinstance(task, str) or task not in _PROPERTIES or task in seen:
            raise TemplateError("UTS tasks must be supported and unique")
        seen.add(task)
        allowed_keys = {"task", "properties", "operators"}
        if task in {"LENGTH", "AREA"}:
            allowed_keys.add("unit")
            unit = "pixel" if task == "LENGTH" else "pixel_squared"
            if spec.get("unit") != unit:
                raise TemplateError(f"UTS {task} requires unit={unit}")
        if task == "SPATIAL_COMBINATION":
            allowed_keys.add("relations")
            if not set(_string_list(spec.get("relations"), "relations")) <= SPATIAL_RELATIONS:
                raise TemplateError("UTS contains an unsupported spatial relation")
        if set(spec) - allowed_keys:
            raise TemplateError(f"UTS {task} contains unsupported fields")
        if not set(_string_list(spec["properties"], "properties")) <= _PROPERTIES[task]:
            raise TemplateError(f"UTS {task} contains unsupported properties")
        if not set(_string_list(spec["operators"], "operators")) <= _OPERATORS[task]:
            raise TemplateError(f"UTS {task} contains unsupported operators")
    return data


def validate_generated_rules(
    rules: Sequence[Rule], uts: Mapping[str, Any], dimensions: Sequence[tuple[int, int]]
) -> None:
    """Check executable semantics beyond what the existing text parser checks."""
    specs = {item["task"]: item for item in uts["tasks"]}
    seen = set()
    counted_subjects: dict[str, tuple[int, ...]] = {}
    for rule in rules:
        if rule.task is TaskType.COUNT:
            if rule.subject in counted_subjects:
                raise TemplateError(f"{rule.rule_id}: use one COUNT rule per subject")
            counted_subjects[rule.subject] = rule.expected
    for index, rule in enumerate(rules, 1):
        prefix = f"{rule.rule_id}: "
        if rule.rule_id != f"C{index:03d}":
            raise TemplateError("generated rule IDs must be consecutive from C001")
        spec = specs.get(rule.task.value)
        if spec is None:
            raise TemplateError(prefix + "task is not permitted by UTS")
        properties = [item.property for item in rule.attributes] if rule.attributes else [rule.property]
        if rule.op not in spec["operators"] or not set(properties) <= set(spec["properties"]):
            raise TemplateError(prefix + "operator or property is not permitted by UTS")
        for name in rule.categories:
            if not name.strip() or name != name.strip() or re.search(r"[|=\[\]\r\n]", name):
                raise TemplateError(prefix + "object names contain invalid delimiters")
        signature = serialize_rule(rule).split("] ", 1)[1]
        if signature in seen:
            raise TemplateError(prefix + "duplicate constraint")
        seen.add(signature)
        if rule.task in {TaskType.LENGTH, TaskType.AREA}:
            if len(set(dimensions)) != 1:
                raise TemplateError(prefix + "pixel measurement rules require equal image dimensions")
            width, height = dimensions[0]
            maximum = hypot(width - 1, height - 1) + 1 if rule.task is TaskType.LENGTH else width * height
            if (rule.minimum is None or rule.maximum is None
                    or not isfinite(rule.minimum) or not isfinite(rule.maximum)
                    or rule.maximum > maximum or rule.maximum <= 0):
                raise TemplateError(prefix + "measurement bounds exceed native image dimensions or are non-finite")
        if rule.count is not None and rule.subject in counted_subjects:
            if max(rule.count) > max(counted_subjects[rule.subject]):
                raise TemplateError(prefix + "matching COUNT exceeds the subject's total count")
        if rule.task is TaskType.SPATIAL_COMBINATION:
            if rule.expected not in spec["relations"]:
                raise TemplateError(prefix + "spatial relation is not permitted by UTS")
            self_relation = str(rule.expected).startswith("self_")
            if self_relation != (rule.property == "spatial.self_symmetry"):
                raise TemplateError(prefix + "self-symmetry relation/property disagree")
            if self_relation and rule.object is not None:
                raise TemplateError(prefix + "self-symmetry must not have OBJECT")
            if not self_relation and rule.subject == rule.object:
                raise TemplateError(prefix + "spatial subject and object must differ")
            if rule.count is None:
                raise TemplateError(prefix + "generated spatial rules require COUNT")
        if rule.task is TaskType.ATTRIBUTE_COMBINATION:
            targets = [(item.subject, item.property) for item in rule.attributes]
            if len(targets) < 2 or len(set(targets)) != len(targets):
                raise TemplateError(prefix + "attribute combinations require at least two distinct targets")
            if (not isinstance(rule.expected, tuple)
                    or not all(isinstance(item, str) for item in rule.expected)
                    or any(item.expected is not None for item in rule.attributes)):
                raise TemplateError(prefix + "use allowed tuple VALUE list, without VALUE_n")
            for combination in rule.expected:
                for target, label in zip(rule.attributes, combination.split(":")):
                    if target.property == "attribute.count":
                        if not label.isascii() or not label.isdigit():
                            raise TemplateError(prefix + "attribute.count labels must be nonnegative integer strings")
                        totals = counted_subjects.get(target.subject)
                        if totals is not None and int(label) not in totals:
                            raise TemplateError(prefix + "attribute.count conflicts with the subject's COUNT rule")
                    if target.property == "attribute.color":
                        try:
                            validate_color_candidates([label])
                        except ValueError as exc:
                            raise TemplateError(prefix + str(exc)) from exc
                        if label != label.lower():
                            raise TemplateError(prefix + "color labels must be lowercase")
                    if target.property == "attribute.shape" and label not in {"square", "round", "elongated"}:
                        raise TemplateError(prefix + "unsupported coarse shape label")
