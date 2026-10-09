"""Parser and serializer for the six fixed V1 rule templates."""

from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Iterable, Mapping

from .errors import TemplateError
from .models import AttributeTarget, Rule, RuleValue, Scalar, TaskType

_RULE_RE = re.compile(r"^\[(?P<rule_id>[A-Za-z][A-Za-z0-9_-]*)\]\s*(?P<body>.+)$")
_INDEXED_RE = re.compile(r"^(SUBJECT|PROPERTY|VALUE)_(\d+)$")
_RANGE_RE = re.compile(r"^\[\s*([^,\]]+)\s*,\s*([^,\]]+)\s*\]$")


def _parse_scalar(text: str) -> Scalar:
    value = text.strip()
    if not value:
        raise TemplateError("VALUE cannot be empty")
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value


def _format_scalar(value: Scalar) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise TemplateError("cannot serialize a missing value")
    return str(value)


def _parse_allowed_combinations(
    text: str, rule_id: str, attribute_count: int
) -> tuple[str, ...]:
    normalized = (
        text.strip()
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\uff1a", ":")
        .replace("\uff0c", ",")
    )
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise TemplateError(
            f"{rule_id}: allowed-combination VALUE must be a JSON string list"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise TemplateError(
            f"{rule_id}: allowed-combination VALUE must be a non-empty list"
        )

    combinations: list[str] = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            raise TemplateError(
                f"{rule_id}: every allowed combination must be a non-empty string"
            )
        labels = tuple(label.strip() for label in item.split(":"))
        if len(labels) != attribute_count or any(not label for label in labels):
            raise TemplateError(
                f"{rule_id}: combination {item!r} must contain "
                f"{attribute_count} colon-separated values"
            )
        combinations.append(":".join(labels))
    if len(set(combinations)) != len(combinations):
        raise TemplateError(f"{rule_id}: allowed combinations must be unique")
    return tuple(combinations)


def _format_rule_value(value: RuleValue) -> str:
    if isinstance(value, tuple):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
    return _format_scalar(value)


def _parse_count_values(text: str, rule_id: str, field_name: str) -> tuple[int, ...]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise TemplateError(
            f"{rule_id}: {field_name} must be a non-empty JSON array of "
            "non-negative integers"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise TemplateError(
            f"{rule_id}: {field_name} must be a non-empty JSON array of "
            "non-negative integers"
        )
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in parsed
    ):
        raise TemplateError(
            f"{rule_id}: {field_name} must contain only non-negative integers"
        )
    if len(set(parsed)) != len(parsed):
        raise TemplateError(f"{rule_id}: {field_name} count values must be unique")
    return tuple(parsed)


def _parse_range(text: str, rule_id: str) -> tuple[float, float]:
    match = _RANGE_RE.fullmatch(text.strip())
    if not match:
        raise TemplateError(
            f"{rule_id}: RANGE VALUE must use [lower,upper]"
        )
    parsed = tuple(_parse_scalar(item) for item in match.groups())
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in parsed
    ):
        raise TemplateError(f"{rule_id}: RANGE bounds must be numeric")
    minimum, maximum = (float(item) for item in parsed)
    if minimum < 0.0 or maximum < minimum:
        raise TemplateError(f"{rule_id}: RANGE requires 0 <= lower <= upper")
    return minimum, maximum


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _parse_fields(body: str, rule_id: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in body.split("|"):
        part = part.strip()
        if not part or "=" not in part:
            raise TemplateError(f"{rule_id}: malformed field {part!r}")
        key, value = (item.strip() for item in part.split("=", 1))
        if not key or not value:
            raise TemplateError(f"{rule_id}: field {key or '<empty>'} has no value")
        if key in fields:
            raise TemplateError(f"{rule_id}: duplicate field {key}")
        fields[key] = value
    return fields


def _require(fields: Mapping[str, str], key: str, rule_id: str) -> str:
    try:
        return fields[key]
    except KeyError as exc:
        raise TemplateError(f"{rule_id}: missing required field {key}") from exc


def _validate_unknown_fields(
    fields: Mapping[str, str], allowed: set[str], rule_id: str
) -> None:
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise TemplateError(f"{rule_id}: unsupported fields: {', '.join(unknown)}")


def _parse_attribute_combination(
    rule_id: str, fields: Mapping[str, str], allow_missing_values: bool
) -> Rule:
    indexed: dict[int, dict[str, str]] = {}
    allowed = {"TASK", "OP", "VALUE"}
    for key, value in fields.items():
        match = _INDEXED_RE.match(key)
        if not match:
            continue
        name, raw_index = match.groups()
        index = int(raw_index)
        if index < 1:
            raise TemplateError(f"{rule_id}: attribute indexes start at 1")
        indexed.setdefault(index, {})[name] = value
        allowed.add(key)

    _validate_unknown_fields(fields, allowed, rule_id)
    if not indexed:
        raise TemplateError(f"{rule_id}: ATTRIBUTE_COMBINATION needs indexed targets")
    expected_indexes = list(range(1, max(indexed) + 1))
    if sorted(indexed) != expected_indexes:
        raise TemplateError(f"{rule_id}: attribute indexes must be contiguous from 1")

    raw_total = fields.get("VALUE")
    allowed_combinations = (
        _parse_allowed_combinations(raw_total, rule_id, len(expected_indexes))
        if raw_total is not None and raw_total.lstrip().startswith("[")
        else None
    )

    attributes = []
    for index in expected_indexes:
        item = indexed[index]
        subject = _require(item, "SUBJECT", rule_id)
        property_name = _require(item, "PROPERTY", rule_id)
        if not property_name.startswith("attribute."):
            raise TemplateError(
                f"{rule_id}: PROPERTY_{index} must start with attribute."
            )
        expected = _parse_scalar(item["VALUE"]) if "VALUE" in item else None
        if allowed_combinations is not None and expected is not None:
            raise TemplateError(
                f"{rule_id}: allowed-combination VALUE cannot be mixed with VALUE_{index}"
            )
        if (
            allowed_combinations is None
            and expected is None
            and not allow_missing_values
        ):
            raise TemplateError(f"{rule_id}: missing required field VALUE_{index}")
        attributes.append(
            AttributeTarget(
                index=index,
                subject=subject,
                property=property_name,
                expected=expected,
            )
        )

    expected: RuleValue = (
        allowed_combinations
        if allowed_combinations is not None
        else _parse_scalar(fields["VALUE"])
        if "VALUE" in fields
        else None
    )
    if expected is None and not allow_missing_values:
        raise TemplateError(f"{rule_id}: missing required field VALUE")
    op = _require(fields, "OP", rule_id).upper()
    if op != "EQ":
        raise TemplateError(f"{rule_id}: V1 supports OP=EQ only")
    if (
        expected is not None
        and not isinstance(expected, tuple)
        and all(item.expected is not None for item in attributes)
    ):
        labels = tuple(_format_scalar(item.expected) for item in attributes)
        supported_combined_values = {"+".join(labels), ":".join(labels)}
        if expected not in supported_combined_values:
            raise TemplateError(
                f"{rule_id}: VALUE must equal the ordered VALUE_n combination; "
                f"choose one of {sorted(supported_combined_values)!r}"
            )
    return Rule(
        rule_id=rule_id,
        task=TaskType.ATTRIBUTE_COMBINATION,
        op=op,
        expected=expected,
        attributes=tuple(attributes),
    )


def parse_rule(line: str, *, allow_missing_values: bool = False) -> Rule:
    """Parse one canonical rule line.

    Generic templates set ``allow_missing_values=True`` because their VALUE and
    VALUE_n fields are intentionally absent.
    """

    match = _RULE_RE.match(line.strip())
    if not match:
        raise TemplateError(f"malformed rule line: {line!r}")
    rule_id = match.group("rule_id")
    fields = _parse_fields(match.group("body"), rule_id)
    raw_task = _require(fields, "TASK", rule_id).upper()
    try:
        task = TaskType(raw_task)
    except ValueError as exc:
        choices = ", ".join(item.value for item in TaskType)
        raise TemplateError(f"{rule_id}: unsupported TASK={raw_task}; choose {choices}") from exc

    if task is TaskType.ATTRIBUTE_COMBINATION:
        return _parse_attribute_combination(rule_id, fields, allow_missing_values)

    op = _require(fields, "OP", rule_id).upper()
    measurement_range = task in {TaskType.LENGTH, TaskType.AREA}
    allowed = {"TASK", "SUBJECT", "PROPERTY", "OP", "VALUE"}
    if measurement_range:
        allowed.add("COUNT")
    if task is TaskType.SPATIAL_COMBINATION:
        allowed.update({"OBJECT", "COUNT"})
    _validate_unknown_fields(fields, allowed, rule_id)

    subject = _require(fields, "SUBJECT", rule_id)
    property_name = _require(fields, "PROPERTY", rule_id)
    object_name = fields.get("OBJECT")
    expected: RuleValue = None
    minimum: float | None = None
    maximum: float | None = None
    count: tuple[int, ...] | None = None

    if measurement_range:
        if op != "RANGE":
            raise TemplateError(f"{rule_id}: {task.value} requires OP=RANGE")
        has_value = "VALUE" in fields
        has_count = "COUNT" in fields
        if has_value != has_count:
            raise TemplateError(f"{rule_id}: RANGE requires both VALUE and COUNT")
        if not has_value and not allow_missing_values:
            raise TemplateError(f"{rule_id}: missing required fields VALUE and COUNT")
        if has_value:
            minimum, maximum = _parse_range(fields["VALUE"], rule_id)
            count = _parse_count_values(fields["COUNT"], rule_id, "COUNT")
    else:
        allowed_ops = (
            {"EQ", "GE"}
            if task is TaskType.SPATIAL_COMBINATION
            else {"EQ"}
        )
        if op not in allowed_ops:
            supported = ", ".join(sorted(allowed_ops))
            raise TemplateError(
                f"{rule_id}: {task.value} supports OP={supported} only"
            )
        expected = (
            _parse_count_values(fields["VALUE"], rule_id, "COUNT VALUE")
            if task is TaskType.COUNT and "VALUE" in fields
            else _parse_scalar(fields["VALUE"])
            if "VALUE" in fields
            else None
        )
        if expected is None and not allow_missing_values:
            raise TemplateError(f"{rule_id}: missing required field VALUE")
        if task is TaskType.SPATIAL_COMBINATION and "COUNT" in fields:
            count = _parse_count_values(fields["COUNT"], rule_id, "COUNT")
        if (
            task is TaskType.SPATIAL_COMBINATION
            and op == "GE"
            and count is None
            and not allow_missing_values
        ):
            raise TemplateError(
                f"{rule_id}: SPATIAL_COMBINATION OP=GE requires COUNT"
            )

    if task is TaskType.COUNT and property_name != "count":
        raise TemplateError(f"{rule_id}: COUNT requires PROPERTY=count")
    if task is TaskType.LENGTH and property_name != "length":
        raise TemplateError(f"{rule_id}: LENGTH requires PROPERTY=length")
    if task is TaskType.AREA and property_name != "area":
        raise TemplateError(f"{rule_id}: AREA requires PROPERTY=area")
    if task is TaskType.SPATIAL_COMBINATION:
        if not property_name.startswith("spatial."):
            raise TemplateError(
                f"{rule_id}: SPATIAL_COMBINATION requires PROPERTY=spatial.<name>"
            )
        if not object_name:
            raise TemplateError(f"{rule_id}: SPATIAL_COMBINATION requires OBJECT")
    if task is TaskType.ATTRIBUTE_ERROR and not property_name.startswith("attribute."):
        raise TemplateError(
            f"{rule_id}: ATTRIBUTE_ERROR requires PROPERTY=attribute.<name>"
        )

    return Rule(
        rule_id=rule_id,
        task=task,
        subject=subject,
        property=property_name,
        object=object_name,
        op=op,
        expected=expected,
        minimum=minimum,
        maximum=maximum,
        count=count,
    )


def parse_template(text: str, *, allow_missing_values: bool = False) -> tuple[Rule, ...]:
    rules = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rule = parse_rule(line, allow_missing_values=allow_missing_values)
        except TemplateError as exc:
            raise TemplateError(f"line {line_number}: {exc}") from exc
        if rule.rule_id in seen:
            raise TemplateError(f"line {line_number}: duplicate rule id {rule.rule_id}")
        seen.add(rule.rule_id)
        rules.append(rule)
    if not rules:
        raise TemplateError("template contains no rules")
    return tuple(rules)


def serialize_rule(rule: Rule, *, include_values: bool = True) -> str:
    fields = [f"TASK={rule.task.value}"]
    if rule.task is TaskType.ATTRIBUTE_COMBINATION:
        for item in rule.attributes:
            fields.extend(
                [
                    f"SUBJECT_{item.index}={item.subject}",
                    f"PROPERTY_{item.index}={item.property}",
                ]
            )
            if include_values and item.expected is not None:
                fields.append(f"VALUE_{item.index}={_format_scalar(item.expected)}")
    else:
        fields.extend([f"SUBJECT={rule.subject}", f"PROPERTY={rule.property}"])
        if rule.object is not None:
            fields.append(f"OBJECT={rule.object}")
    fields.append(f"OP={rule.op}")
    if include_values:
        if (
            rule.op == "RANGE"
            and rule.minimum is not None
            and rule.maximum is not None
            and rule.count is not None
        ):
            fields.extend(
                [
                    f"VALUE=[{_format_number(rule.minimum)},{_format_number(rule.maximum)}]",
                    f"COUNT={_format_rule_value(rule.count)}",
                ]
            )
        elif rule.expected is not None:
            fields.append(f"VALUE={_format_rule_value(rule.expected)}")
            if (
                rule.task is TaskType.SPATIAL_COMBINATION
                and rule.count is not None
            ):
                fields.append(f"COUNT={_format_rule_value(rule.count)}")
    return f"[{rule.rule_id}] " + " | ".join(fields)


def serialize_template(rules: Iterable[Rule], *, include_values: bool = True) -> str:
    return "\n".join(
        serialize_rule(rule, include_values=include_values) for rule in rules
    )


def to_generic_template(rules: Iterable[Rule]) -> tuple[Rule, ...]:
    generic = []
    for rule in rules:
        attributes = tuple(replace(item, expected=None) for item in rule.attributes)
        generic.append(
            replace(
                rule,
                expected=None,
                minimum=None,
                maximum=None,
                count=None,
                attributes=attributes,
            )
        )
    return tuple(generic)


def collect_categories(rules: Iterable[Rule]) -> tuple[str, ...]:
    """Collect category fields only, preserving first-seen order."""

    categories: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        for category in rule.categories:
            if category not in seen:
                seen.add(category)
                categories.append(category)
    return tuple(categories)


def validate_generic_values_missing(rules: Iterable[Rule]) -> tuple[Rule, ...]:
    generic = tuple(rules)
    for rule in generic:
        if (
            rule.expected is not None
            or rule.minimum is not None
            or rule.maximum is not None
            or rule.count is not None
            or any(
                item.expected is not None for item in rule.attributes
            )
        ):
            raise TemplateError(
                f"{rule.rule_id}: generic templates must not contain expected values"
            )
    return generic


def validate_template_alignment(
    generic_rules: Iterable[Rule], standard_rules: Iterable[Rule]
) -> tuple[tuple[Rule, Rule], ...]:
    generic = validate_generic_values_missing(generic_rules)
    standard = tuple(standard_rules)
    if [item.rule_id for item in generic] != [item.rule_id for item in standard]:
        raise TemplateError("generic and standard templates must contain the same ordered rule ids")

    pairs = []
    for observed, expected in zip(generic, standard):
        left = replace(
            observed, expected=None, minimum=None, maximum=None, count=None
        )
        right = replace(
            expected, expected=None, minimum=None, maximum=None, count=None
        )
        left = replace(
            left,
            attributes=tuple(replace(item, expected=None) for item in left.attributes),
        )
        right = replace(
            right,
            attributes=tuple(replace(item, expected=None) for item in right.attributes),
        )
        if left != right:
            raise TemplateError(
                f"{observed.rule_id}: generic and standard rule structures differ"
            )
        pairs.append((observed, expected))
    return tuple(pairs)
