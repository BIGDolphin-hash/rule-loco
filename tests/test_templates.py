from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pro_innovation.errors import TemplateError
from pro_innovation.template_store import TemplateRepository
from pro_innovation.templates import (
    collect_categories,
    parse_template,
    serialize_template,
    to_generic_template,
    validate_generic_values_missing,
)

ROOT = Path(__file__).resolve().parents[1]


class TemplateTests(unittest.TestCase):
    def test_six_task_example_round_trip(self) -> None:
        text = (ROOT / "examples" / "standard.rules").read_text(encoding="utf-8")
        standard = parse_template(text)
        self.assertEqual(len(standard), 6)
        generic = to_generic_template(standard)
        reparsed = parse_template(
            serialize_template(generic), allow_missing_values=True
        )
        self.assertEqual(generic, reparsed)
        self.assertEqual(collect_categories(generic), ("part_a", "part_b"))

    def test_attribute_indexes_must_be_contiguous(self) -> None:
        text = (
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=a | PROPERTY_1=attribute.color | VALUE_1=red | "
            "SUBJECT_3=b | PROPERTY_3=attribute.shape | VALUE_3=round | "
            "OP=EQ | VALUE=red+round"
        )
        with self.assertRaisesRegex(TemplateError, "contiguous"):
            parse_template(text)

    def test_attribute_combination_total_value_must_match_members(self) -> None:
        text = (
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=a | PROPERTY_1=attribute.color | VALUE_1=red | "
            "SUBJECT_2=b | PROPERTY_2=attribute.shape | VALUE_2=round | "
            "OP=EQ | VALUE=red+square"
        )
        with self.assertRaisesRegex(TemplateError, "ordered VALUE_n combination"):
            parse_template(text)

    def test_attribute_combination_accepts_allowed_value_list(self) -> None:
        text = (
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=fruit_icon | PROPERTY_1=attribute.type | "
            "SUBJECT_2=liquid | PROPERTY_2=attribute.type | OP=EQ | "
            'VALUE=["cherry:cherry","banana:banana","orange:orange"]'
        )
        standard = parse_template(text)
        self.assertEqual(
            standard[0].expected,
            ("cherry:cherry", "banana:banana", "orange:orange"),
        )
        self.assertTrue(all(item.expected is None for item in standard[0].attributes))
        self.assertEqual(serialize_template(standard), text)
        generic = to_generic_template(standard)
        self.assertNotIn("VALUE=", serialize_template(generic))

    def test_allowed_value_list_checks_combination_arity(self) -> None:
        text = (
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=fruit_icon | PROPERTY_1=attribute.type | "
            "SUBJECT_2=liquid | PROPERTY_2=attribute.type | OP=EQ | "
            'VALUE=["cherry"]'
        )
        with self.assertRaisesRegex(TemplateError, "2 colon-separated values"):
            parse_template(text)

    def test_allowed_value_list_normalizes_chinese_punctuation(self) -> None:
        text = (
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=fruit_icon | PROPERTY_1=attribute.type | "
            "SUBJECT_2=liquid | PROPERTY_2=attribute.type | OP=EQ | "
            "VALUE=[“cherry：cherry”，“banana：banana”]"
        )
        standard = parse_template(text)
        self.assertEqual(
            standard[0].expected,
            ("cherry:cherry", "banana:banana"),
        )
        self.assertIn(
            'VALUE=["cherry:cherry","banana:banana"]',
            serialize_template(standard),
        )

    def test_spatial_relation_is_stored_in_value(self) -> None:
        text = (
            "[C001] TASK=SPATIAL_COMBINATION | SUBJECT=pattern_label | "
            "PROPERTY=spatial.position | OBJECT=juice_bottle | "
            "OP=EQ | VALUE=middle"
        )
        standard = parse_template(text)
        self.assertEqual(standard[0].expected, "middle")
        self.assertEqual(serialize_template(standard), text)

    def test_spatial_relation_count_round_trip(self) -> None:
        text = (
            "[C001] TASK=SPATIAL_COMBINATION | SUBJECT=compartment | "
            "PROPERTY=spatial.position | OBJECT=pin | OP=EQ | "
            "VALUE=contains | COUNT=[14,15]"
        )
        standard = parse_template(text)
        self.assertEqual(standard[0].expected, "contains")
        self.assertEqual(standard[0].count, (14, 15))
        self.assertEqual(serialize_template(standard), text)

        generic = to_generic_template(standard)
        self.assertIsNone(generic[0].expected)
        self.assertIsNone(generic[0].count)
        serialized_generic = serialize_template(generic)
        self.assertNotIn("VALUE=", serialized_generic)
        self.assertNotIn("COUNT=", serialized_generic)
        self.assertEqual(
            generic,
            parse_template(serialized_generic, allow_missing_values=True),
        )

    def test_spatial_relation_count_at_least_round_trip(self) -> None:
        text = (
            "[C001] TASK=SPATIAL_COMBINATION | SUBJECT=almonds | "
            "PROPERTY=spatial.position | OBJECT=box | OP=GE | "
            "VALUE=inside | COUNT=[1]"
        )
        standard = parse_template(text)
        self.assertEqual(standard[0].op, "GE")
        self.assertEqual(standard[0].count, (1,))
        self.assertEqual(serialize_template(standard), text)

        generic = to_generic_template(standard)
        self.assertEqual(generic[0].op, "GE")
        self.assertEqual(
            generic,
            parse_template(serialize_template(generic), allow_missing_values=True),
        )

    def test_spatial_relation_ge_requires_count_in_standard_template(self) -> None:
        text = (
            "[C001] TASK=SPATIAL_COMBINATION | SUBJECT=almonds | "
            "PROPERTY=spatial.position | OBJECT=box | OP=GE | VALUE=inside"
        )
        with self.assertRaisesRegex(TemplateError, "OP=GE requires COUNT"):
            parse_template(text)

    def test_standard_requires_values_but_generic_does_not(self) -> None:
        text = "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | OP=EQ"
        with self.assertRaisesRegex(TemplateError, "missing required field VALUE"):
            parse_template(text)
        parsed = parse_template(text, allow_missing_values=True)
        self.assertIsNone(parsed[0].expected)

    def test_generic_rejects_concrete_values(self) -> None:
        rules = parse_template(
            "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | OP=EQ | VALUE=[2]",
            allow_missing_values=True,
        )
        with self.assertRaisesRegex(TemplateError, "must not contain"):
            validate_generic_values_missing(rules)

    def test_measurement_range_and_count_round_trip(self) -> None:
        standard = parse_template(
            "[L001] TASK=LENGTH | SUBJECT=bolt | PROPERTY=length | "
            "OP=RANGE | VALUE=[200,250] | COUNT=[1,2]"
        )
        self.assertEqual(standard[0].minimum, 200)
        self.assertEqual(standard[0].maximum, 250)
        self.assertEqual(standard[0].count, (1, 2))
        generic = to_generic_template(standard)
        self.assertIsNone(generic[0].minimum)
        self.assertIsNone(generic[0].count)
        text = serialize_template(generic)
        self.assertNotIn("VALUE=", text)
        self.assertNotIn("COUNT=", text)
        reparsed = parse_template(text, allow_missing_values=True)
        self.assertEqual(generic, reparsed)

    def test_measurement_range_requires_value_and_count(self) -> None:
        text = (
            "[A001] TASK=AREA | SUBJECT=washer | PROPERTY=area | "
            "OP=RANGE | VALUE=[100,200]"
        )
        with self.assertRaisesRegex(TemplateError, "both VALUE and COUNT"):
            parse_template(text)

    def test_count_values_require_non_empty_integer_arrays(self) -> None:
        scalar = (
            "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | "
            "OP=EQ | VALUE=2"
        )
        with self.assertRaisesRegex(TemplateError, "non-empty JSON array"):
            parse_template(scalar)

        invalid_member = (
            "[L001] TASK=LENGTH | SUBJECT=part | PROPERTY=length | "
            'OP=RANGE | VALUE=[10,20] | COUNT=[1,"2"]'
        )
        with self.assertRaisesRegex(TemplateError, "non-negative integers"):
            parse_template(invalid_member)

    def test_count_value_array_round_trip(self) -> None:
        text = (
            "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | "
            "OP=EQ | VALUE=[1,2]"
        )
        rules = parse_template(text)
        self.assertEqual(rules[0].expected, (1, 2))
        self.assertEqual(serialize_template(rules), text)

    def test_repository_saves_validated_pair(self) -> None:
        text = (ROOT / "examples" / "standard.rules").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            repository = TemplateRepository(directory)
            standard_path, generic_path = repository.save_standard("demo", text)
            self.assertTrue(standard_path.is_file())
            self.assertTrue(generic_path.is_file())
            generic, standard = repository.load("demo")
            self.assertEqual(len(generic), len(standard))
            with self.assertRaisesRegex(TemplateError, "overwrite"):
                repository.save_standard("demo", text)


if __name__ == "__main__":
    unittest.main()
