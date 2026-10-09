"""End-to-end V1 routing over one shared category mask pool."""

from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite
from typing import Any, Mapping, Sequence

from .errors import ExecutionError
from .geometry import mask_area, mask_length
from .interfaces import (
    AttributeDetector,
    MaskProcessor,
    PatchAnomalyScorer,
    Segmenter,
)
from .mask_processing import MaskPostprocessor
from .models import (
    AttributePrediction,
    AttributeTarget,
    InferenceReport,
    MaskInstance,
    Rule,
    RuleResult,
    TaskType,
)
from .scoring import (
    attribute_satisfaction_score,
    conjunction_satisfaction_score,
    count_satisfaction_score,
    relation_count_at_least_satisfaction_score,
    relation_count_satisfaction_score,
    score_pair,
)
from .spatial import (
    SpatialConfig,
    evaluate_spatial_relation,
    evaluate_spatial_relation_by_subject,
    score_spatial_relation,
    score_spatial_relation_by_subject,
)
from .templates import (
    collect_categories,
    serialize_template,
    validate_template_alignment,
)


FUSION_DECISION_BOUNDARY = 0.5
SCORE_FUSION_POLICY = "rule_patch_decision_boundary_max_v4"
SCORE_FUSION_RUN_TAG = "fusioncalv4"
SCORE_TIE_ABSOLUTE_TOLERANCE = 1e-12


def highest_scoring_branch(
    scores: Mapping[str, float | None],
) -> tuple[str | None, float | None]:
    """Return the unique highest branch, or ``mixed`` for an exact score tie."""

    available = {
        str(name): float(score)
        for name, score in scores.items()
        if score is not None
    }
    if not available:
        return None, None
    highest = max(available.values())
    winners = [
        name
        for name, score in available.items()
        if isclose(
            score,
            highest,
            rel_tol=0.0,
            abs_tol=SCORE_TIE_ABSOLUTE_TOLERANCE,
        )
    ]
    return (winners[0] if len(winners) == 1 else "mixed"), highest


def calibrated_rule_fusion_score(
    results: Sequence[RuleResult],
) -> float | None:
    """Map rule evidence onto the shared threshold-relative scale.

    Rule violation scores are confidence-like evidence and their raw magnitude
    is not directly comparable with patch evidence. The explicit PASS/FAIL
    decision supplies the missing calibration anchor: PASS evidence stays
    below 0.5, while FAIL evidence stays above 0.5.
    """

    calibrated = []
    for result in results:
        if not result.evidence_valid or result.violation_score is None:
            continue
        violation = float(result.violation_score)
        if result.satisfied is False:
            calibrated.append(
                FUSION_DECISION_BOUNDARY
                + (1.0 - FUSION_DECISION_BOUNDARY) * violation
            )
        else:
            calibrated.append(FUSION_DECISION_BOUNDARY * violation)
    return max(calibrated) if calibrated else None


class LogicAnomalyPipeline:
    """Execute rules and fuse optional patch evidence."""

    def __init__(
        self,
        segmenter: Segmenter | None,
        attribute_detector: AttributeDetector | None,
        *,
        mask_processor: MaskProcessor | None = None,
        patch_scorer: PatchAnomalyScorer | None = None,
        grouping_tolerance: float = 0.10,
        spatial_config: SpatialConfig | None = None,
        rule_inference_enabled: bool = True,
    ) -> None:
        self.segmenter = segmenter
        self.attribute_detector = attribute_detector
        self.mask_processor = (
            mask_processor or MaskPostprocessor()
            if rule_inference_enabled
            else None
        )
        self.patch_scorer = patch_scorer
        self.grouping_tolerance = grouping_tolerance
        self.spatial_config = spatial_config or SpatialConfig()
        self.rule_inference_enabled = rule_inference_enabled
        if self.rule_inference_enabled and (
            self.segmenter is None or self.attribute_detector is None
        ):
            raise ValueError(
                "rule inference requires a segmenter and attribute detector"
            )
        if not self.rule_inference_enabled and self.patch_scorer is None:
            raise ValueError("disabled rule inference requires patch matching")

    def run(
        self,
        image: Any,
        generic_rules: Sequence[Rule],
        standard_rules: Sequence[Rule],
    ) -> InferenceReport:
        if not self.rule_inference_enabled:
            return self._run_patch_only(image)

        pairs = validate_template_alignment(generic_rules, standard_rules)

        # This is the deliberate pre-SAM3 boundary: category fields only.
        categories = collect_categories(generic_rules)
        assert self.segmenter is not None
        assert self.mask_processor is not None
        raw_pool = self.segmenter.segment(image, categories)
        raw_pool = self.mask_processor.process(categories, raw_pool)
        return self._run_from_validated_pool(image, pairs, categories, raw_pool)

    def _run_patch_only(self, image: Any) -> InferenceReport:
        """Run the independent full-image Patch branch without rule dependencies."""

        assert self.patch_scorer is not None
        patch_evidence = self.patch_scorer.score(image)
        return InferenceReport(
            image_anomaly=patch_evidence.anomaly,
            image_anomaly_score=patch_evidence.calibrated_score,
            logical_anomaly=None,
            anomaly_source="structural" if patch_evidence.anomaly else None,
            categories=(),
            mask_counts={},
            results=(),
            filled_template=None,
            patch_evidence=patch_evidence,
            rule_inference_enabled=False,
            execution_mode="patch_only",
        )

    def run_from_mask_pool(
        self,
        image: Any,
        generic_rules: Sequence[Rule],
        standard_rules: Sequence[Rule],
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> InferenceReport:
        """Evaluate an already processed pool without segmenting or filtering it.

        Normal-only CLIP calibration uses this entry point to compare several
        verification thresholds against the exact same SAM3 proposals.
        """

        pairs = validate_template_alignment(generic_rules, standard_rules)
        categories = collect_categories(generic_rules)
        return self._run_from_validated_pool(image, pairs, categories, mask_pool)

    def _run_from_validated_pool(
        self,
        image: Any,
        pairs: Sequence[tuple[Rule, Rule]],
        categories: tuple[str, ...],
        raw_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> InferenceReport:
        mask_pool: dict[str, tuple[MaskInstance, ...]] = {
            category: tuple(raw_pool.get(category, ())) for category in categories
        }
        spatial_rules: dict[
            tuple[str, str], list[tuple[Rule, Rule]]
        ] = {}
        for generic, standard in pairs:
            if generic.task is TaskType.SPATIAL_COMBINATION:
                key = (generic.subject or "", generic.object or "")
                spatial_rules.setdefault(key, []).append((generic, standard))

        results: list[RuleResult] = []
        filled_rules: list[Rule] = []
        for generic, standard in pairs:
            try:
                result, filled = self._execute_rule(
                    image=image,
                    generic=generic,
                    standard=standard,
                    mask_pool=mask_pool,
                    spatial_rules=spatial_rules,
                )
            except ExecutionError as exc:
                result = RuleResult(
                    rule_id=generic.rule_id,
                    task=generic.task,
                    satisfied=None,
                    actual=None,
                    expected=self._expected_view(standard),
                    satisfaction_score=None,
                    violation_score=None,
                    evidence_valid=False,
                    error=str(exc),
                )
                filled = generic
            results.append(result)
            filled_rules.append(filled)

        if any(result.satisfied is False for result in results):
            logical_anomaly: bool | None = True
        elif any(result.satisfied is None for result in results):
            logical_anomaly = None
        else:
            logical_anomaly = False
        valid_scores = [
            result.violation_score
            for result in results
            if result.evidence_valid and result.violation_score is not None
        ]
        rule_score = max(valid_scores) if valid_scores else None
        rule_fusion_score = calibrated_rule_fusion_score(results)

        patch_evidence = (
            self.patch_scorer.score(image)
            if self.patch_scorer is not None
            else None
        )

        if patch_evidence is None:
            image_score = rule_score
            image_anomaly = logical_anomaly
            anomaly_source = "rule" if image_anomaly is True else None
        else:
            anomaly_source, image_score = highest_scoring_branch(
                {
                    "rule": rule_fusion_score,
                    "structural": patch_evidence.calibrated_score,
                }
            )
            branch_anomaly = bool(
                any(result.satisfied is False for result in results)
                or patch_evidence.anomaly
            )
            if branch_anomaly:
                image_anomaly = True
            elif logical_anomaly is None:
                image_anomaly = None
                anomaly_source = None
            else:
                image_anomaly = False
                anomaly_source = None

        return InferenceReport(
            image_anomaly=image_anomaly,
            image_anomaly_score=image_score,
            logical_anomaly=logical_anomaly,
            anomaly_source=anomaly_source,
            categories=categories,
            mask_counts={key: len(value) for key, value in mask_pool.items()},
            results=tuple(results),
            filled_template=serialize_template(filled_rules, include_values=True),
            patch_evidence=patch_evidence,
            execution_mode="rule_patch" if patch_evidence is not None else "rule_only",
        )

    @staticmethod
    def _expected_view(rule: Rule) -> Any:
        if rule.op == "RANGE":
            return {
                "range": [rule.minimum, rule.maximum],
                "count": list(rule.count) if rule.count is not None else None,
            }
        if rule.task is TaskType.COUNT and isinstance(rule.expected, tuple):
            return list(rule.expected)
        if rule.task is TaskType.SPATIAL_COMBINATION and rule.count is not None:
            return {
                "relation": rule.expected,
                "count": list(rule.count),
            }
        if rule.task is not TaskType.ATTRIBUTE_COMBINATION:
            return rule.expected
        if isinstance(rule.expected, tuple):
            return {"VALUE": list(rule.expected)}
        result = {f"VALUE_{item.index}": item.expected for item in rule.attributes}
        result["VALUE"] = rule.expected
        return result

    def _execute_rule(
        self,
        *,
        image: Any,
        generic: Rule,
        standard: Rule,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        spatial_rules: Mapping[
            tuple[str, str], Sequence[tuple[Rule, Rule]]
        ],
    ) -> tuple[RuleResult, Rule]:
        if generic.task is TaskType.COUNT:
            instances = mask_pool[generic.subject or ""]
            actual = len(instances)
            if not isinstance(standard.expected, tuple) or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in standard.expected
            ):
                raise ExecutionError(
                    f"{standard.rule_id}: COUNT VALUE must be an integer array"
                )
            support = count_satisfaction_score(instances, standard.expected)
            satisfaction, violation = score_pair(support)
            details = {
                "score_method": "poisson_binomial_mask_existence",
                "mask_scores": [item.score for item in instances],
            }
            result = RuleResult(
                rule_id=generic.rule_id,
                task=generic.task,
                satisfied=actual in standard.expected,
                actual=actual,
                expected=list(standard.expected),
                satisfaction_score=satisfaction,
                violation_score=violation,
                details=details,
            )
            return result, replace(generic, expected=(actual,))

        if generic.task in {TaskType.LENGTH, TaskType.AREA}:
            return self._measurement_range_result(
                generic=generic,
                standard=standard,
                instances=mask_pool[generic.subject or ""],
            )

        if generic.task is TaskType.SPATIAL_COMBINATION:
            observed, support, details = self._spatial_rule_evidence(
                generic,
                standard,
                mask_pool,
            )
            if standard.count is not None:
                actual_count = int(details["subject_relation_count"])
                relation = str(details["relation"])
                expected = {
                    "relation": relation,
                    "count": list(standard.count),
                }
                if standard.op == "GE":
                    expected["operator"] = "GE"
                satisfaction, violation = score_pair(support)
                result = RuleResult(
                    rule_id=generic.rule_id,
                    task=generic.task,
                    satisfied=(
                        actual_count >= min(standard.count)
                        if standard.op == "GE"
                        else actual_count in standard.count
                    ),
                    actual={"relation": relation, "count": actual_count},
                    expected=expected,
                    satisfaction_score=satisfaction,
                    violation_score=violation,
                    details=details,
                )
                return result, replace(
                    generic,
                    expected=relation,
                    count=(actual_count,),
                )
            return self._scalar_result(
                generic, standard, observed, support, details
            ), replace(generic, expected=observed)

        if generic.task is TaskType.ATTRIBUTE_ERROR:
            if self._is_count_attribute(generic.property or ""):
                instances = mask_pool[generic.subject or ""]
                actual, candidate_supports, details = self._count_attribute_evidence(
                    instances,
                    (standard.expected,),
                    rule_id=standard.rule_id,
                )
                support = candidate_supports[str(standard.expected)]
                details["expected_satisfaction_score"] = support
                details["violation_score"] = 1.0 - support
                return (
                    self._scalar_result(
                        generic, standard, actual, support, details
                    ),
                    replace(generic, expected=actual),
                )
            spatial_object = self._spatial_attribute_object(
                generic.property or ""
            )
            if spatial_object is not None:
                actual, candidate_supports, details = (
                    self._spatial_attribute_evidence(
                        generic.subject or "",
                        spatial_object,
                        (standard.expected,),
                        spatial_rules=spatial_rules,
                        mask_pool=mask_pool,
                        rule_id=standard.rule_id,
                    )
                )
                support = candidate_supports[str(standard.expected)]
                details["expected_satisfaction_score"] = support
                details["violation_score"] = 1.0 - support
                return (
                    self._scalar_result(
                        generic, standard, actual, support, details
                    ),
                    replace(generic, expected=actual),
                )
            measurement = self._measurement_attribute(generic.property or "")
            if measurement is not None:
                instances = mask_pool[generic.subject or ""]
                actual, candidate_supports, details = (
                    self._measurement_attribute_evidence(
                        instances,
                        measurement,
                        (standard.expected,),
                        rule_id=standard.rule_id,
                    )
                )
                support = candidate_supports[str(standard.expected)]
                details["expected_satisfaction_score"] = support
                details["violation_score"] = 1.0 - support
                return (
                    self._scalar_result(
                        generic, standard, actual, support, details
                    ),
                    replace(generic, expected=actual),
                )
            prediction = self.attribute_detector.detect(
                image,
                generic.subject or "",
                generic.property or "",
                mask_pool[generic.subject or ""],
            )
            support, score_method = attribute_satisfaction_score(
                prediction, standard.expected
            )
            details = {
                "confidence": prediction.confidence,
                "per_instance": list(prediction.per_instance),
                "candidate_probabilities": dict(prediction.probabilities),
                "score_method": score_method,
            }
            return (
                self._scalar_result(
                    generic, standard, prediction.value, support, details
                ),
                replace(generic, expected=prediction.value),
            )

        if generic.task is TaskType.ATTRIBUTE_COMBINATION:
            if isinstance(standard.expected, tuple):
                return self._allowed_attribute_combination_result(
                    image=image,
                    generic=generic,
                    standard=standard,
                    mask_pool=mask_pool,
                    spatial_rules=spatial_rules,
                )
            actual_attributes: list[AttributeTarget] = []
            details: dict[str, Any] = {"attributes": {}}
            supports: list[float] = []
            for item, expected_item in zip(generic.attributes, standard.attributes):
                if self._is_count_attribute(item.property):
                    actual_count, candidate_supports, attribute_details = (
                        self._count_attribute_evidence(
                            mask_pool[item.subject],
                            (expected_item.expected,),
                            rule_id=standard.rule_id,
                        )
                    )
                    support = candidate_supports[str(expected_item.expected)]
                    actual_attributes.append(
                        replace(item, expected=actual_count)
                    )
                    supports.append(support)
                    attribute_details["expected_satisfaction_score"] = support
                    attribute_details["violation_score"] = 1.0 - support
                    details["attributes"][f"VALUE_{item.index}"] = (
                        attribute_details
                    )
                    continue
                spatial_object = self._spatial_attribute_object(item.property)
                if spatial_object is not None:
                    actual_spatial, candidate_supports, attribute_details = (
                        self._spatial_attribute_evidence(
                            item.subject,
                            spatial_object,
                            (expected_item.expected,),
                            spatial_rules=spatial_rules,
                            mask_pool=mask_pool,
                            rule_id=standard.rule_id,
                        )
                    )
                    support = candidate_supports[str(expected_item.expected)]
                    actual_attributes.append(
                        replace(item, expected=actual_spatial)
                    )
                    supports.append(support)
                    attribute_details["expected_satisfaction_score"] = support
                    attribute_details["violation_score"] = 1.0 - support
                    details["attributes"][f"VALUE_{item.index}"] = (
                        attribute_details
                    )
                    continue
                measurement = self._measurement_attribute(item.property)
                if measurement is not None:
                    actual_measurement, candidate_supports, attribute_details = (
                        self._measurement_attribute_evidence(
                            mask_pool[item.subject],
                            measurement,
                            (expected_item.expected,),
                            rule_id=standard.rule_id,
                        )
                    )
                    support = candidate_supports[str(expected_item.expected)]
                    actual_attributes.append(
                        replace(item, expected=actual_measurement)
                    )
                    supports.append(support)
                    attribute_details["expected_satisfaction_score"] = support
                    attribute_details["violation_score"] = 1.0 - support
                    details["attributes"][f"VALUE_{item.index}"] = (
                        attribute_details
                    )
                    continue
                prediction = self.attribute_detector.detect(
                    image,
                    item.subject,
                    item.property,
                    mask_pool[item.subject],
                )
                actual_attributes.append(replace(item, expected=prediction.value))
                support, score_method = attribute_satisfaction_score(
                    prediction, expected_item.expected
                )
                supports.append(support)
                details["attributes"][f"VALUE_{item.index}"] = {
                    "confidence": prediction.confidence,
                    "per_instance": list(prediction.per_instance),
                    "candidate_probabilities": dict(prediction.probabilities),
                    "expected_satisfaction_score": support,
                    "violation_score": 1.0 - support,
                    "score_method": score_method,
                }

            combined = "+".join(str(item.expected) for item in actual_attributes)
            actual = {
                f"VALUE_{item.index}": item.expected for item in actual_attributes
            }
            actual["VALUE"] = combined
            expected = self._expected_view(standard)
            passed = actual == expected
            support = conjunction_satisfaction_score(supports)
            satisfaction_score, violation_score = score_pair(support)
            details["score_method"] = "hard_conjunction_min_support"
            details["combined_value_scored_separately"] = False
            result = RuleResult(
                rule_id=generic.rule_id,
                task=generic.task,
                satisfied=passed,
                actual=actual,
                expected=expected,
                satisfaction_score=satisfaction_score,
                violation_score=violation_score,
                details=details,
            )
            return result, replace(
                generic,
                expected=combined,
                attributes=tuple(actual_attributes),
            )

        raise ExecutionError(f"no executor registered for task {generic.task.value}")

    def _allowed_attribute_combination_result(
        self,
        *,
        image: Any,
        generic: Rule,
        standard: Rule,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        spatial_rules: Mapping[
            tuple[str, str], Sequence[tuple[Rule, Rule]]
        ],
    ) -> tuple[RuleResult, Rule]:
        allowed = standard.expected
        if not isinstance(allowed, tuple) or not allowed:
            raise ExecutionError(
                f"{standard.rule_id}: attribute combination has no allowed VALUE"
            )

        evidence: list[AttributePrediction | Mapping[str, float]] = []
        actual_attributes: list[AttributeTarget] = []
        details: dict[str, Any] = {"attributes": {}}
        allowed_labels = [combination.split(":") for combination in allowed]
        for position, item in enumerate(generic.attributes):
            if self._is_count_attribute(item.property):
                labels = tuple(values[position] for values in allowed_labels)
                actual_count, candidate_supports, attribute_details = (
                    self._count_attribute_evidence(
                        mask_pool[item.subject],
                        labels,
                        rule_id=standard.rule_id,
                    )
                )
                evidence.append(candidate_supports)
                actual_attributes.append(replace(item, expected=actual_count))
                details["attributes"][f"VALUE_{item.index}"] = attribute_details
                continue
            spatial_object = self._spatial_attribute_object(item.property)
            if spatial_object is not None:
                labels = tuple(values[position] for values in allowed_labels)
                actual_spatial, candidate_supports, attribute_details = (
                    self._spatial_attribute_evidence(
                        item.subject,
                        spatial_object,
                        labels,
                        spatial_rules=spatial_rules,
                        mask_pool=mask_pool,
                        rule_id=standard.rule_id,
                    )
                )
                evidence.append(candidate_supports)
                actual_attributes.append(
                    replace(item, expected=actual_spatial)
                )
                details["attributes"][f"VALUE_{item.index}"] = attribute_details
                continue
            measurement = self._measurement_attribute(item.property)
            if measurement is not None:
                labels = tuple(values[position] for values in allowed_labels)
                actual_measurement, candidate_supports, attribute_details = (
                    self._measurement_attribute_evidence(
                        mask_pool[item.subject],
                        measurement,
                        labels,
                        rule_id=standard.rule_id,
                    )
                )
                evidence.append(candidate_supports)
                actual_attributes.append(
                    replace(item, expected=actual_measurement)
                )
                details["attributes"][f"VALUE_{item.index}"] = attribute_details
                continue
            prediction = self.attribute_detector.detect(
                image,
                item.subject,
                item.property,
                mask_pool[item.subject],
            )
            evidence.append(prediction)
            actual_attributes.append(replace(item, expected=prediction.value))
            details["attributes"][f"VALUE_{item.index}"] = {
                "predicted": prediction.value,
                "confidence": prediction.confidence,
                "per_instance": list(prediction.per_instance),
                "candidate_probabilities": dict(prediction.probabilities),
            }

        combination_supports: dict[str, float] = {}
        for combination in allowed:
            labels = combination.split(":")
            supports = [
                source.get(label, 0.0)
                if isinstance(source, Mapping)
                else attribute_satisfaction_score(source, label)[0]
                for source, label in zip(evidence, labels)
            ]
            combination_supports[combination] = conjunction_satisfaction_score(
                supports
            )

        combined = ":".join(
            str(item.expected) if item.expected is not None else ""
            for item in actual_attributes
        )
        passed = combined in allowed
        support = max(combination_supports.values())
        satisfaction_score, violation_score = score_pair(support)
        actual = {
            f"VALUE_{item.index}": item.expected for item in actual_attributes
        }
        actual["VALUE"] = combined
        expected = self._expected_view(standard)
        details["allowed_combination_supports"] = combination_supports
        details["score_method"] = "allowed_combination_max_of_min_support"
        result = RuleResult(
            rule_id=generic.rule_id,
            task=generic.task,
            satisfied=passed,
            actual=actual,
            expected=expected,
            satisfaction_score=satisfaction_score,
            violation_score=violation_score,
            details=details,
        )
        return result, replace(
            generic,
            expected=combined,
            attributes=tuple(actual_attributes),
        )

    def _spatial_rule_evidence(
        self,
        generic: Rule,
        standard: Rule,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> tuple[Any, float, dict[str, Any]]:
        if isinstance(standard.expected, tuple) or standard.expected is None:
            raise ExecutionError(
                f"{standard.rule_id}: spatial VALUE must name one relation"
            )
        legacy_boolean = isinstance(standard.expected, bool)
        relation = (
            generic.property or ""
            if legacy_boolean
            else str(standard.expected)
        )
        subjects = mask_pool[generic.subject or ""]
        objects = mask_pool[generic.object or ""]
        if standard.count is not None:
            if legacy_boolean:
                raise ExecutionError(
                    f"{standard.rule_id}: counted spatial rules require a named "
                    "relation VALUE"
                )
            subject_matches = evaluate_spatial_relation_by_subject(
                subjects,
                objects,
                relation,
                config=self.spatial_config,
            )
            subject_supports = score_spatial_relation_by_subject(
                subjects,
                objects,
                relation,
                config=self.spatial_config,
            )
            actual_count = sum(subject_matches)
            relation_holds = (
                actual_count >= min(standard.count)
                if standard.op == "GE"
                else actual_count in standard.count
            )
            relation_support = (
                relation_count_at_least_satisfaction_score(
                    subject_supports,
                    standard.count,
                )
                if standard.op == "GE"
                else relation_count_satisfaction_score(
                    subject_supports,
                    standard.count,
                )
            )
            details = {
                "pair_semantics": "unique_subject_count",
                "count_semantics": "subjects_with_any_matching_object",
                "relation": relation,
                "count_operator": standard.op,
                "relation_true_support": relation_support,
                "subject_relation_count": actual_count,
                "expected_subject_relation_counts": list(standard.count),
                "subject_relation_matches": list(subject_matches),
                "subject_relation_supports": list(subject_supports),
                "score_method": (
                    "fuzzy_at_least_unique_subject_relation_count"
                    if standard.op == "GE"
                    else "fuzzy_exact_unique_subject_relation_count"
                ),
            }
            observed = relation if relation_holds else f"not_{relation}"
            return observed, relation_support, details

        relation_holds = evaluate_spatial_relation(
            subjects,
            objects,
            relation,
            config=self.spatial_config,
        )
        relation_support = score_spatial_relation(
            subjects,
            objects,
            relation,
            config=self.spatial_config,
        )
        if legacy_boolean:
            observed: Any = relation_holds
            support = (
                relation_support
                if standard.expected
                else 1.0 - relation_support
            )
        else:
            observed = (
                str(standard.expected)
                if relation_holds
                else f"not_{standard.expected}"
            )
            support = relation_support
        details = {
            "pair_semantics": "any_pair",
            "relation": relation,
            "relation_true_support": relation_support,
            "score_method": "continuous_geometry_with_mask_existence",
        }
        return observed, support, details

    @staticmethod
    def _spatial_attribute_object(property_name: str) -> str | None:
        prefix = "attribute.spatial_combination."
        if not property_name.lower().startswith(prefix):
            return None
        object_name = property_name[len(prefix) :].strip()
        if not object_name:
            raise ExecutionError(
                "attribute.spatial_combination.<OBJECT> requires an object name"
            )
        return object_name

    def _spatial_attribute_evidence(
        self,
        subject: str,
        object_name: str,
        expected_values: Sequence[object],
        *,
        spatial_rules: Mapping[
            tuple[str, str], Sequence[tuple[Rule, Rule]]
        ],
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        rule_id: str,
    ) -> tuple[Any, dict[str, float], dict[str, Any]]:
        matches = tuple(spatial_rules.get((subject, object_name), ()))
        if not matches:
            raise ExecutionError(
                f"{rule_id}: attribute.spatial_combination.{object_name} needs "
                f"one SPATIAL_COMBINATION rule with SUBJECT={subject} and "
                f"OBJECT={object_name}"
            )
        if len(matches) > 1:
            raise ExecutionError(
                f"{rule_id}: attribute.spatial_combination.{object_name} is "
                f"ambiguous because {len(matches)} matching "
                "SPATIAL_COMBINATION rules exist"
            )
        spatial_generic, spatial_standard = matches[0]
        observed, _, spatial_details = self._spatial_rule_evidence(
            spatial_generic,
            spatial_standard,
            mask_pool,
        )
        relation = str(spatial_details["relation"])
        relation_support = float(spatial_details["relation_true_support"])
        legacy_boolean = isinstance(spatial_standard.expected, bool)
        candidate_supports: dict[str, float] = {}
        for value in expected_values:
            label = str(value)
            if legacy_boolean:
                normalized = label.strip().lower()
                candidate_supports[label] = (
                    relation_support
                    if normalized == "true"
                    else 1.0 - relation_support
                    if normalized == "false"
                    else 0.0
                )
            else:
                candidate_supports[label] = (
                    relation_support
                    if label == relation
                    else 1.0 - relation_support
                    if label == f"not_{relation}"
                    else 0.0
                )
        actual = (
            ("true" if observed else "false")
            if legacy_boolean
            and all(isinstance(value, str) for value in expected_values)
            else observed
        )
        details = {
            **spatial_details,
            "predicted": actual,
            "source": "spatial_combination_rule",
            "reused_rule_id": spatial_standard.rule_id,
            "candidate_probabilities": candidate_supports,
        }
        return actual, candidate_supports, details

    @staticmethod
    def _is_count_attribute(property_name: str) -> bool:
        return property_name.split(".", 1)[-1].strip().lower() == "count"

    @staticmethod
    def _measurement_attribute(property_name: str) -> str | None:
        attribute = property_name.split(".", 1)[-1].strip().lower()
        return attribute if attribute in {"length", "area"} else None

    @staticmethod
    def _parse_attribute_count(value: object, *, rule_id: str) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        text = str(value).strip()
        if not text.isdigit():
            raise ExecutionError(
                f"{rule_id}: attribute.count values must be non-negative integers"
            )
        return int(text)

    @classmethod
    def _count_attribute_evidence(
        cls,
        instances: Sequence[MaskInstance],
        expected_values: Sequence[object],
        *,
        rule_id: str,
    ) -> tuple[int, dict[str, float], dict[str, Any]]:
        labels = tuple(dict.fromkeys(str(value) for value in expected_values))
        candidate_supports = {
            label: count_satisfaction_score(
                instances,
                (cls._parse_attribute_count(label, rule_id=rule_id),),
            )
            for label in labels
        }
        actual = len(instances)
        details = {
            "predicted": actual,
            "source": "shared_mask_pool_count",
            "mask_scores": [item.score for item in instances],
            "candidate_probabilities": candidate_supports,
            "score_method": "poisson_binomial_mask_existence",
        }
        return actual, candidate_supports, details

    @staticmethod
    def _parse_attribute_measurement(value: object, *, rule_id: str) -> float:
        if isinstance(value, bool):
            raise ExecutionError(
                f"{rule_id}: attribute.length/area values must be finite numbers"
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                f"{rule_id}: attribute.length/area values must be finite numbers"
            ) from exc
        if not isfinite(number) or number < 0.0:
            raise ExecutionError(
                f"{rule_id}: attribute.length/area values must be finite "
                "non-negative numbers"
            )
        return number

    @classmethod
    def _measurement_attribute_evidence(
        cls,
        instances: Sequence[MaskInstance],
        measurement: str,
        expected_values: Sequence[object],
        *,
        rule_id: str,
    ) -> tuple[object, dict[str, float], dict[str, Any]]:
        if len(instances) != 1:
            raise ExecutionError(
                f"{rule_id}: attribute.{measurement} requires exactly one "
                f"segmented instance, found {len(instances)}"
            )
        instance = instances[0]
        measured = (
            mask_length(instance) if measurement == "length" else mask_area(instance)
        )
        candidates = tuple(
            (value, str(value), cls._parse_attribute_measurement(value, rule_id=rule_id))
            for value in expected_values
        )
        candidate_supports = {
            label: instance.score
            if isclose(measured, expected, rel_tol=1e-9, abs_tol=1e-9)
            else 0.0
            for _, label, expected in candidates
        }
        matched = next(
            (
                original
                for original, _, expected in candidates
                if isclose(measured, expected, rel_tol=1e-9, abs_tol=1e-9)
            ),
            None,
        )
        normalized = int(measured) if measured.is_integer() else measured
        actual = matched if matched is not None else normalized
        details = {
            "predicted": normalized,
            "source": f"shared_mask_pool_{measurement}",
            "measurements": [measured],
            "mask_scores": [instance.score],
            "candidate_probabilities": candidate_supports,
            "score_method": "deterministic_geometry_with_mask_existence",
        }
        return actual, candidate_supports, details

    def _measurement_range_result(
        self,
        *,
        generic: Rule,
        standard: Rule,
        instances: Sequence[MaskInstance],
    ) -> tuple[RuleResult, Rule]:
        if (
            standard.minimum is None
            or standard.maximum is None
            or standard.count is None
        ):
            raise ExecutionError(f"{standard.rule_id}: RANGE rule has no VALUE or COUNT")
        measure = "length" if generic.task is TaskType.LENGTH else "area"
        measure_fn = mask_length if measure == "length" else mask_area
        measurements = [measure_fn(instance) for instance in instances]
        matching_indexes = [
            index
            for index, value in enumerate(measurements)
            if standard.minimum <= value <= standard.maximum
        ]
        matching_instances = [instances[index] for index in matching_indexes]
        actual_count = len(matching_instances)
        support = count_satisfaction_score(matching_instances, standard.count)
        satisfaction_score, violation_score = score_pair(support)
        satisfied = actual_count in standard.count
        expected = self._expected_view(standard)
        actual = {
            "measurements": sorted(measurements, reverse=True),
            "matching_count": actual_count,
        }
        result = RuleResult(
            rule_id=generic.rule_id,
            task=generic.task,
            satisfied=satisfied,
            actual=actual,
            expected=expected,
            satisfaction_score=satisfaction_score,
            violation_score=violation_score,
            details={
                "measurement": measure,
                "unit": "pixel" if measure == "length" else "pixel_squared",
                "matching_instance_indexes": [index + 1 for index in matching_indexes],
                "matching_mask_scores": [item.score for item in matching_instances],
                "normal_range": [standard.minimum, standard.maximum],
                "score_method": "poisson_binomial_in_range_mask_existence",
            },
        )
        filled = replace(
            generic,
            minimum=standard.minimum,
            maximum=standard.maximum,
            count=(actual_count,),
        )
        return result, filled

    @staticmethod
    def _scalar_result(
        generic: Rule,
        standard: Rule,
        actual: Any,
        satisfaction_score: float,
        details: Mapping[str, Any],
    ) -> RuleResult:
        satisfaction, violation = score_pair(satisfaction_score)
        return RuleResult(
            rule_id=generic.rule_id,
            task=generic.task,
            satisfied=actual == standard.expected,
            actual=actual,
            expected=standard.expected,
            satisfaction_score=satisfaction,
            violation_score=violation,
            details=details,
        )
