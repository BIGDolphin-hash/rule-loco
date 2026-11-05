"""End-to-end V1 routing over one shared category mask pool."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import isclose, isfinite
from time import perf_counter
from typing import Any, Mapping, Sequence

from .errors import ExecutionError
from .geometry import mask_area, mask_length
from .interfaces import (
    AttributeDetector,
    CompositionAnomalyScorer,
    MaskProcessor,
    PatchAnomalyScorer,
    Segmenter,
    SharedFeatureExtractor,
)
from .mask_processing import MaskPostprocessor
from .models import (
    InferenceReport,
    MaskInstance,
    Rule,
    RuleResult,
    TaskType,
)
from .spatial import (
    SpatialConfig,
    evaluate_spatial_relation,
    evaluate_spatial_relation_by_subject,
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
POST_FUSION_SUM_THEN_CALIBRATE = "sum_then_calibrate"
RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE = "binary-selective"
RULE_MEMORY_SCHEDULE_BINARY_PARALLEL = "binary-parallel"
RULE_MEMORY_SCHEDULES = (
    RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
    RULE_MEMORY_SCHEDULE_BINARY_PARALLEL,
)


def _new_timings() -> dict[str, Any]:
    return {
        "mask_preparation": 0.0,
        "rule_total": 0.0,
        "rule_by_task": {task.value: 0.0 for task in TaskType},
        "rule_by_id": {},
        "memory_total": 0.0,
        "memory_feature_extraction": 0.0,
        "memory_composition": 0.0,
        "memory_global_patch": 0.0,
        "branch_wall_total": 0.0,
        "branch_overlap": 0.0,
        "pipeline_total": 0.0,
    }


def _with_pipeline_timings(
    report: InferenceReport,
    *,
    pipeline_started: float,
    mask_preparation: float,
) -> InferenceReport:
    timings = _new_timings()
    timings.update(report.timings_seconds)
    timings["mask_preparation"] = float(mask_preparation)
    timings["pipeline_total"] = perf_counter() - pipeline_started
    return replace(report, timings_seconds=timings)


def bounded_anomaly_score(raw_score: float) -> float:
    """Map one non-negative adaptive ratio to the common [0, 1) score range."""

    value = float(raw_score)
    if not isfinite(value) or value < 0.0:
        raise ValueError("raw anomaly score must be finite and non-negative")
    return value / (1.0 + value)


def scaled_bounded_anomaly_score(raw_score: float, tau: float) -> float:
    """Map a non-negative score with the explicit ``r / (r + tau)`` rule."""

    value = float(raw_score)
    scale = float(tau)
    if not isfinite(value) or value < 0.0:
        raise ValueError("raw anomaly score must be finite and non-negative")
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("post-fusion tau must be finite and positive")
    return value / (value + scale)


def highest_scoring_branch(
    scores: Mapping[str, float | None],
) -> tuple[str | None, float | None]:
    """Return the unique highest branch, or ``mixed`` for an exact score tie."""

    available = {
        str(name): float(score) for name, score in scores.items() if score is not None
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


class LogicAnomalyPipeline:
    """Execute rules and fuse optional patch evidence."""

    def __init__(
        self,
        segmenter: Segmenter | None,
        attribute_detector: AttributeDetector | None,
        *,
        mask_processor: MaskProcessor | None = None,
        patch_scorer: PatchAnomalyScorer | None = None,
        composition_scorer: CompositionAnomalyScorer | None = None,
        shared_feature_extractor: SharedFeatureExtractor | None = None,
        grouping_tolerance: float = 0.1,
        spatial_config: SpatialConfig | None = None,
        rule_inference_enabled: bool = True,
        rule_memory_schedule: str = RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
        post_fusion_tau: float = 1.0,
        mask_categories: Sequence[str] = (),
    ) -> None:
        self.segmenter = segmenter
        self.attribute_detector = attribute_detector
        needs_masks = rule_inference_enabled or composition_scorer is not None
        self.mask_processor = (
            mask_processor or MaskPostprocessor() if needs_masks else None
        )
        self.patch_scorer = patch_scorer
        self.composition_scorer = composition_scorer
        self.shared_feature_extractor = shared_feature_extractor
        self.grouping_tolerance = grouping_tolerance
        self.spatial_config = spatial_config or SpatialConfig()
        self.rule_inference_enabled = rule_inference_enabled
        self.mask_categories = tuple(
            dict.fromkeys((str(category) for category in mask_categories))
        )
        if rule_memory_schedule not in RULE_MEMORY_SCHEDULES:
            raise ValueError(
                "rule-memory schedule must be one of "
                + ", ".join(RULE_MEMORY_SCHEDULES)
            )
        self.rule_memory_schedule = rule_memory_schedule
        if not isfinite(float(post_fusion_tau)) or float(post_fusion_tau) <= 0.0:
            raise ValueError("post-fusion tau must be finite and positive")
        self.post_fusion_tau = float(post_fusion_tau)
        if self.rule_inference_enabled and (
            self.segmenter is None or self.attribute_detector is None
        ):
            raise ValueError(
                "rule inference requires a segmenter and attribute detector"
            )
        if self.composition_scorer is not None and self.segmenter is None:
            raise ValueError("composition matching requires a segmenter")
        if (
            not self.rule_inference_enabled
            and self.patch_scorer is None
            and (self.composition_scorer is None)
        ):
            raise ValueError("disabled rule inference requires patch matching")
        if self.composition_scorer is not None:
            patch_shares_features = self.patch_scorer is not None and all(
                (
                    hasattr(self.patch_scorer, name)
                    for name in ("extract_features", "score_features")
                )
            )
            if not patch_shares_features and self.shared_feature_extractor is None:
                raise ValueError(
                    "composition matching requires a shared DINO feature extractor"
                )

    def run(
        self, image: Any, generic_rules: Sequence[Rule], standard_rules: Sequence[Rule]
    ) -> InferenceReport:
        pipeline_started = perf_counter()
        mask_preparation = 0.0
        if not self.rule_inference_enabled and self.composition_scorer is not None:
            categories = collect_categories(generic_rules) or self.mask_categories
            if not categories:
                raise ValueError("composition-only inference requires mask categories")
            assert self.segmenter is not None
            assert self.mask_processor is not None
            mask_started = perf_counter()
            raw_pool = self.segmenter.segment(image, categories)
            raw_pool = self.mask_processor.process(categories, raw_pool)
            mask_preparation = perf_counter() - mask_started
            return _with_pipeline_timings(
                self._run_composition_only(
                    image=image, categories=categories, raw_pool=raw_pool
                ),
                pipeline_started=pipeline_started,
                mask_preparation=mask_preparation,
            )
        if not self.rule_inference_enabled:
            return _with_pipeline_timings(
                self._run_patch_only(image),
                pipeline_started=pipeline_started,
                mask_preparation=mask_preparation,
            )
        pairs = validate_template_alignment(generic_rules, standard_rules)
        categories = collect_categories(generic_rules)
        assert self.segmenter is not None
        assert self.mask_processor is not None
        mask_started = perf_counter()
        raw_pool = self.segmenter.segment(image, categories)
        raw_pool = self.mask_processor.process(categories, raw_pool)
        mask_preparation = perf_counter() - mask_started
        return _with_pipeline_timings(
            self._run_from_validated_pool(image, pairs, categories, raw_pool),
            pipeline_started=pipeline_started,
            mask_preparation=mask_preparation,
        )

    def _run_patch_only(self, image: Any) -> InferenceReport:
        """Run the independent global Patch branch."""
        timings = _new_timings()
        memory_started = perf_counter()
        assert self.patch_scorer is not None
        stage_started = perf_counter()
        patch_evidence = self.patch_scorer.score(image)
        timings["memory_global_patch"] = perf_counter() - stage_started
        timings["memory_total"] = perf_counter() - memory_started
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
            timings_seconds=timings,
        )

    def _run_composition_only(
        self,
        *,
        image: Any,
        categories: tuple[str, ...],
        raw_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> InferenceReport:
        """Run post paths from SAM3 masks without executing the rule branch."""
        timings = _new_timings()
        memory_started = perf_counter()
        assert self.composition_scorer is not None
        mask_pool = {
            category: tuple(raw_pool.get(category, ())) for category in categories
        }
        extractor = self.patch_scorer or self.shared_feature_extractor
        assert extractor is not None
        stage_started = perf_counter()
        features = extractor.extract_features(image)
        timings["memory_feature_extraction"] = perf_counter() - stage_started
        if self.patch_scorer is None:
            patch_evidence = None
        else:
            stage_started = perf_counter()
            patch_evidence = self.patch_scorer.score_features(features)
            timings["memory_global_patch"] = perf_counter() - stage_started
        stage_started = perf_counter()
        evidence = self.composition_scorer.score_features(features, mask_pool)
        timings["memory_composition"] = perf_counter() - stage_started
        timings["memory_total"] = perf_counter() - memory_started
        patch_ratio = (
            float(patch_evidence.fusion_score)
            if patch_evidence is not None and patch_evidence.fusion_score is not None
            else 0.0
        )
        score = scaled_bounded_anomaly_score(
            float(evidence.adaptive_score) + patch_ratio, self.post_fusion_tau
        )
        composition_anomaly = bool(evidence.anomaly)
        structural_anomaly = bool(patch_evidence is not None and patch_evidence.anomaly)
        fusion_anomaly = score > FUSION_DECISION_BOUNDARY
        return InferenceReport(
            image_anomaly=fusion_anomaly,
            image_anomaly_score=score,
            logical_anomaly=None,
            anomaly_source=(
                "mixed"
                if composition_anomaly and structural_anomaly
                else (
                    "composition"
                    if composition_anomaly
                    else (
                        "structural"
                        if structural_anomaly
                        else "mixed" if fusion_anomaly else None
                    )
                )
            ),
            categories=categories,
            mask_counts={key: len(value) for key, value in mask_pool.items()},
            results=(),
            filled_template=None,
            patch_evidence=patch_evidence,
            composition_evidence=evidence,
            rule_inference_enabled=False,
            execution_mode=(
                "dual_post" if patch_evidence is not None else "composition_only"
            ),
            timings_seconds=timings,
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

    def _evaluate_rule_branch(
        self,
        image: Any,
        pairs: Sequence[tuple[Rule, Rule]],
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        *,
        stop_on_failure: bool = False,
    ) -> tuple[
        tuple[RuleResult, ...],
        tuple[Rule, ...],
        bool | None,
        float | None,
        float | None,
        dict[str, Any],
    ]:
        timings = _new_timings()
        spatial_rules: dict[tuple[str, str], list[tuple[Rule, Rule]]] = {}
        for generic, standard in pairs:
            if generic.task is TaskType.SPATIAL_COMBINATION:
                key = (generic.subject, generic.object)
                spatial_rules.setdefault(key, []).append((generic, standard))
        results: list[RuleResult] = []
        filled_rules: list[Rule] = []
        rule_started = perf_counter()
        for generic, standard in pairs:
            task_started = perf_counter()
            try:
                result, filled = self._execute_binary_rule(
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
            task_elapsed = perf_counter() - task_started
            timings["rule_by_task"][generic.task.value] += task_elapsed
            timings["rule_by_id"][generic.rule_id] = task_elapsed
            results.append(result)
            filled_rules.append(filled)
            if stop_on_failure and result.satisfied is False:
                break
        timings["rule_total"] = perf_counter() - rule_started
        if any((result.satisfied is False for result in results)):
            logical_anomaly: bool | None = True
        elif any((result.satisfied is None for result in results)):
            logical_anomaly = None
        else:
            logical_anomaly = False
        rule_score = (
            1.0
            if logical_anomaly is True
            else 0.0 if logical_anomaly is False else None
        )
        rule_fusion_score = rule_score
        return (
            tuple(results),
            tuple(filled_rules),
            logical_anomaly,
            rule_score,
            rule_fusion_score,
            timings,
        )

    def _evaluate_memory_branch(
        self, image: Any, mask_pool: Mapping[str, Sequence[MaskInstance]]
    ) -> tuple[Any, Any, dict[str, Any]]:
        timings = _new_timings()
        patch_evidence = None
        composition_evidence = None
        memory_started = perf_counter()
        if self.composition_scorer is not None:
            extractor = self.patch_scorer or self.shared_feature_extractor
            assert extractor is not None
            stage_started = perf_counter()
            features = extractor.extract_features(image)
            timings["memory_feature_extraction"] = perf_counter() - stage_started
            if self.patch_scorer is not None:
                stage_started = perf_counter()
                patch_evidence = self.patch_scorer.score_features(features)
                timings["memory_global_patch"] = perf_counter() - stage_started
            stage_started = perf_counter()
            composition_evidence = self.composition_scorer.score_features(
                features, mask_pool
            )
            timings["memory_composition"] = perf_counter() - stage_started
        elif self.patch_scorer is not None:
            stage_started = perf_counter()
            patch_evidence = self.patch_scorer.score(image)
            timings["memory_global_patch"] = perf_counter() - stage_started
        timings["memory_total"] = perf_counter() - memory_started
        return (patch_evidence, composition_evidence, timings)

    @staticmethod
    def _merge_branch_timings(
        target: dict[str, Any], source: Mapping[str, Any]
    ) -> None:
        for key in (
            "rule_total",
            "memory_total",
            "memory_feature_extraction",
            "memory_composition",
            "memory_global_patch",
        ):
            target[key] = float(source.get(key, target[key]))
        target["rule_by_task"].update(source.get("rule_by_task", {}))
        target["rule_by_id"].update(source.get("rule_by_id", {}))

    def _run_from_validated_pool(
        self,
        image: Any,
        pairs: Sequence[tuple[Rule, Rule]],
        categories: tuple[str, ...],
        raw_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> InferenceReport:
        timings = _new_timings()
        mask_pool: dict[str, tuple[MaskInstance, ...]] = {
            category: tuple(raw_pool.get(category, ())) for category in categories
        }
        has_memory = (
            self.composition_scorer is not None or self.patch_scorer is not None
        )
        binary_selective = (
            self.rule_memory_schedule == RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE
        )
        binary_parallel = (
            self.rule_memory_schedule == RULE_MEMORY_SCHEDULE_BINARY_PARALLEL
        )
        selective = binary_selective
        parallel_schedule = binary_parallel
        parallel_execution = parallel_schedule and has_memory
        branch_started = perf_counter()

        if parallel_execution:
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="rule-memory"
            ) as executor:
                rule_future = executor.submit(
                    self._evaluate_rule_branch,
                    image,
                    pairs,
                    mask_pool,
                    stop_on_failure=True,
                )
                memory_future = executor.submit(
                    self._evaluate_memory_branch, image, mask_pool
                )
                rule_output = rule_future.result()
                patch_evidence, composition_evidence, memory_timings = (
                    memory_future.result()
                )
            branch_wall = perf_counter() - branch_started
            execution_mode = "rule_binary_memory_parallel"
        else:
            rule_output = self._evaluate_rule_branch(
                image,
                pairs,
                mask_pool,
                stop_on_failure=True,
            )
            patch_evidence = None
            composition_evidence = None
            memory_timings = _new_timings()
            branch_wall = perf_counter() - branch_started
            execution_mode = "rule_binary_only"

        (
            results,
            filled_rules,
            logical_anomaly,
            rule_score,
            rule_fusion_score,
            rule_timings,
        ) = rule_output
        self._merge_branch_timings(timings, rule_timings)

        rule_failed = logical_anomaly is True
        if selective and (logical_anomaly is None or rule_failed):
            timings["branch_wall_total"] = branch_wall
            return InferenceReport(
                image_anomaly=True if rule_failed else None,
                image_anomaly_score=(
                    rule_fusion_score
                    if rule_failed and rule_fusion_score is not None
                    else rule_score
                ),
                logical_anomaly=logical_anomaly,
                anomaly_source="rule" if rule_failed else None,
                categories=categories,
                mask_counts={key: len(value) for key, value in mask_pool.items()},
                results=results,
                filled_template=serialize_template(filled_rules, include_values=True),
                execution_mode=(
                    "rule_binary_early_exit" if rule_failed else "rule_binary_only"
                ),
                timings_seconds=timings,
            )

        if selective and has_memory:
            memory_started = perf_counter()
            patch_evidence, composition_evidence, memory_timings = (
                self._evaluate_memory_branch(image, mask_pool)
            )
            branch_wall += perf_counter() - memory_started
        self._merge_branch_timings(timings, memory_timings)
        timings["branch_wall_total"] = branch_wall
        timings["branch_overlap"] = max(
            0.0,
            timings["rule_total"] + timings["memory_total"] - branch_wall,
        )

        composition_anomaly = bool(
            composition_evidence is not None and composition_evidence.anomaly
        )
        structural_anomaly = bool(patch_evidence is not None and patch_evidence.anomaly)
        memory_score: float | None = None
        if patch_evidence is not None or composition_evidence is not None:
            composition_ratio = (
                float(composition_evidence.adaptive_score)
                if composition_evidence is not None
                else 0.0
            )
            patch_ratio = (
                float(patch_evidence.fusion_score)
                if patch_evidence is not None
                and patch_evidence.fusion_score is not None
                else 0.0
            )
            if (
                composition_evidence is None
                and patch_evidence is not None
                and patch_evidence.fusion_score is None
            ):
                memory_score = float(patch_evidence.calibrated_score)
            else:
                memory_score = scaled_bounded_anomaly_score(
                    composition_ratio + patch_ratio,
                    self.post_fusion_tau,
                )

        if parallel_execution:
            binary_rule_score = (
                rule_fusion_score if rule_fusion_score is not None else rule_score
            )
            available_scores = tuple(
                float(value)
                for value in (binary_rule_score, memory_score)
                if value is not None
            )
            image_score = max(available_scores) if available_scores else None
            image_anomaly = (
                image_score > FUSION_DECISION_BOUNDARY
                if image_score is not None
                else None
            )
            rule_anomaly = bool(
                rule_failed
                or (
                    binary_rule_score is not None
                    and binary_rule_score > FUSION_DECISION_BOUNDARY
                )
            )
            memory_anomaly = bool(
                memory_score is not None and memory_score > FUSION_DECISION_BOUNDARY
            )
            if rule_anomaly and memory_anomaly:
                anomaly_source = "mixed"
            elif rule_anomaly:
                anomaly_source = "rule"
            elif composition_anomaly and structural_anomaly:
                anomaly_source = "mixed"
            elif composition_anomaly:
                anomaly_source = "composition"
            elif structural_anomaly:
                anomaly_source = "structural"
            elif memory_anomaly:
                anomaly_source = "mixed"
            else:
                anomaly_source = None
            parallel_logical_anomaly = (
                True if rule_failed or composition_anomaly else logical_anomaly
            )
            return InferenceReport(
                image_anomaly=image_anomaly,
                image_anomaly_score=image_score,
                logical_anomaly=parallel_logical_anomaly,
                anomaly_source=anomaly_source,
                categories=categories,
                mask_counts={key: len(value) for key, value in mask_pool.items()},
                results=results,
                filled_template=serialize_template(filled_rules, include_values=True),
                patch_evidence=patch_evidence,
                composition_evidence=composition_evidence,
                execution_mode=execution_mode,
                timings_seconds=timings,
            )

        if logical_anomaly is None:
            return InferenceReport(
                image_anomaly=None,
                image_anomaly_score=rule_score,
                logical_anomaly=None,
                anomaly_source=None,
                categories=categories,
                mask_counts={key: len(value) for key, value in mask_pool.items()},
                results=results,
                filled_template=serialize_template(filled_rules, include_values=True),
                patch_evidence=patch_evidence,
                composition_evidence=composition_evidence,
                execution_mode=execution_mode,
                timings_seconds=timings,
            )

        if rule_failed:
            image_score = (
                rule_fusion_score if rule_fusion_score is not None else rule_score
            )
            image_anomaly = True
            logical_anomaly = True
            anomaly_source = "rule"
        elif patch_evidence is None and composition_evidence is None:
            image_score = rule_fusion_score
            image_anomaly = logical_anomaly
            anomaly_source = None
        else:
            assert memory_score is not None
            image_score = memory_score
            image_anomaly = image_score > FUSION_DECISION_BOUNDARY
            logical_anomaly = composition_anomaly
            if composition_anomaly and structural_anomaly:
                anomaly_source = "mixed"
            elif composition_anomaly:
                anomaly_source = "composition"
            elif structural_anomaly:
                anomaly_source = "structural"
            elif image_anomaly:
                anomaly_source = "mixed"
            else:
                anomaly_source = None

        if not parallel_execution:
            execution_mode = (
                "rule_binary_dual_post"
                if composition_evidence is not None
                else (
                    "rule_binary_patch"
                    if patch_evidence is not None
                    else "rule_binary_only"
                )
            )
        return InferenceReport(
            image_anomaly=image_anomaly,
            image_anomaly_score=image_score,
            logical_anomaly=logical_anomaly,
            anomaly_source=anomaly_source,
            categories=categories,
            mask_counts={key: len(value) for key, value in mask_pool.items()},
            results=results,
            filled_template=serialize_template(filled_rules, include_values=True),
            patch_evidence=patch_evidence,
            composition_evidence=composition_evidence,
            execution_mode=execution_mode,
            timings_seconds=timings,
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

    @staticmethod
    def _binary_result(
        generic: Rule,
        *,
        actual: Any,
        expected: Any,
        satisfied: bool,
        details: Mapping[str, Any] | None = None,
    ) -> RuleResult:
        """Build a hard-decision result without continuous evidence scoring."""

        payload = {"score_method": "binary_hard_decision"}
        if details:
            payload.update(details)
        return RuleResult(
            rule_id=generic.rule_id,
            task=generic.task,
            satisfied=satisfied,
            actual=actual,
            expected=expected,
            satisfaction_score=1.0 if satisfied else 0.0,
            violation_score=0.0 if satisfied else 1.0,
            details=payload,
        )

    def _execute_binary_rule(
        self,
        *,
        image: Any,
        generic: Rule,
        standard: Rule,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        spatial_rules: Mapping[tuple[str, str], Sequence[tuple[Rule, Rule]]],
    ) -> tuple[RuleResult, Rule]:
        """Execute one rule using only the minimum hard pass/fail evidence."""
        if generic.task is TaskType.COUNT:
            actual = len(mask_pool[generic.subject])
            if not isinstance(standard.expected, tuple) or not all(
                (
                    isinstance(item, int) and (not isinstance(item, bool))
                    for item in standard.expected
                )
            ):
                raise ExecutionError(
                    f"{standard.rule_id}: COUNT VALUE must be an integer array"
                )
            result = self._binary_result(
                generic,
                actual=actual,
                expected=list(standard.expected),
                satisfied=actual in standard.expected,
            )
            return (result, replace(generic, expected=(actual,)))
        if generic.task in {TaskType.LENGTH, TaskType.AREA}:
            return self._binary_measurement_range_result(
                generic=generic, standard=standard, instances=mask_pool[generic.subject]
            )
        if generic.task is TaskType.SPATIAL_COMBINATION:
            observed, satisfied, actual, expected, details = (
                self._binary_spatial_observation(generic, standard, mask_pool)
            )
            result = self._binary_result(
                generic,
                actual=actual,
                expected=expected,
                satisfied=satisfied,
                details=details,
            )
            if standard.count is not None:
                return (
                    result,
                    replace(
                        generic,
                        expected=str(details["relation"]),
                        count=(int(details["subject_relation_count"]),),
                    ),
                )
            return (result, replace(generic, expected=observed))
        if generic.task is TaskType.ATTRIBUTE_COMBINATION:
            if isinstance(standard.expected, tuple):
                allowed = standard.expected
                if not allowed:
                    raise ExecutionError(
                        f"{standard.rule_id}: attribute combination has no allowed VALUE"
                    )
                allowed_labels = [value.split(":") for value in allowed]
                actual_attributes = []
                for position, item in enumerate(generic.attributes):
                    expected_values = tuple(
                        (labels[position] for labels in allowed_labels)
                    )
                    actual_value = self._binary_attribute_value(
                        image=image,
                        subject=item.subject,
                        property_name=item.property,
                        expected_values=expected_values,
                        mask_pool=mask_pool,
                        spatial_rules=spatial_rules,
                        rule_id=standard.rule_id,
                    )
                    actual_attributes.append(replace(item, expected=actual_value))
                combined = ":".join(
                    (
                        str(item.expected) if item.expected is not None else ""
                        for item in actual_attributes
                    )
                )
                actual = {
                    f"VALUE_{item.index}": item.expected for item in actual_attributes
                }
                actual["VALUE"] = combined
                result = self._binary_result(
                    generic,
                    actual=actual,
                    expected=self._expected_view(standard),
                    satisfied=combined in allowed,
                )
                return (
                    result,
                    replace(
                        generic, expected=combined, attributes=tuple(actual_attributes)
                    ),
                )
            actual_attributes = []
            for item, expected_item in zip(generic.attributes, standard.attributes):
                actual_value = self._binary_attribute_value(
                    image=image,
                    subject=item.subject,
                    property_name=item.property,
                    expected_values=(expected_item.expected,),
                    mask_pool=mask_pool,
                    spatial_rules=spatial_rules,
                    rule_id=standard.rule_id,
                )
                actual_attributes.append(replace(item, expected=actual_value))
            combined = "+".join((str(item.expected) for item in actual_attributes))
            actual = {
                f"VALUE_{item.index}": item.expected for item in actual_attributes
            }
            actual["VALUE"] = combined
            expected = self._expected_view(standard)
            result = self._binary_result(
                generic, actual=actual, expected=expected, satisfied=actual == expected
            )
            return (
                result,
                replace(
                    generic, expected=combined, attributes=tuple(actual_attributes)
                ),
            )
        raise ExecutionError(f"no executor registered for task {generic.task.value}")

    def _binary_measurement_range_result(
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
            raise ExecutionError(
                f"{standard.rule_id}: RANGE rule has no VALUE or COUNT"
            )
        measure = "length" if generic.task is TaskType.LENGTH else "area"
        measure_fn = mask_length if measure == "length" else mask_area
        measurements = [measure_fn(instance) for instance in instances]
        actual_count = sum(
            standard.minimum <= value <= standard.maximum for value in measurements
        )
        result = self._binary_result(
            generic,
            actual={
                "measurements": measurements,
                "matching_count": actual_count,
            },
            expected=self._expected_view(standard),
            satisfied=actual_count in standard.count,
            details={"measurement": measure},
        )
        return result, replace(
            generic,
            minimum=standard.minimum,
            maximum=standard.maximum,
            count=(actual_count,),
        )

    def _binary_spatial_observation(
        self,
        generic: Rule,
        standard: Rule,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> tuple[Any, bool, Any, Any, dict[str, Any]]:
        if isinstance(standard.expected, tuple) or standard.expected is None:
            raise ExecutionError(
                f"{standard.rule_id}: spatial VALUE must name one relation"
            )
        legacy_boolean = isinstance(standard.expected, bool)
        relation = generic.property if legacy_boolean else str(standard.expected)
        subjects = mask_pool[generic.subject]
        objects = mask_pool[generic.object] if generic.object else ()
        if standard.count is not None:
            if legacy_boolean:
                raise ExecutionError(
                    f"{standard.rule_id}: counted spatial rules require a named relation VALUE"
                )
            matches = evaluate_spatial_relation_by_subject(
                subjects, objects, relation, config=self.spatial_config
            )
            actual_count = sum(matches)
            satisfied = (
                actual_count >= min(standard.count)
                if standard.op == "GE"
                else actual_count in standard.count
            )
            expected = {"relation": relation, "count": list(standard.count)}
            if standard.op == "GE":
                expected["operator"] = "GE"
            details = {
                "relation": relation,
                "subject_relation_count": actual_count,
                "subject_relation_matches": list(matches),
            }
            observed = relation if satisfied else f"not_{relation}"
            return (
                observed,
                satisfied,
                {"relation": relation, "count": actual_count},
                expected,
                details,
            )
        relation_holds = evaluate_spatial_relation(
            subjects, objects, relation, config=self.spatial_config
        )
        observed = (
            relation_holds
            if legacy_boolean
            else relation if relation_holds else f"not_{relation}"
        )
        return (
            observed,
            observed == standard.expected,
            observed,
            standard.expected,
            {"relation": relation},
        )

    def _binary_attribute_value(
        self,
        *,
        image: Any,
        subject: str,
        property_name: str,
        expected_values: Sequence[object],
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        spatial_rules: Mapping[tuple[str, str], Sequence[tuple[Rule, Rule]]],
        rule_id: str,
    ) -> Any:
        if self._is_count_attribute(property_name):
            for value in expected_values:
                self._parse_attribute_count(value, rule_id=rule_id)
            return len(mask_pool[subject])

        spatial_object = self._spatial_attribute_object(property_name)
        if spatial_object is not None:
            matches = tuple(spatial_rules.get((subject, spatial_object), ()))
            if not matches:
                raise ExecutionError(
                    f"{rule_id}: attribute.spatial_combination.{spatial_object} "
                    f"needs one SPATIAL_COMBINATION rule with SUBJECT={subject} "
                    f"and OBJECT={spatial_object}"
                )
            if len(matches) > 1:
                raise ExecutionError(
                    f"{rule_id}: attribute.spatial_combination.{spatial_object} "
                    f"is ambiguous because {len(matches)} matching "
                    "SPATIAL_COMBINATION rules exist"
                )
            spatial_generic, spatial_standard = matches[0]
            observed, _, _, _, _ = self._binary_spatial_observation(
                spatial_generic,
                spatial_standard,
                mask_pool,
            )
            if isinstance(spatial_standard.expected, bool) and all(
                isinstance(value, str) for value in expected_values
            ):
                return "true" if observed else "false"
            return observed

        measurement = self._measurement_attribute(property_name)
        if measurement is not None:
            instances = mask_pool[subject]
            if len(instances) != 1:
                raise ExecutionError(
                    f"{rule_id}: attribute.{measurement} requires exactly one "
                    f"segmented instance, found {len(instances)}"
                )
            measured = (
                mask_length(instances[0])
                if measurement == "length"
                else mask_area(instances[0])
            )
            for value in expected_values:
                expected = self._parse_attribute_measurement(
                    value,
                    rule_id=rule_id,
                )
                if isclose(measured, expected, rel_tol=1e-9, abs_tol=1e-9):
                    return value
            return int(measured) if measured.is_integer() else measured

        prediction = self.attribute_detector.detect(
            image,
            subject,
            property_name,
            mask_pool[subject],
        )
        return prediction.value

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
