"""Generate CQT and NRT from descriptions, reference images and a UTS."""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, replace
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image

from ..errors import TemplateError
from ..models import Rule, TaskType
from ..templates import parse_template, serialize_template, validate_template_alignment
from ..vlm import (
    VLMError, VisionImage, VisionLanguageClient, VisionMessage, VisionRequest, create_vlm_client,
)
from .errors import TemplateGenerationError
from .prompts import (
    DESCRIPTION_CHECK, DESCRIPTION_PROMPT, REPAIR_PROMPT, SYSTEM_PROMPT, USER_PROMPT,
)
from .uts import UTSInput, load_uts, validate_generated_rules

ImageInput = str | Path | Image.Image
ImagesInput = ImageInput | Sequence[ImageInput]
_MAX_ATTEMPTS = 3
_DESCRIPTION_RULE = re.compile(r"^\s*\[[A-Za-z][A-Za-z0-9_-]*\]\s*TASK\s*=")


@dataclass(frozen=True)
class GeneratedTemplates:
    """Only the two canonical rule templates are exposed as generation output."""

    cqt: str
    nrt: str

    def as_dict(self) -> dict[str, str]:
        return {"cqt": self.cqt, "nrt": self.nrt}


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"cqt": {"type": "string"}, "nrt": {"type": "string"}},
        "required": ["cqt", "nrt"],
        "additionalProperties": False,
    }


def _encode_images(images: ImagesInput) -> list[str | VisionImage]:
    if isinstance(images, (str, Path, Image.Image)):
        items = [images]
    else:
        try:
            items = list(images)
        except TypeError as exc:
            raise TemplateError("images must be an image path, PIL image, or a sequence of them") from exc
    if not items:
        raise TemplateError("provide at least one normal reference image")
    content: list[str | VisionImage] = []
    for index, item in enumerate(items, 1):
        opened = None
        try:
            if isinstance(item, Image.Image):
                image = item
            elif isinstance(item, (str, Path)):
                opened = Image.open(Path(item).expanduser())
                image = opened
            else:
                raise TemplateError(f"image {index} must be a local path or PIL image")
            if getattr(image, "n_frames", 1) != 1:
                raise TemplateError(f"image {index} must be a single still image")
            width, height = image.size
            if width < 1 or height < 1:
                raise TemplateError(f"image {index} must have positive dimensions")
            # Preserve native pixels, not EXIF orientation or automatic resizing.
            # PNG avoids introducing JPEG artifacts around tiny parts and icons.
            with BytesIO() as buffer:
                with image.convert("RGB") as rgb:
                    rgb.info.clear()
                    rgb.save(buffer, format="PNG")
                data = base64.b64encode(buffer.getvalue()).decode("ascii")
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
            if isinstance(exc, TemplateError):
                raise
            raise TemplateError(f"image {index} is not a readable still image") from exc
        finally:
            if opened is not None:
                opened.close()
        content.extend([
            f"Normal reference {index}; native width={width}, height={height}.",
            VisionImage(data=data, width=width, height=height),
        ])
    return content


def _read_description(description: str | Path) -> str:
    if isinstance(description, Path):
        try:
            description = description.expanduser().read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise TemplateError("description file must be readable UTF-8 text") from exc
    if not isinstance(description, str) or not description.strip():
        raise TemplateError("description must be nonempty text")
    return description


def _description_rules(description: str, uts: Mapping[str, Any]) -> tuple[Rule, ...]:
    lines = [line for line in description.splitlines() if _DESCRIPTION_RULE.match(line)]
    if not lines:
        return ()
    try:
        rules = parse_template("\n".join(lines), allow_unknown_values=True)
        required = tuple(
            replace(rule, minimum="unknown", maximum="unknown")
            if rule.task in {TaskType.LENGTH, TaskType.AREA}
            else rule
            for rule in rules
        )
        validate_generated_rules(
            tuple(replace(rule, rule_id=f"C{i:03d}") for i, rule in enumerate(required, 1)),
            uts,
        )
    except (TemplateError, ValueError, OverflowError) as exc:
        raise TemplateError(f"invalid rule in description: {exc}") from exc
    return required


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TemplateError("response JSON must not contain duplicate keys")
        result[key] = value
    return result


def _validate_response(
    text: str, uts: Mapping[str, Any], required_rules: Sequence[Rule]
) -> GeneratedTemplates:
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except TemplateError:
        raise
    except (TypeError, ValueError) as exc:
        raise TemplateError("response must be a JSON object containing cqt and nrt") from exc
    if (not isinstance(data, dict) or set(data) != {"cqt", "nrt"}
            or any(not isinstance(value, str) or not value.strip() for value in data.values())):
        raise TemplateError("response must contain only nonempty cqt and nrt strings")
    # Comments are accepted by the general parser but have no place in this output.
    if any(line.lstrip().startswith("#") for value in data.values() for line in value.splitlines()):
        raise TemplateError("generated templates must not contain comments")
    try:
        cqt = parse_template(data["cqt"], allow_missing_values=True)
        nrt = parse_template(
            data["nrt"], allow_missing_values=False, allow_unknown_values=True
        )
        validate_template_alignment(cqt, nrt)
        validate_generated_rules(nrt, uts)
        if required_rules:
            available = Counter(replace(rule, rule_id="") for rule in nrt)
            for rule in required_rules:
                signature = replace(rule, rule_id="")
                if available[signature] == 0:
                    raise TemplateError(
                        f"description rule {rule.rule_id} is missing or changed in the NRT; "
                        "preserve its structure and values except for deferred measurement bounds"
                    )
                available[signature] -= 1
    except TemplateError:
        raise
    except (ValueError, OverflowError) as exc:
        raise TemplateError("generated rule values could not be parsed") from exc
    return GeneratedTemplates(cqt=serialize_template(cqt), nrt=serialize_template(nrt))


class TemplateGenerator:
    """Fixed-prompt generator; an injected client is useful for offline testing."""

    def __init__(self, client: VisionLanguageClient | None = None) -> None:
        self._client = client

    def generate(
        self, images: ImagesInput, uts: UTSInput, description: str | Path | None = None
    ) -> GeneratedTemplates:
        definition = load_uts(uts)
        required_rules: tuple[Rule, ...] = ()
        instruction = USER_PROMPT.format(
            uts=json.dumps(definition, ensure_ascii=False, sort_keys=True)
        )
        if description is not None:
            description_text = _read_description(description)
            required_rules = _description_rules(description_text, definition)
            instruction += "\n" + DESCRIPTION_PROMPT.format(
                description=json.dumps(description_text, ensure_ascii=False)
            )
        image_content = _encode_images(images)
        if description is not None:
            image_content.append(DESCRIPTION_CHECK)
        try:
            if self._client is None:
                self._client = create_vlm_client()
        except VLMError as exc:
            raise TemplateGenerationError(str(exc)) from None
        request = VisionRequest(
            system_prompt=SYSTEM_PROMPT,
            messages=(VisionMessage(role="user", content=(
                instruction,
                *image_content,
            )),),
            json_schema=_output_schema(), schema_name="cqt_nrt_pair",
        )
        for attempt in range(_MAX_ATTEMPTS):
            try:
                answer = self._client.complete(request)
            except VLMError as exc:
                raise TemplateGenerationError(str(exc)) from None
            try:
                return _validate_response(answer, definition, required_rules)
            except TemplateError as exc:
                error = str(exc)
                if attempt + 1 == _MAX_ATTEMPTS:
                    raise TemplateGenerationError(
                        f"VLM did not produce a valid CQT/NRT pair after {_MAX_ATTEMPTS} attempts: {error}"
                    ) from None
                # The exact invalid answer is included as assistant data for repair;
                # the fixed system instructions and original references stay intact.
                request = replace(request, messages=(*request.messages,
                    VisionMessage(role="assistant", content=(answer,)),
                    VisionMessage(role="user", content=(REPAIR_PROMPT.format(error=error),)),
                ))
        raise AssertionError("unreachable")


def generate_templates(
    images: ImagesInput, uts: UTSInput, description: str | Path | None = None
) -> GeneratedTemplates:
    """Generate CQT/NRT with an optional authoritative product description."""
    return TemplateGenerator().generate(images, uts, description)
