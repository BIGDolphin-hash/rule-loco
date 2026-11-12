from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from PIL import Image

from pro_innovation.errors import TemplateError
from pro_innovation.vlm import VLMError, VisionImage, VisionMessage, VisionRequest
from pro_innovation.template_generation import (
    GeneratedTemplates, QwenClient, TemplateGenerationError, TemplateGenerator,
)
from pro_innovation.template_generation.cli import main
from pro_innovation.template_generation.prompts import SYSTEM_PROMPT
from pro_innovation.template_generation.uts import load_uts
from pro_innovation.templates import (
    parse_template, serialize_template, to_generic_template, validate_template_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
UTS_PATH = ROOT / "config" / "template_generation" / "uts.v1.json"
NRT = "\n".join([
    "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | OP=EQ | VALUE=[2]",
    "[C002] TASK=LENGTH | SUBJECT=part | PROPERTY=length | OP=RANGE | VALUE=[8,14] | COUNT=[1]",
    "[C003] TASK=AREA | SUBJECT=part | PROPERTY=area | OP=RANGE | VALUE=[30,90] | COUNT=[2]",
    "[C004] TASK=SPATIAL_COMBINATION | SUBJECT=part | PROPERTY=spatial.position | OBJECT=container | OP=EQ | VALUE=inside | COUNT=[2]",
    '[C005] TASK=ATTRIBUTE_COMBINATION | SUBJECT_1=part | PROPERTY_1=attribute.color | SUBJECT_2=container | PROPERTY_2=attribute.shape | OP=EQ | VALUE=["red:square"]',
])


def setUpModule() -> None:
    global _local_config_patch
    _local_config_patch = patch(
        "pro_innovation.vlm.config._DEFAULT_CONFIG_PATH",
        Path("/tmp/rulead-tests-no-local-config.json"),
    )
    _local_config_patch.start()


def tearDownModule() -> None:
    _local_config_patch.stop()


def pair(nrt: str = NRT) -> dict[str, str]:
    return {
        "cqt": serialize_template(to_generic_template(parse_template(nrt))),
        "nrt": nrt,
    }


def api_response(content: str, finish_reason: str = "stop") -> BytesIO:
    return BytesIO(json.dumps({"choices": [{
        "finish_reason": finish_reason, "message": {"content": content},
    }]}).encode())


def empty_request() -> VisionRequest:
    return VisionRequest(
        system_prompt="Return JSON.",
        messages=(VisionMessage(role="user", content=("test",)),),
        json_schema={"type": "object"},
    )


class ScriptedClient:
    def __init__(self, *answers: str) -> None:
        self.answers = iter(answers)
        self.requests = []

    def complete(self, request):
        self.requests.append(deepcopy(request))
        return next(self.answers)


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (64, 48), "white")
        self.addCleanup(self.image.close)

    def generate(self, output, uts=UTS_PATH):
        answer = json.dumps(output)
        return TemplateGenerator(ScriptedClient(answer, answer, answer)).generate(self.image, uts)

    def test_full_image_uts_http_output_contract(self) -> None:
        opener = Mock()
        opener.open.return_value = api_response(json.dumps(pair()))
        with patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
            result = TemplateGenerator(QwenClient("test-secret")).generate(self.image, UTS_PATH)
        self.assertEqual(set(result.as_dict()), {"cqt", "nrt"})
        self.assertEqual(result.nrt, NRT)
        validate_template_alignment(
            parse_template(result.cqt, allow_missing_values=True), parse_template(result.nrt)
        )
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3.8-max")
        self.assertEqual(payload["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertTrue(payload["vl_high_resolution_images"])
        self.assertFalse(payload["enable_thinking"])
        self.assertFalse(payload["stream"])
        self.assertNotIn("max_tokens", payload)
        schema = payload["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        content = payload["messages"][1]["content"]
        self.assertIn('"version": 1', content[0]["text"])
        self.assertIn("width=64, height=48", content[1]["text"])
        self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_repair_retains_images_and_uts_and_checks_both_templates(self) -> None:
        bad = pair()
        bad["cqt"] = bad["cqt"].replace("SUBJECT=part", "SUBJECT=wrong", 1)
        client = ScriptedClient(json.dumps(bad), json.dumps(pair()))
        result = TemplateGenerator(client).generate([self.image, self.image], UTS_PATH)
        self.assertEqual(result.nrt, NRT)
        self.assertEqual(len(client.requests), 2)
        first, second = client.requests
        self.assertEqual(second.messages[:1], first.messages)
        self.assertEqual(second.system_prompt, first.system_prompt)
        self.assertEqual(second.messages[1].role, "assistant")
        self.assertIn("structures differ", second.messages[2].content[0])
        self.assertEqual(sum(isinstance(item, VisionImage) for item in first.messages[0].content), 2)

    def test_invalid_pair_never_returns_partial_result_and_retries_are_bounded(self) -> None:
        client = ScriptedClient("{}", "{}", "{}")
        with self.assertRaisesRegex(TemplateGenerationError, "after 3 attempts"):
            TemplateGenerator(client).generate(self.image, UTS_PATH)
        self.assertEqual(len(client.requests), 3)

    def test_unparseable_numeric_values_trigger_repair_instead_of_escaping(self) -> None:
        bad = pair()
        bad["nrt"] = NRT.replace("VALUE=[2]", "VALUE=[" + "9" * 5000 + "]", 1)
        client = ScriptedClient(json.dumps(bad), json.dumps(pair()))
        result = TemplateGenerator(client).generate(self.image, UTS_PATH)
        self.assertEqual(result.nrt, NRT)
        self.assertEqual(len(client.requests), 2)

    def test_rejects_bad_json_extra_output_cqt_values_and_misalignment(self) -> None:
        cases = []
        cases.append({**pair(), "iot": "unexpected"})
        cases.append({**pair(), "cqt": NRT})
        cases.append({**pair(), "nrt": ""})
        cases.append({**pair(), "cqt": pair()["cqt"].replace("C001", "C999")})
        cases.append({**pair(), "nrt": NRT.replace("VALUE=[2]", "VALUE=[-1]", 1)})
        cases.append({**pair(), "nrt": "# comment\n" + NRT})
        for output in cases:
            with self.subTest(output=output), self.assertRaises(TemplateGenerationError):
                self.generate(output)
        for answer in ("```json\n{}\n```", '{"cqt":"a","cqt":"b","nrt":"c"}', "null"):
            with self.subTest(answer=answer), self.assertRaises(TemplateGenerationError):
                TemplateGenerator(ScriptedClient(answer, answer, answer)).generate(self.image, UTS_PATH)

    def test_uts_subsets_restrict_tasks_properties_operators_and_relations(self) -> None:
        for mutate in (
            lambda uts: uts["tasks"].pop(1),
            lambda uts: uts["tasks"][3].update(relations=["left_of"]),
            lambda uts: uts["tasks"][3].update(operators=["GE"]),
            lambda uts: uts["tasks"][4].update(properties=["attribute.type"]),
        ):
            uts = load_uts(UTS_PATH)
            mutate(uts)
            with self.subTest(uts=uts), self.assertRaises(TemplateGenerationError):
                self.generate(pair(), uts)
        count_only = load_uts(UTS_PATH)
        count_only["tasks"] = count_only["tasks"][:1]
        self.assertEqual(self.generate(pair(NRT.splitlines()[0]), count_only).nrt, NRT.splitlines()[0])

    def test_measurements_cannot_exceed_native_dimensions_or_match_impossible_counts(self) -> None:
        for old, new in (("[8,14]", "[8,1000]"), ("[30,90]", "[30,4000]"), ("COUNT=[1]", "COUNT=[3]")):
            with self.subTest(new=new), self.assertRaises(TemplateGenerationError):
                self.generate(pair(NRT.replace(old, new)))
        with Image.new("RGB", (32, 48)) as other:
            answers = [json.dumps(pair())] * 3
            with self.assertRaisesRegex(TemplateGenerationError, "equal image dimensions"):
                TemplateGenerator(ScriptedClient(*answers)).generate([self.image, other], UTS_PATH)

    def test_validates_spatial_subject_count_and_self_symmetry(self) -> None:
        for nrt in (
            NRT.replace("VALUE=inside | COUNT=[2]", "VALUE=inside | COUNT=[3]"),
            NRT.replace("VALUE=inside", "VALUE=self_left_right_symmetric"),
            NRT.replace("OBJECT=container", "OBJECT=part"),
        ):
            with self.subTest(nrt=nrt), self.assertRaises(TemplateGenerationError):
                self.generate(pair(nrt))
        nrt = "[C001] TASK=SPATIAL_COMBINATION | SUBJECT=part | PROPERTY=spatial.self_symmetry | OP=EQ | VALUE=self_left_right_symmetric | COUNT=[1]"
        self.assertEqual(self.generate(pair(nrt)).nrt, nrt)

    def test_attribute_tuples_must_have_multiple_targets_and_supported_labels(self) -> None:
        for nrt in (
            NRT.replace("red:square", "red:triangle"),
            NRT.replace("red:square", "cerulean:square"),
            NRT.replace("red:square", "Red:square"),
            NRT.replace("PROPERTY_1=attribute.color", "PROPERTY_1=attribute.count").replace("red:square", "3:square"),
            '[C001] TASK=ATTRIBUTE_COMBINATION | SUBJECT_1=part | PROPERTY_1=attribute.color | OP=EQ | VALUE=["red"]',
        ):
            with self.subTest(nrt=nrt), self.assertRaises(TemplateGenerationError):
                self.generate(pair(nrt))

    def test_missing_key_and_bad_input_fail_before_any_request(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(TemplateGenerationError, "DASHSCOPE_API_KEY"):
                TemplateGenerator().generate(self.image, UTS_PATH)
        client = ScriptedClient()
        for images in ([], ["/tmp/rulead-definitely-missing-image.png"], [123]):
            with self.subTest(images=images), self.assertRaises(TemplateError):
                TemplateGenerator(client).generate(images, UTS_PATH)
        self.assertEqual(client.requests, [])

    def test_paths_and_pil_images_produce_the_same_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal.png"
            self.image.save(path)
            client = ScriptedClient(json.dumps(pair()))
            result = TemplateGenerator(client).generate(path, json.dumps(load_uts(UTS_PATH)))
            self.assertEqual(result.as_dict(), pair())
            self.assertNotIn(str(path), repr(client.requests))

    def test_cli_stdout_contains_only_cqt_and_nrt(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        result = GeneratedTemplates(**pair())
        with patch("pro_innovation.template_generation.cli.generate_templates", return_value=result) as generate:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["--images", "normal.png", "--uts", str(UTS_PATH)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), pair())
        self.assertEqual(stderr.getvalue(), "")
        generate.assert_called_once_with(["normal.png"], UTS_PATH)


class UTSAndTransportTests(unittest.TestCase):
    def test_invalid_uts_is_rejected_locally(self) -> None:
        for value in (
            "plain prose", {}, {"version": True, "tasks": []},
            {"version": 1, "tasks": []}, {"version": 1, "tasks": [{}]},
            {"version": 1, "tasks": [{"task": "invented", "properties": ["x"], "operators": ["EQ"]}]},
        ):
            with self.subTest(value=value), self.assertRaises(TemplateError):
                load_uts(value)
        uts = load_uts(UTS_PATH)
        uts["tasks"][1]["unit"] = "mm"
        with self.assertRaisesRegex(TemplateError, "unit=pixel"):
            load_uts(uts)

    def test_configuration_uses_env_without_exposing_key(self) -> None:
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-secret", "DASHSCOPE_BASE_URL": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"}):
            client = QwenClient.from_env()
        self.assertNotIn("test-secret", repr(client))
        self.assertIn("workspace.cn-beijing", client.config.base_url)
        for url in ("http://example.com/v1", "https://key@example.com/v1", "https://example.com/v1?key=secret"):
            with self.subTest(url=url), self.assertRaises(VLMError):
                QwenClient("test-secret", base_url=url)

    def test_http_auth_failure_is_not_retried_and_body_is_not_exposed(self) -> None:
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example.com", 401, "test-secret", {}, BytesIO(b"test-secret"))
        with patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
            with self.assertRaises(VLMError) as caught:
                QwenClient("test-secret").complete(empty_request())
        self.assertEqual(opener.open.call_count, 1)
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))

    def test_rate_limit_retry_then_success(self) -> None:
        opener = Mock()
        opener.open.side_effect = [
            HTTPError("https://example.com", 429, "rate limited", {}, BytesIO()),
            api_response("ok"),
        ]
        with patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
            with patch("pro_innovation.vlm.transport.time.sleep") as sleep:
                self.assertEqual(QwenClient("test-secret").complete(empty_request()), "ok")
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_incomplete_refused_or_malformed_responses_do_not_produce_templates(self) -> None:
        for response in (
            api_response("{}", "length"), api_response("{}", "content_filter"),
            BytesIO(b"not json"), BytesIO(b'{"choices": [null]}'),
            BytesIO(b'{"choices": [{"finish_reason":"stop","message":{"refusal":"declined"}}]}'),
        ):
            opener = Mock()
            opener.open.return_value = response
            with self.subTest(response=response), patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
                with self.assertRaises(VLMError):
                    QwenClient("test-secret").complete(empty_request())

    def test_connection_error_is_not_automatically_resubmitted(self) -> None:
        opener = Mock()
        opener.open.side_effect = URLError("test-secret")
        with patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
            with self.assertRaises(VLMError) as caught:
                QwenClient("test-secret").complete(empty_request())
        self.assertEqual(opener.open.call_count, 1)
        self.assertNotIn("test-secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
