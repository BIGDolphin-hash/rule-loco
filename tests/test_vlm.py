from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from PIL import Image

from pro_innovation.template_generation import TemplateGenerationError, TemplateGenerator
from pro_innovation.templates import parse_template, serialize_template, to_generic_template
from pro_innovation.vlm import (
    AnthropicClient, ChatCompletionsClient, GeminiClient, VLMConfig, VLMError,
    VisionImage, VisionLanguageClient, VisionMessage, VisionRequest, create_vlm_client,
)
from pro_innovation.vlm.transport import JsonTransport

UTS_PATH = Path(__file__).resolve().parents[1] / "config" / "template_generation" / "uts.v1.json"
NRT = "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | OP=EQ | VALUE=[2]"
PAIR = {"cqt": serialize_template(to_generic_template(parse_template(NRT))), "nrt": NRT}


def setUpModule() -> None:
    global _local_config_patch
    _local_config_patch = patch(
        "pro_innovation.vlm.config._DEFAULT_CONFIG_PATH",
        Path("/tmp/rulead-tests-no-local-config.json"),
    )
    _local_config_patch.start()


def tearDownModule() -> None:
    _local_config_patch.stop()


def request() -> VisionRequest:
    return VisionRequest(
        system_prompt="Fixed system instructions.",
        messages=(VisionMessage(role="user", content=(
            "Reference image.", VisionImage(data="YWJj", width=32, height=24),
        )),),
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        schema_name="test_schema",
    )


def model_response(provider: str, text: str) -> dict:
    if provider == "anthropic":
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}
    if provider == "gemini":
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}


def settings(provider: str = "qwen", **kwargs) -> VLMConfig:
    return VLMConfig(provider=provider, model="test-vision-model", api_key="test-secret", **kwargs)


class AdapterTests(unittest.TestCase):
    def test_each_provider_generates_the_same_valid_templates(self) -> None:
        for provider in ("qwen", "openai", "openai-compatible", "anthropic", "gemini"):
            with self.subTest(provider=provider):
                extra = {"base_url": "https://example.com/v1"} if provider == "openai-compatible" else {}
                client = create_vlm_client(settings(provider, **extra))
                self.assertIsInstance(client, VisionLanguageClient)
                client.transport = Mock(post=Mock(return_value=model_response(provider, json.dumps(PAIR))))
                with Image.new("RGB", (64, 48), "white") as image:
                    result = TemplateGenerator(client).generate(image, UTS_PATH)
                self.assertEqual(result.as_dict(), PAIR)
                self.assertEqual(client.transport.post.call_count, 1)

    def test_chat_adapter_is_provider_neutral_and_has_no_qwen_options(self) -> None:
        transport = Mock(post=Mock(return_value=model_response("openai", "answer")))
        client = ChatCompletionsClient(settings("openai"), transport)
        self.assertEqual(client.complete(request()), "answer")
        url, headers, payload = transport.post.call_args.args
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers, {"Authorization": "Bearer test-secret"})
        self.assertEqual(payload["model"], "test-vision-model")
        self.assertEqual(payload["messages"][0]["content"], request().system_prompt)
        image = payload["messages"][1]["content"][1]
        self.assertEqual(image["image_url"]["url"], "data:image/png;base64,YWJj")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], request().json_schema)
        for name in ("enable_thinking", "temperature", "vl_high_resolution_images", "max_tokens"):
            self.assertNotIn(name, payload)

    def test_qwen_settings_are_scoped_to_its_adapter(self) -> None:
        transport = Mock(post=Mock(return_value=model_response("qwen", "answer")))
        client = ChatCompletionsClient(settings(request_options={"enable_thinking": True}), transport)
        client.complete(request())
        payload = transport.post.call_args.args[2]
        self.assertTrue(payload["enable_thinking"])
        self.assertTrue(payload["vl_high_resolution_images"])
        self.assertEqual(payload["temperature"], 0)

    def test_anthropic_serializes_native_images_schema_and_repair_history(self) -> None:
        transport = Mock(post=Mock(return_value=model_response("anthropic", "answer")))
        client = AnthropicClient(settings("anthropic", max_output_tokens=2048), transport)
        original = request()
        repaired = replace(original, messages=(*original.messages,
            VisionMessage("assistant", ("invalid answer",)), VisionMessage("user", ("fix it",)),
        ))
        self.assertEqual(client.complete(repaired), "answer")
        url, headers, payload = transport.post.call_args.args
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "test-secret")
        self.assertEqual(headers["anthropic-version"], "2026-10-21")
        self.assertEqual(payload["system"], original.system_prompt)
        self.assertEqual(payload["messages"][0]["content"][1], {
            "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"},
        })
        self.assertEqual([message["role"] for message in payload["messages"]], ["user", "assistant", "user"])
        self.assertEqual(payload["output_config"]["format"]["schema"], original.json_schema)
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertNotIn("response_format", payload)

    def test_gemini_serializes_native_images_schema_and_repair_history(self) -> None:
        transport = Mock(post=Mock(return_value=model_response("gemini", "answer")))
        client = GeminiClient(settings("gemini", max_output_tokens=2048), transport)
        original = request()
        repaired = replace(original, messages=(*original.messages,
            VisionMessage("assistant", ("invalid answer",)), VisionMessage("user", ("fix it",)),
        ))
        self.assertEqual(client.complete(repaired), "answer")
        url, headers, payload = transport.post.call_args.args
        self.assertTrue(url.endswith("/models/test-vision-model:generateContent"))
        self.assertNotIn("test-secret", url)
        self.assertEqual(headers, {"x-goog-api-key": "test-secret"})
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], original.system_prompt)
        self.assertEqual(payload["contents"][0]["parts"][1], {"inlineData": {
            "mimeType": "image/png", "data": "YWJj",
        }})
        self.assertEqual([message["role"] for message in payload["contents"]], ["user", "model", "user"])
        self.assertEqual(payload["generationConfig"]["responseJsonSchema"], original.json_schema)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 2048)

    def test_output_modes_are_explicit_and_keep_local_validation(self) -> None:
        for mode in ("json_schema", "json_object", "prompt"):
            transport = Mock(post=Mock(return_value=model_response("openai", "answer")))
            client = ChatCompletionsClient(settings("openai", structured_output=mode), transport)
            client.complete(request())
            payload = transport.post.call_args.args[2]
            if mode == "prompt":
                self.assertNotIn("response_format", payload)
            else:
                self.assertEqual(payload["response_format"]["type"], mode)
        transport = Mock(post=Mock(return_value=model_response("qwen", '{"cqt":"bad","nrt":"bad"}')))
        client = ChatCompletionsClient(settings(structured_output="prompt"), transport)
        with Image.new("RGB", (64, 48)) as image:
            with self.assertRaises(TemplateGenerationError):
                TemplateGenerator(client).generate(image, UTS_PATH)
        self.assertEqual(transport.post.call_count, 3)

    def test_repair_roundtrip_works_with_native_providers(self) -> None:
        for provider in ("anthropic", "gemini"):
            with self.subTest(provider=provider):
                client = create_vlm_client(settings(provider))
                client.transport = Mock(post=Mock(side_effect=[
                    model_response(provider, "{}"), model_response(provider, json.dumps(PAIR)),
                ]))
                with Image.new("RGB", (64, 48)) as image:
                    self.assertEqual(TemplateGenerator(client).generate(image, UTS_PATH).as_dict(), PAIR)
                first = client.transport.post.call_args_list[0].args[2]
                second = client.transport.post.call_args_list[1].args[2]
                key = "contents" if provider == "gemini" else "messages"
                self.assertEqual(second[key][:1], first[key])
                self.assertEqual(len(second[key]), 3)

    def test_native_responses_do_not_leak_thinking(self) -> None:
        anthropic = AnthropicClient(settings("anthropic"), Mock(post=Mock(return_value={
            "stop_reason": "end_turn", "content": [
                {"type": "thinking", "thinking": "private"}, {"type": "text", "text": "answer"},
            ],
        })))
        gemini = GeminiClient(settings("gemini"), Mock(post=Mock(return_value={
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "private", "thought": True}, {"text": "answer"},
            ]}}],
        })))
        for client in (anthropic, gemini):
            with self.subTest(client=type(client).__name__):
                self.assertEqual(client.complete(request()), "answer")

    def test_invalid_blocked_truncated_and_tool_responses_are_rejected(self) -> None:
        for provider, outputs in (
            ("openai", [{}, {"choices": []}, {"choices": [None]},
                {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
                {"choices": [{"finish_reason": "stop", "message": {"content": [{"type": "refusal"}]}}]}]),
            ("anthropic", [{}, {"stop_reason": "refusal"}, {"stop_reason": "max_tokens"},
                {"stop_reason": "end_turn", "content": [None]},
                {"stop_reason": "end_turn", "content": [{"type": "tool_use"}]},
                {"stop_reason": "end_turn", "content": [{"type": "text", "text": 123}]}]),
            ("gemini", [{"promptFeedback": {"blockReason": "SAFETY"}}, {"candidates": []},
                {"candidates": [None]}, {"candidates": [{"finishReason": "MAX_TOKENS"}]},
                {"candidates": [{"finishReason": "STOP", "content": {"parts": [None]}}]},
                {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"functionCall": {}}]}}]}]),
        ):
            for output in outputs:
                with self.subTest(provider=provider, output=output):
                    client = create_vlm_client(settings(provider))
                    client.transport = Mock(post=Mock(return_value=output))
                    with self.assertRaises(VLMError):
                        client.complete(request())

    def test_provider_limits_are_not_baked_into_generator(self) -> None:
        oversized = replace(request().messages[0].content[1], width=5000, height=5000)
        value = replace(request(), messages=(VisionMessage("user", (oversized,)),))
        transport = Mock(post=Mock(return_value=model_response("openai", "answer")))
        with self.assertRaisesRegex(VLMError, "16M"):
            ChatCompletionsClient(settings(), transport).complete(value)
        transport.post.assert_not_called()
        self.assertEqual(ChatCompletionsClient(settings("openai"), transport).complete(value), "answer")

    def test_custom_client_needs_only_the_public_interface(self) -> None:
        class CustomClient:
            def complete(self, value: VisionRequest) -> str:
                self.last_request = value
                return json.dumps(PAIR)
        client = CustomClient()
        with Image.new("RGB", (64, 48)) as image:
            self.assertEqual(TemplateGenerator(client).generate(image, UTS_PATH).as_dict(), PAIR)
        self.assertIsInstance(client.last_request, VisionRequest)
        self.assertEqual(set(client.last_request.json_schema["properties"]), {"cqt", "nrt"})

    def test_mismatched_client_and_provider_are_rejected_locally(self) -> None:
        for client, provider in ((ChatCompletionsClient, "anthropic"),
                                 (AnthropicClient, "gemini"), (GeminiClient, "qwen")):
            with self.subTest(client=client.__name__), self.assertRaisesRegex(VLMError, "protocol"):
                client(settings(provider))

    def test_generator_reuses_client_and_starts_fresh_conversations(self) -> None:
        client = Mock(complete=Mock(return_value=json.dumps(PAIR)))
        with patch("pro_innovation.template_generation.generator.create_vlm_client", return_value=client) as factory:
            generator = TemplateGenerator()
            with Image.new("RGB", (64, 48)) as image:
                generator.generate(image, UTS_PATH)
                generator.generate(image, UTS_PATH)
        factory.assert_called_once_with()
        self.assertEqual([len(call.args[0].messages) for call in client.complete.call_args_list], [1, 1])


class ConfigurationAndTransportTests(unittest.TestCase):
    def test_private_file_is_loaded_and_environment_overrides_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vlm.local.json"
            path.write_text(json.dumps({
                "provider": "qwen", "model": "saved-model", "api_key": "saved-secret",
                "base_url": "https://workspace.example.com/v1",
            }))
            with patch.dict("os.environ", {"VLM_CONFIG_FILE": str(path)}, clear=True):
                config = VLMConfig.from_env()
                self.assertEqual(config.model, "saved-model")
                self.assertEqual(config.api_key, "saved-secret")
                self.assertNotIn("saved-secret", repr(config))
            with patch.dict("os.environ", {
                "VLM_CONFIG_FILE": str(path), "DASHSCOPE_API_KEY": "override-secret",
                "VLM_MODEL": "override-model",
            }, clear=True):
                config = VLMConfig.from_env()
                self.assertEqual(config.model, "override-model")
                self.assertEqual(config.api_key, "override-secret")
            with patch.dict("os.environ", {
                "VLM_CONFIG_FILE": str(path), "VLM_PROVIDER": "openai", "VLM_MODEL": "vision-model",
            }, clear=True), self.assertRaisesRegex(VLMError, "OPENAI_API_KEY"):
                VLMConfig.from_env()

    def test_config_file_can_be_disabled_and_errors_hide_its_contents(self) -> None:
        with patch.dict("os.environ", {"VLM_CONFIG_FILE": ""}, clear=True):
            with self.assertRaisesRegex(VLMError, "DASHSCOPE_API_KEY"):
                VLMConfig.from_env()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"api_key": "private-secret", invalid}')
            with patch.dict("os.environ", {"VLM_CONFIG_FILE": str(path)}, clear=True):
                with self.assertRaises(VLMError) as caught:
                    VLMConfig.from_env()
            self.assertNotIn("private-secret", str(caught.exception))

    def test_env_selects_provider_without_cross_provider_key_fallback(self) -> None:
        for provider, key in (("qwen", "DASHSCOPE_API_KEY"), ("openai", "OPENAI_API_KEY"),
                              ("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")):
            with self.subTest(provider=provider), patch.dict("os.environ", {
                "VLM_PROVIDER": provider, "VLM_MODEL": "selected-model", key: "selected-secret",
                "DASHSCOPE_BASE_URL": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            }, clear=True):
                config = VLMConfig.from_env()
                self.assertEqual(config.api_key, "selected-secret")
                self.assertEqual(config.model, "selected-model")
                self.assertNotIn("selected-secret", repr(config))
                if provider != "qwen":
                    self.assertNotIn("workspace", config.base_url)
        with patch.dict("os.environ", {
            "VLM_PROVIDER": "openai", "VLM_MODEL": "selected-model", "DASHSCOPE_API_KEY": "qwen-only",
        }, clear=True), self.assertRaisesRegex(VLMError, "OPENAI_API_KEY"):
            VLMConfig.from_env()

    def test_generic_endpoint_model_and_api_key_are_configurable(self) -> None:
        with patch.dict("os.environ", {
            "VLM_PROVIDER": "openai-compatible", "VLM_MODEL": "vendor/vision-model",
            "VLM_API_KEY": "test-secret", "VLM_BASE_URL": "https://models.example.com/v1/",
            "VLM_STRUCTURED_OUTPUT": "json_object", "VLM_REQUEST_OPTIONS": '{"temperature":0.1}',
        }, clear=True):
            client = create_vlm_client()
        self.assertIsInstance(client, ChatCompletionsClient)
        self.assertEqual(client.config.model, "vendor/vision-model")
        self.assertEqual(client.config.base_url, "https://models.example.com/v1")
        self.assertEqual(client.config.structured_output, "json_object")
        self.assertEqual(client.config.request_options["temperature"], 0.1)

    def test_invalid_config_is_rejected_before_any_network_request(self) -> None:
        for changes in (
            {"provider": "unknown"}, {"model": "with spaces"}, {"timeout_seconds": float("nan")},
            {"timeout_seconds": 0}, {"timeout_seconds": True}, {"max_output_tokens": -1},
            {"max_request_bytes": 0}, {"base_url": "https://example.com:bad/v1"},
            {"structured_output": "invented"}, {"request_options": {"messages": []}},
            {"request_options": {"generationConfig": {}}}, {"request_options": {"model": "other"}},
            {"request_options": {"temperature": float("nan")}},
        ):
            kwargs = {"provider": "qwen", "model": "selected-model", "api_key": "test-secret", **changes}
            with self.subTest(changes=changes), self.assertRaises(VLMError):
                VLMConfig(**kwargs)
        with self.assertRaisesRegex(VLMError, "VLM_MODEL"):
            VLMConfig(provider="anthropic", api_key="test-secret")
        with self.assertRaises(VLMError):
            settings("anthropic", structured_output="json_object")
        for value in ("not-json", "[]", "null"):
            with self.subTest(value=value), patch.dict("os.environ", {
                "DASHSCOPE_API_KEY": "test-secret", "VLM_REQUEST_OPTIONS": value,
            }, clear=True), self.assertRaises(VLMError):
                VLMConfig.from_env()

    def test_model_options_are_snapshotted_and_cannot_change_task_messages(self) -> None:
        options = {"thinking": {"type": "disabled"}}
        config = settings("anthropic", request_options=options)
        options["thinking"]["type"] = "adaptive"
        self.assertEqual(config.request_options["thinking"]["type"], "disabled")
        with self.assertRaises(TypeError):
            config.request_options["system"] = "new instructions"

    def test_transport_retries_are_bounded_and_size_limits_are_checked(self) -> None:
        opener = Mock()
        opener.open.side_effect = [HTTPError("https://example.com", 503, "private", {}, BytesIO()) for _ in range(3)]
        with patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
            with patch("pro_innovation.vlm.transport.time.sleep"):
                with self.assertRaisesRegex(VLMError, "503"):
                    JsonTransport().post("https://example.com", {}, {})
        self.assertEqual(opener.open.call_count, 3)
        with patch("pro_innovation.vlm.transport.build_opener") as build:
            with self.assertRaisesRegex(VLMError, "body limit"):
                JsonTransport(max_request_bytes=1).post("https://example.com", {}, {"x": "big"})
        build.assert_not_called()
        opener = Mock(open=Mock(return_value=BytesIO(b"123456")))
        with patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
            with self.assertRaisesRegex(VLMError, "size limit"):
                JsonTransport(max_response_bytes=5).post("https://example.com", {}, {})

    def test_transport_rejects_provider_error_envelopes_and_nonobjects(self) -> None:
        for raw in (b'[]', b'{"error":{"message":"test-secret"}}', b'not JSON'):
            opener = Mock(open=Mock(return_value=BytesIO(raw)))
            with self.subTest(raw=raw), patch("pro_innovation.vlm.transport.build_opener", return_value=opener):
                with self.assertRaises(VLMError) as caught:
                    JsonTransport().post("https://example.com", {}, {})
            self.assertNotIn("test-secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
