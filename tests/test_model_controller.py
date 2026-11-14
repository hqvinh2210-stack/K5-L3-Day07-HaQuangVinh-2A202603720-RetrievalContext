import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from model_controller import AnthropicController, OpenAIController


class FakeResponses:
    def __init__(self, action):
        self.action = action
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.action), usage={"input_tokens": 10})


class FakeMessages:
    def __init__(self, action):
        self.action = action
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        block = SimpleNamespace(type="text", text=json.dumps(self.action))
        return SimpleNamespace(content=[block], usage={"input_tokens": 10})


class ModelControllerTests(unittest.TestCase):
    def test_structured_search_action(self):
        responses = FakeResponses(
            {"action": "search", "query": "hiệu lực nghị định", "answer": "", "citations": []}
        )
        client = SimpleNamespace(responses=responses)
        controller = OpenAIController(model="test-model", client=client)
        action = controller.decide("Có hiệu lực khi nào?", [])
        self.assertEqual(action["action"], "search")
        self.assertEqual(responses.kwargs["text"]["format"]["type"], "json_schema")

    def test_rejects_malformed_business_action(self):
        client = SimpleNamespace(
            responses=FakeResponses(
                {"action": "search", "query": "", "answer": "guess", "citations": []}
            )
        )
        controller = OpenAIController(model="test-model", client=client)
        with self.assertRaises(RuntimeError):
            controller.decide("q", [])

    def test_anthropic_structured_search_action(self):
        messages = FakeMessages(
            {"action": "search", "query": "hiệu lực nghị định", "answer": "", "citations": []}
        )
        client = SimpleNamespace(messages=messages)
        controller = AnthropicController(model="test-claude", client=client)
        action = controller.decide("Có hiệu lực khi nào?", [])
        self.assertEqual(action["action"], "search")
        self.assertEqual(
            messages.kwargs["output_config"]["format"]["type"], "json_schema"
        )


if __name__ == "__main__":
    unittest.main()
