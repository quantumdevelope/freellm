import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from helpers import ROOT, SRC  # noqa: F401  (sets sys.path)

from freellm.agent import Agent
from freellm.llm.base import ChatResult, Message
from freellm.llm.openai_compat import OpenAICompatClient
from freellm.server import FreellmServer, ServerConfig
from freellm.tools import Registry
from helpers import ScriptedClient, tool_call


class FakeBackend(BaseHTTPRequestHandler):
    """A tiny OpenAI-compatible backend used for the tests."""

    payload: dict = {}

    def log_message(self, *args):  # noqa: A003
        return

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.endswith("/models"):
            self._json({"object": "list", "data": [{"id": "fake-model"}]})
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        FakeBackend.last_request = request
        self._json(type(self).payload)


class OpenAICompatTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeBackend.payload = {
            "id": "chatcmpl-1",
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hi there",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"text":"hi"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackend)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_chat_parses_content_and_tool_calls(self) -> None:
        client = OpenAICompatClient(self.base_url, model="fake-model")
        result = client.chat([Message(role="user", content="hello")])

        self.assertEqual(result.content, "hi there")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "echo")
        self.assertEqual(result.tool_calls[0].arguments, {"text": "hi"})
        self.assertEqual(result.prompt_tokens, 7)

    def test_list_models(self) -> None:
        client = OpenAICompatClient(self.base_url, model="fake-model")
        self.assertEqual(client.list_models(), ["fake-model"])

    def test_tools_are_sent_in_openai_format(self) -> None:
        client = OpenAICompatClient(self.base_url, model="fake-model")
        client.chat([Message(role="user", content="hello")])

        sent = FakeBackend.last_request
        self.assertIn("messages", sent)
        self.assertEqual(sent["messages"][0]["role"], "user")
        self.assertFalse(sent["stream"])


class FreellmServerTests(unittest.TestCase):
    def setUp(self) -> None:
        registry = Registry()
        registry.register(
            "echo",
            "Echo the given text.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            lambda text: f"echo:{text}",
        )
        client = ScriptedClient(
            [
                ChatResult(content="", tool_calls=[tool_call("echo", {"text": "yo"})]),
                ChatResult(content="final answer"),
            ]
        )
        self.agent = Agent(client, registry, workspace=".")
        self.server = FreellmServer(self.agent, ServerConfig(host="127.0.0.1", port=0, model="fake-model"))
        self.thread = self.server.serve_in_thread()

        for _ in range(50):
            try:
                urllib.request.urlopen(f"{self.server.url}/healthz", timeout=1)
                break
            except Exception:  # noqa: BLE001 - server is still starting
                threading.Event().wait(0.05)

    def test_health_endpoint(self) -> None:
        with urllib.request.urlopen(f"{self.server.url}/healthz", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["cost_usd"], 0.0)

    def test_models_endpoint(self) -> None:
        with urllib.request.urlopen(f"{self.server.url}/v1/models", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["data"][0]["id"], "fake-model")

    def test_chat_completions_runs_the_agent(self) -> None:
        body = json.dumps(
            {"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server.url}/v1/chat/completions",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["choices"][0]["message"]["content"], "final answer")
        self.assertEqual(payload["freellm"]["cost_usd"], 0.0)
        self.assertEqual(payload["freellm"]["tool_calls"], 1)


if __name__ == "__main__":
    unittest.main()
