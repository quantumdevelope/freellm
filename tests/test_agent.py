import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, SRC  # noqa: F401  (sets sys.path)

from freellm.agent import Agent
from freellm.llm.base import ChatResult
from freellm.tools import Registry
from helpers import ScriptedClient, tool_call


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry()
        self.registry.register(
            "echo",
            "Echo the given text.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            lambda text: f"echo:{text}",
        )

    def test_tool_call_is_executed_and_fed_back(self) -> None:
        client = ScriptedClient(
            [
                ChatResult(content="", tool_calls=[tool_call("echo", {"text": "hi"})]),
                ChatResult(content="All done.", model="scripted", prompt_tokens=10, completion_tokens=4),
            ]
        )
        agent = Agent(client, self.registry, workspace=".")

        events: list[tuple[str, dict]] = []
        result = agent.run("say hi", handler=lambda kind, payload: events.append((kind, payload)))

        self.assertEqual(result.text, "All done.")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.cost_usd, 0.0)
        self.assertIn(("tool_result", {"name": "echo", "output": "echo:hi"}), events)

        tool_messages = [message for message in result.messages if message.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].content, "echo:hi")

    def test_unknown_tool_does_not_crash_the_loop(self) -> None:
        client = ScriptedClient(
            [
                ChatResult(content="", tool_calls=[tool_call("missing")]),
                ChatResult(content="Recovered."),
            ]
        )
        agent = Agent(client, self.registry, workspace=".")

        result = agent.run("do something")

        self.assertEqual(result.text, "Recovered.")
        tool_messages = [message for message in result.messages if message.role == "tool"]
        self.assertIn("unknown tool", tool_messages[0].content)

    def test_max_steps_stops_the_loop(self) -> None:
        client = ScriptedClient(
            [
                ChatResult(content="", tool_calls=[tool_call("echo", {"text": "x"})]),
                ChatResult(content="", tool_calls=[tool_call("echo", {"text": "y"})]),
            ]
        )
        agent = Agent(client, self.registry, workspace=".", max_steps=2)

        result = agent.run("loop forever")

        self.assertEqual(result.steps, 2)
        self.assertTrue(result.errors)

    def test_history_is_kept_between_runs(self) -> None:
        client = ScriptedClient([ChatResult(content="one"), ChatResult(content="two")])
        agent = Agent(client, self.registry, workspace=".")

        agent.run("first")
        agent.run("second")

        self.assertEqual(len(agent.messages), 4)


if __name__ == "__main__":
    unittest.main()
