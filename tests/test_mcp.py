import json
import subprocess
import sys
import unittest

from helpers import ROOT, SRC  # noqa: F401  (sets sys.path)

from freellm.mcp import McpClient, McpServerConfig, load_servers
from freellm.tools import Registry

FAKE_SERVER = r'''
import json, sys

TOOLS = [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        text = (message.get("params", {}).get("arguments") or {}).get("text", "")
        result = {"content": [{"type": "text", "text": "echo:" + text}]}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message.get("id"), "result": result}) + "\n")
    sys.stdout.flush()
'''


class McpTests(unittest.TestCase):
    def test_lists_and_calls_tools(self) -> None:
        client = McpClient(McpServerConfig(command=sys.executable, args=["-c", FAKE_SERVER], name="demo"))
        try:
            client.connect()
            tools = client.list_tools()
            self.assertEqual(tools[0]["name"], "echo")

            output = client.call_tool("echo", {"text": "hi"})
            self.assertEqual(output, "echo:hi")
        finally:
            client.close()

    def test_registers_mcp_tools_locally(self) -> None:
        registry = Registry()
        client = McpClient(McpServerConfig(command=sys.executable, args=["-c", FAKE_SERVER], name="demo"))
        try:
            client.connect()
            registered = client.register_into(registry, prefix="mcp_demo_")

            self.assertEqual(registered, ["mcp_demo_echo"])
            self.assertIn("mcp_demo_echo", registry.names())
            self.assertEqual(registry.call("mcp_demo_echo", {"text": "yo"}), "echo:yo")
        finally:
            client.close()

    def test_load_servers_from_json(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp.json"
            path.write_text(
                json.dumps({"mcpServers": {"fs": {"command": "npx", "args": ["-y", "server"]}}}),
                encoding="utf-8",
            )
            servers = load_servers(path)

            self.assertEqual(len(servers), 1)
            self.assertEqual(servers[0].command, "npx")
            self.assertEqual(servers[0].name, "fs")


if __name__ == "__main__":
    unittest.main()
