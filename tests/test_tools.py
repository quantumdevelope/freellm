import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, SRC  # noqa: F401  (sets sys.path)

from freellm.tools import Registry
from freellm.tools import fs as fs_tools
from freellm.tools import shell as shell_tools
from freellm.tools.web import html_to_text


class FileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = Registry()
        fs_tools.register(self.registry, self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_read_write_edit(self) -> None:
        self.registry.call("write_file", {"path": "a.txt", "content": "hello world"})
        self.assertEqual(self.registry.call("read_file", {"path": "a.txt"}), "hello world")

        self.registry.call("edit_file", {"path": "a.txt", "old_string": "world", "new_string": "there"})
        self.assertEqual(self.registry.call("read_file", {"path": "a.txt"}), "hello there")

    def test_edit_requires_unique_match(self) -> None:
        self.registry.call("write_file", {"path": "b.txt", "content": "x x"})
        result = self.registry.call("edit_file", {"path": "b.txt", "old_string": "x", "new_string": "y"})
        self.assertIn("appears 2 times", result)

    def test_paths_cannot_escape_the_workspace(self) -> None:
        result = self.registry.call("read_file", {"path": str(Path(self.root).parent / "secret.txt")})
        self.assertIn("escapes the workspace", result)

    def test_list_and_find_and_search(self) -> None:
        self.registry.call("write_file", {"path": "sub/one.py", "content": "print('needle')"})
        self.registry.call("write_file", {"path": "sub/two.txt", "content": "nothing"})

        listing = self.registry.call("list_dir", {"path": "sub"})
        self.assertIn("one.py", listing)

        found = self.registry.call("find_files", {"pattern": "*.py"})
        self.assertIn("sub/one.py", found)

        hits = self.registry.call("search_text", {"query": "needle"})
        self.assertIn("one.py", hits)

    def test_unknown_tool_is_reported(self) -> None:
        self.assertIn("unknown tool", self.registry.call("nope", {}))


class ShellToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.executable = Path(sys.executable).name.lower()
        self.executable = self.executable[:-4] if self.executable.endswith(".exe") else self.executable
        self.registry = Registry()
        shell_tools.register(self.registry, self.root, allowed_commands=[self.executable])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_allowed_command_runs(self) -> None:
        output = self.registry.call("run_command", {"command": f"{sys.executable} -c print(123)"})
        self.assertIn("123", output)

    def test_blocked_command_is_refused(self) -> None:
        result = self.registry.call("run_command", {"command": "rm -rf /"})
        self.assertIn("not allowed", result)

    def test_allow_any_command(self) -> None:
        registry = Registry()
        shell_tools.register(registry, self.root, allowed_commands=[], allow_any=True)
        self.assertNotIn("not allowed", registry.call("run_command", {"command": f"{sys.executable} -c print(1)"}))


class WebHelperTests(unittest.TestCase):
    def test_html_to_text_strips_markup_and_scripts(self) -> None:
        html = "<html><head><style>a{}</style><script>bad()</script></head><body><p>Hello &amp; bye</p></body></html>"
        text = html_to_text(html)

        self.assertIn("Hello & bye", text)
        self.assertNotIn("bad()", text)
        self.assertNotIn("<p>", text)


if __name__ == "__main__":
    unittest.main()
