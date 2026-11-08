import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from helpers import ROOT, SRC  # noqa: F401  (sets sys.path)

from freellm.cli import main


class CliTests(unittest.TestCase):
    def test_config_command_prints_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["config"])

        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("model", payload)
        self.assertIn("base_url", payload)

    def test_version_flag(self) -> None:
        with self.assertRaises(SystemExit) as context:
            main(["--version"])
        self.assertEqual(context.exception.code, 0)

    def test_doctor_reports_failures_without_crashing(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["--base-url", "http://127.0.0.1:1/v1", "doctor"])

        output = buffer.getvalue()
        self.assertIn("backend reachable", output)
        self.assertEqual(code, 1)

    def test_plugins_command_lists_tools(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["--no-mcp", "--no-web", "plugins"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("read_file", output)
        self.assertIn("run_command", output)


if __name__ == "__main__":
    unittest.main()
