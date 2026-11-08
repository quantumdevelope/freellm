import json
import os
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, SRC  # noqa: F401  (sets sys.path)

from freellm.config import Config, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_are_local_and_free(self) -> None:
        config = Config()
        self.assertEqual(config.provider, "ollama")
        self.assertIn("11434", config.base_url)
        self.assertIsNone(config.api_key)

    def test_env_overrides_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"model": "from-file", "temperature": 0.1}), encoding="utf-8")

            config = load_config(path, env={"FREELLM_MODEL": "from-env"})

            self.assertEqual(config.model, "from-env")
            self.assertEqual(config.temperature, 0.1)

    def test_broken_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_config(path, env={}).model, Config().model)

    def test_save_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "config.json"
            config = Config(model="saved-model")
            config.save(path)

            self.assertTrue(path.exists())
            self.assertEqual(load_config(path, env={}).model, "saved-model")


if __name__ == "__main__":
    unittest.main()
