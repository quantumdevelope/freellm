import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, SRC  # noqa: F401  (sets sys.path)

from freellm.rag import RagStore, chunk_text, cosine, index_directory, search


def fake_embed(text: str) -> list[float]:
    """Deterministic 3-dimensional embedding based on character frequencies."""
    vector = [0.0, 0.0, 0.0]
    for char in text.lower():
        vector[ord(char) % 3] += 1.0
    return vector


class ChunkTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self) -> None:
        self.assertEqual(chunk_text("hello"), ["hello"])

    def test_long_text_is_split_with_overlap(self) -> None:
        chunks = chunk_text("a" * 5000, size=1000, overlap=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))

    def test_empty_text(self) -> None:
        self.assertEqual(chunk_text("   "), [])


class VectorTests(unittest.TestCase):
    def test_cosine_of_identical_vectors(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_cosine_of_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_handles_empty_vectors(self) -> None:
        self.assertEqual(cosine([], [1.0]), 0.0)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = RagStore(Path(self._tmp.name) / "index.db")

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_index_and_search(self) -> None:
        root = Path(self._tmp.name) / "project"
        root.mkdir()
        (root / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
        (root / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
        (root / "notes.txt").write_text("unrelated text\n", encoding="utf-8")

        files, chunks = index_directory(self.store, root, fake_embed)
        self.assertEqual(files, 3)
        self.assertGreaterEqual(chunks, 3)

        hits = search(self.store, "alpha", fake_embed, top_k=2)
        self.assertTrue(hits)
        self.assertTrue(hits[0].path.endswith("a.py"))

    def test_upsert_replaces_old_chunks(self) -> None:
        root = Path(self._tmp.name) / "project"
        root.mkdir()
        (root / "a.py").write_text("first version\n", encoding="utf-8")
        index_directory(self.store, root, fake_embed)

        (root / "a.py").write_text("second version with more text\n", encoding="utf-8")
        index_directory(self.store, root, fake_embed)

        paths = [path for path, _, _, _ in self.store.all_chunks()]
        self.assertEqual(len(paths), 1)


if __name__ == "__main__":
    unittest.main()
