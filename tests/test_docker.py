import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DockerignoreTests(unittest.TestCase):
    def test_secrets_and_data_are_excluded_from_build_context(self) -> None:
        lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        entries = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
        self.assertIn(".env", entries)
        self.assertIn("data/", entries)

    def test_index_files_are_never_committed_or_copied_into_images(self) -> None:
        for name in (".gitignore", ".dockerignore"):
            with self.subTest(file=name):
                lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
                self.assertIn("*.sqlite*", {line.strip() for line in lines})


if __name__ == "__main__":
    unittest.main()
