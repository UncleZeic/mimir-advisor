import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dockerignore_rules() -> list[str]:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _pattern_regex(pattern: str) -> re.Pattern:
    """Go filepath.Match plus Docker's **: '*' and '?' never cross '/', '**'
    spans any number of folders (including none)."""
    out = ""
    i = 0

    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1

    return re.compile(out + r"\Z")


def _matches(pattern: str, path: str) -> bool:
    """A rule matches the path itself or any parent folder of it."""
    regex = _pattern_regex(pattern.rstrip("/"))
    parts = path.split("/")
    return any(regex.match("/".join(parts[: i + 1])) for i in range(len(parts)))


def in_build_context(path: str) -> bool:
    included = True

    for rule in dockerignore_rules():
        negated = rule.startswith("!")

        if _matches(rule.lstrip("!"), path):
            included = negated

    return included


class DockerignoreTests(unittest.TestCase):
    def test_only_code_and_requirements_reach_the_image(self) -> None:
        for path in ("src/main.py", "src/store.py", "requirements.txt"):
            with self.subTest(path=path):
                self.assertTrue(in_build_context(path))

    def test_secrets_data_and_indexes_never_reach_the_image(self) -> None:
        for path in (
            ".env",
            ".env.local",
            "data/notes/private.md",
            "nested/.env",
            "src/.env",
            "src/.env.backup",
            "x.sqlite",
            "deep/folder/vault-1234.sqlite.tmp",
            "src/stray.sqlite-journal",
            "private-eval.txt",
            "tests/fixtures/eval.txt",
            ".git/config",
            ".claude/settings.json",
            ".venv/bin/python",
            "src/__pycache__/main.cpython-312.pyc",
        ):
            with self.subTest(path=path):
                self.assertFalse(in_build_context(path))

    def test_dockerignore_starts_by_excluding_everything(self) -> None:
        rules = dockerignore_rules()

        self.assertEqual(rules[0], "*")
        self.assertEqual({rule for rule in rules if rule.startswith("!")}, {"!src/", "!requirements.txt"})

    def test_index_files_are_never_committed(self) -> None:
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("*.sqlite*", {line.strip() for line in lines})

    def test_dependencies_are_pinned_to_exact_versions(self) -> None:
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        requirements = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

        self.assertTrue(requirements)
        for requirement in requirements:
            self.assertRegex(requirement, r"^[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*$")


class ComposeTests(unittest.TestCase):
    def test_container_user_has_no_default_uid(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        user = re.search(r'^\s*user:\s*"(.+)"\s*$', compose, re.MULTILINE).group(1)

        self.assertRegex(user, r"^\$\{HOST_UID:\?[^}]+\}:\$\{HOST_GID:\?[^}]+\}$")
        self.assertNotIn(":-", user)

    def test_env_template_does_not_hardcode_ids(self) -> None:
        template = (ROOT / ".env.template").read_text(encoding="utf-8")

        self.assertNotRegex(template, r"(?m)^HOST_[UG]ID=\S")

    def test_compose_maps_host_docker_internal_via_the_hosts_file(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn('"host.docker.internal:host-gateway"', compose)


if __name__ == "__main__":
    unittest.main()
