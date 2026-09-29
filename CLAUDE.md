# mimir-advisor
Local-first private advisor. See README.md for principles.
- Python 3.11, stdlib first, unittest; mock the network, tests never hit real services.
- Run tests: python3 -m unittest discover -s tests -v
- Only synthetic data: tests/fixtures/vault. Never read .env, data/, /tank/vault or ~/.local/share/mimir-advisor.
- No network calls except the configured OLLAMA_BASE_URL.
