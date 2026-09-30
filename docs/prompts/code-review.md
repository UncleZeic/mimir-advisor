Review the changes in {{RANGE}}. Read-only: do not edit files.
Check against README.md and AGENTS.md:
1. Correctness: logic errors, edge cases, error handling, data consistency.
2. Privacy (highest priority): any network call other than OLLAMA_BASE_URL; anything that could read real notes by default or write note text inside the repo; permissions of files that store note text; secrets in code or config.
3. Tests: what's untested or tested only superficially.
4. Drift: anything contradicting README principles or AGENTS.md.
{{FOCUS}}
Report findings ranked by severity, each with file:line and a concrete failure scenario. Skip style nitpicks.
