# CLAUDE.md

## Project

agent-core: a lightweight, extensible Python core for building AI agents. Read ARCHITECTURE.md before any multi-file change — it's the target design.

## Stack

Python 3.13 · uv · Pydantic v2 · ruff · mypy --strict · pytest

## Commands

Once per clone: `uv run pre-commit install`

Run in this order; all must pass before considering work done:

```
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/
uv run pytest tests/unit/ tests/integration/ -q
E2E_TESTS=1 uv run pytest tests/e2e/ -q
```

Single test: `uv run pytest tests/unit/path/to/test_file.py::test_name -q`

## Testing

Tests mirror `src/agent/` 1:1 across three tiers, auto-marked by folder (root `conftest.py`) so `pytest -m "not e2e"` filters across all of them:

- `tests/unit/` — pure logic, no network/filesystem/subprocess/mocking
- `tests/integration/` — adapter wiring, all externals mocked
- `tests/e2e/` — real runtime, opt-in via `E2E_TESTS=1`, never in CI

One assertion concept per test, named `test_<what>_<when>_<expected>`. Structured test data as Pydantic models, not raw `dict`/`tuple`.

## Documentation

- Docstrings: Google convention (ruff `D` rules enforce it) — one-line summary, then only `Args:`/`Returns:`/`Raises:`/`Yields:` as applicable, nothing else. Each entry is a short "what it is" phrase, never behavior, edge cases, or rationale. Complete always: every parameter and every non-`None` return gets an entry, regardless of arity, visibility, or whether the summary hints at it. The only exemption is a signature fixed by an external base class/protocol this codebase doesn't control — never a pattern the codebase merely repeats on its own — plus small decorator-registered closures (routes, exception handlers) whose contract lives in the docs instead.
- Pydantic `Field(description=...)` follows the same rule as docstrings.
- Comments: non-obvious rationale only, never a restatement of the line.
- Every docstring, comment, and `Field` description is standalone — never names another module, class, service, or function. Only root `README.md`, `ARCHITECTURE.md`, and per-folder `README.md` files may cross-reference other units.
- A genuinely cross-cutting fact (an invariant, a provider quirk, a system-wide contract) that would otherwise live only in a docstring or comment belongs in `ARCHITECTURE.md` instead — trim the docstring, add the fact there.
- Per-folder `README.md`: a one-sentence purpose statement plus one line per file/symbol — an index, not a design doc. No behavioral deep-dives; that's what the code and its docstrings are for.
- Every `src/` folder gets a short `README.md` (purpose + contents); update it when the folder's contents change.
- After any change to an interface, config shape, or entry point: update `README.md`, `ARCHITECTURE.md`, and any affected per-folder `README.md`.
