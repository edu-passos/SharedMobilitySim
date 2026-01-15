.PHONY: ruff ruff-format ruff-check ruff-fix mypy

ruff: ruff-format ruff-fix

ruff-format:
	uv run ruff format .

ruff-check:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

mypy:
	uv run mypy .
