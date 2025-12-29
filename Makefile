.PHONY: run ruff ruff-format ruff-check ruff-fix mypy

run:
	uv run python3 -m scripts.run_sim

ruff: ruff-format ruff-fix

ruff-format:
	uv run ruff format .

ruff-check:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

mypy:
	uv run mypy .
