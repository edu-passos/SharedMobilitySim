run:
	uv run python3 -m scripts.run_sim

.PHONY: ruff ruff-format ruff-check ruff-fix
ruff:
	uv run ruff format .
	uv run ruff check . --fix

ruff-format:
	uv run ruff format .

ruff-check:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

.PHONY: mypy
mypy:
	uv run mypy .
