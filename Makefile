.PHONY: install test lint typecheck up down fmt

install:
	uv sync --all-extras --dev

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src

up:
	docker compose up -d

down:
	docker compose down
