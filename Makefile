.PHONY: check lint type test sync
sync:
	uv sync --group dev
lint:
	uv run ruff check reviewsys tests
	uv run ruff format --check reviewsys tests
type:
	uv run mypy reviewsys
test:
	uv run pytest
check: lint type test
fmt:
	uv run ruff format reviewsys tests
	uv run ruff check --fix reviewsys tests
