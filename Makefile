.PHONY: clean test test-file

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@echo "Cleaned all cache/pycache directories."

test:
	uv run pytest vqabench/backend/tests -v

test-file:
	uv run pytest vqabench/backend/tests/$(FILE) -v