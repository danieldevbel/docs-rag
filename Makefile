.DEFAULT_GOAL := help
.PHONY: help install lint fmt type test check clean eval

help: ## Lista os alvos disponiveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependencias e hooks de pre-commit
	uv sync --all-extras --dev
	uv run pre-commit install

lint: ## Roda o linter
	uv run ruff check src tests

fmt: ## Formata o codigo
	uv run ruff format src tests
	uv run ruff check --fix src tests

type: ## Checagem estatica de tipos
	uv run mypy src

test: ## Roda os testes com cobertura
	uv run pytest --cov=src --cov-report=term-missing

check: lint type test ## Roda tudo o que a CI roda

clean: ## Remove artefatos de build e cache
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -exec rm -rf {} +

eval: ## Mede recall@k e MRR sobre o conjunto dourado
	uv run docs-rag eval --k 5
	uv run docs-rag eval --k 3
