.PHONY: help setup db pipeline dashboard test clean

PYTHON := python3
VENV   := venv
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Local setup ──────────────────────────────────────────────────────────────

venv:          ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv  ## Install Python dependencies
	$(PIP) install -r requirements.txt

setup: install ## Full local setup (venv + deps + .env check)
	@test -f .env || (cp .env.template .env && echo "⚠  .env created from template — add your API keys")

# ── Database ─────────────────────────────────────────────────────────────────

db-docker:     ## Start PostgreSQL+pgvector via Docker (recommended)
	docker compose up -d db
	@echo "⏳ Waiting for Postgres..." && sleep 5

db-init:       ## Initialize schema and load all data
	$(VENV)/bin/python db/setup_db.py
	$(VENV)/bin/python data/load_reviews.py
	$(VENV)/bin/python data/load_public_health_data.py
	$(VENV)/bin/python agent/build_rag.py

# ── Pipeline ─────────────────────────────────────────────────────────────────

pipeline:      ## Run full analysis pipeline (requires DB)
	$(VENV)/bin/python run_pipeline.py

# ── App ──────────────────────────────────────────────────────────────────────

dashboard:     ## Launch Streamlit dashboard
	$(VENV)/bin/streamlit run streamlit_app/app.py

docker-up:     ## Start everything in Docker (db + app)
	docker compose up --build

docker-down:   ## Stop Docker services
	docker compose down

# ── Tests ────────────────────────────────────────────────────────────────────

test:          ## Run test suite
	$(PYTEST) tests/ -v --tb=short

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:         ## Remove generated outputs and caches
	rm -rf outputs/*.csv outputs/*.json outputs/*.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
