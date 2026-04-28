PYTHON ?= python3

.PHONY: install test lint format run-backend eval

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest ruff pip-audit

test:
	pytest -q

lint:
	ruff check .

format:
	ruff check . --fix

run-backend:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

eval:
	$(PYTHON) scripts/evaluate_text_detector.py --output reports/evaluation_results.json
