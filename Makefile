PYTHON ?= python3
TORCH_CPU = --extra-index-url https://download.pytorch.org/whl/cpu

.PHONY: install install-train test lint format run-backend train-text train-image

install:
	$(PYTHON) -m pip install -r requirements-dev.txt $(TORCH_CPU)

install-train:
	$(PYTHON) -m pip install -r requirements-train.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check . --fix

run-backend:
	$(PYTHON) -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

train-text:
	$(PYTHON) training/train_text_model.py

train-image:
	$(PYTHON) training/train_image_model.py
