PYTHON := python3
VENV := .venv
ACTIVATE := . $(VENV)/bin/activate

-include .env
export

.PHONY: setup train run-api run-ui

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

train:
	$(ACTIVATE) && MPLBACKEND=Agg $(PYTHON) scripts/bike_theft_training_pipeline.py

run-api:
	$(ACTIVATE) && $(PYTHON) -m backend.quant_agent_api

run-ui:
	$(ACTIVATE) && $(PYTHON) -m http.server 5500 --directory frontend
