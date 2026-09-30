.PHONY: install test train evaluate benchmark app

install:
	python -m pip install -e . pytest ruff flake8

test:
	pytest tests/

train:
	python -m src.train --epochs 2

evaluate:
	python -m src.evaluate --run-robustness

benchmark:
	python -m src.benchmark --iterations 1000

app:
	streamlit run app/dashboard.py
