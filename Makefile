install:
	pip install -r requirements.txt

run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

run-ui:
	python ui/gradio_app.py

test:
	pytest tests/ -v
