run_all:
	python scripts/seed_demo_data.py
	python -m src.clv.run_all

run_api:
	uvicorn src.api.main:app --reload

run_app:
	streamlit run src/app/app.py

test:
	pytest -q
