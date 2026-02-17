python scripts/seed_demo_data.py
python -m src.clv.run_all
uvicorn src.api.main:app --reload
