# Deployment checklist

## Local

1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. Test a normal RAG query.
4. Test a billing/refund query and enter a human response.

## Streamlit Cloud

- Main file: `app.py`
- Secrets: `GOOGLE_API_KEY`
- Python dependencies: `requirements.txt`
- Repository data paths must remain relative.

## Production note

`InMemorySaver` is suitable for this demo, but it is process-local. A production deployment with multiple replicas should use a durable checkpointer such as Postgres.
