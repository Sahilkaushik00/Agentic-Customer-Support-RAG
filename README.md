# AmazonHelp Support AI — Streamlit Demo

This folder wraps the supplied `pipeline.py` core project in a Streamlit UI.

## Architecture

Customer message → Classical ML intent classifier → RAG retrieval → structured LLM judge → RAG response or LangGraph HITL → final response.

When the graph takes the HITL path, the Streamlit UI displays the customer query, predicted intent, judge reason and historical evidence. The human enters the final response; that exact response is used to resume the LangGraph checkpoint and becomes `final_response`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Gemini routing/generation, create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_key_here"
```

The app will also run without a Gemini key using the deterministic fallback implemented in the supplied core pipeline.

## Streamlit Cloud

1. Push this folder to GitHub.
2. In Streamlit Community Cloud choose `app.py` as the entrypoint.
3. Add `GOOGLE_API_KEY` under App settings → Secrets.
4. Deploy.

## Data

The supplied core pipeline first looks for:

`data/processed/amazonhelp_brand_conversations.csv.gz`

For a lightweight demo deployment, this repository includes `data/sample_amazonhelp_pairs.json`. For a real dataset-backed deployment, copy the processed AmazonHelp corpus into `data/processed/`.

## Important project distinction

This folder is the **interactive demo/deployment layer** around the core pipeline. Keep the Hiver evaluation artifacts (golden set, benchmark results, report and decision log) separate from the Streamlit UI submission.
