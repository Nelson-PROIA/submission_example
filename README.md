# Polars Bench Submission — Devstral Small 2

Text → Polars codegen service. FastAPI `/chat` endpoint accepts a natural-language analytics question plus table schemas and returns executable Polars Python code that assigns the answer to a variable named `result`.

## Model choice

**`mistralai/Devstral-Small-2507`** (24B, code-specialized, Apache 2.0). Chosen for strong code-generation quality in a size that still fits on a single 48GB+ GPU at fp16 — better than Qwen2.5-Coder-7B on code benchmarks and faster than a 70B general-purpose model.

## Key optimizations

- **Few-shot system prompt** with 4 worked examples (aggregation, filter, join, window) plus an explicit Polars gotcha list (`.filter` vs `.where`, `.group_by` vs `.groupby`, `pl.col` vs bracket indexing, `.collect()` on LazyFrames).
- **Hardened output parser** (`strip_code_fence`): extracts fenced blocks when present, loops prose-prefix removal (handles "Sure! Here's the code:"), strips chat-template token leakage (`<|im_end|>`, `[/INST]`, etc.), drops trailing explanations. Smoke-tested against 10 synthetic LLM output patterns.
- **Generation config**: greedy (`do_sample=False`), `max_new_tokens=512`, stop strings `["```\n", "\n\n\n"]` to cut generation at the end of the code block.
- **Pinned dependencies** in both `requirements.txt` and `pyproject.toml` for reproducible runner builds.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check: `GET /` returns `{"status": "ok"}`

## Endpoint contract

`POST /chat`

Request:
```json
{
  "message": "total revenue per region",
  "tables": {"sales": {"columns": ["region", "amount"], "dtypes": ["str", "f64"]}}
}
```

Response:
```json
{"response": "result = sales.group_by(\"region\").agg(pl.col(\"amount\").sum().alias(\"revenue\"))"}
```

## Reproducibility

- Python `==3.12`, `torch==2.8.0`, `transformers==5.3.0` — all pinned
- Model revision resolved by HuggingFace at first load; no runtime downloads beyond the initial model fetch
- Greedy decoding → deterministic outputs across runs
