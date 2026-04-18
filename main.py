import json
import re
import sys
import traceback

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Devstral-Small-2507"

SYSTEM_PROMPT = """You are an expert Polars engineer. Output ONLY executable Python code. No markdown, no prose, no comments, no explanations.

Rules:
- `import polars as pl` is already done; do not re-import
- Load each table you reference with: `<name> = pl.read_csv("/app/data/<name>.csv", try_parse_dates=True)`
- Only load tables you actually use
- Assign the final DataFrame to a variable named `result`
- Wrap chained calls in parentheses for readability
- Use `pl.col("name")`, not bracket indexing
- Use `.filter()`, not `.where()`
- Use `.group_by()`, not `.groupby()` (Polars v1+)
- Use `.alias(...)` to name new columns
- For window functions, use `.over("col")` on an expression inside `.with_columns(...)` or `.agg(...)`

Examples:

Q: total revenue per region from sales
A: sales = pl.read_csv("/app/data/sales.csv", try_parse_dates=True)

result = (
    sales
    .group_by("region")
    .agg(pl.col("amount").sum().alias("revenue"))
)

Q: orders above 100 placed in 2024 from orders
A: orders = pl.read_csv("/app/data/orders.csv", try_parse_dates=True)

result = orders.filter((pl.col("amount") > 100) & (pl.col("year") == 2024))

Q: each customer's name with their total order value using customers and orders
A: customers = pl.read_csv("/app/data/customers.csv", try_parse_dates=True)
orders = pl.read_csv("/app/data/orders.csv", try_parse_dates=True)

totals = (
    orders
    .group_by("customer_id")
    .agg(pl.col("amount").sum().alias("total"))
)

result = customers.join(totals, on="customer_id", how="inner")

Q: 7-day rolling average of amount per store from sales
A: sales = pl.read_csv("/app/data/sales.csv", try_parse_dates=True)

result = sales.with_columns(
    pl.col("amount").rolling_mean(window_size=7).over("store").alias("avg_7d")
)
"""

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_PROSE_PREFIX_RE = re.compile(
    r"^(sure[!,.]?|certainly[!,.]?|okay[!,.]?|ok[!,.]?|here(?:'s| is)[^\n]*:?|the code(?: is)?:?|i'll[^\n]*:?)\s*",
    re.IGNORECASE,
)
_TRAILING_PROSE_RE = re.compile(
    r"\n(?:this |the above |note:|explanation:|# this |# the )",
    re.IGNORECASE,
)
_CHAT_TEMPLATE_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "[/INST]",
    "</s>",
    "<s>",
)


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

print("Model device:", next(model.parameters()).device)


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()

    for token in _CHAT_TEMPLATE_TOKENS:
        text = text.replace(token, "")
    text = text.strip()

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        for _ in range(4):
            stripped = _PROSE_PREFIX_RE.sub("", text).strip()
            if stripped == text:
                break
            text = stripped
        if text.startswith("python\n"):
            text = text[len("python\n"):].lstrip()

    trailing_match = _TRAILING_PROSE_RE.search(text)
    if trailing_match:
        text = text[: trailing_match.start()].rstrip()

    return text.strip()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        tables_json = json.dumps(payload.tables, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\nAvailable tables: {tables_json}",
            },
            {
                "role": "user",
                "content": payload.message,
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return ChatResponse(response=strip_code_fence(response))
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[CHAT ERROR] {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr, flush=True)
        return ChatResponse(response=f"# ERROR {type(exc).__name__}: {str(exc)[:400]}")