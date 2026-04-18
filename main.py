import json
import re

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

MODEL_NAME = "mistralai/Devstral-Small-2507"

SYSTEM_PROMPT = """You are an expert Polars engineer. Output ONLY executable Python code. No markdown, no prose, no comments, no explanations, no imports.

Rules:
- `import polars as pl` is already done; do not re-import
- Assign the final DataFrame to a variable named `result`
- Reference DataFrames by the exact table names listed under "Available tables"
- Use `pl.col("name")`, not bracket indexing
- Use `.filter()`, not `.where()`
- Use `.group_by()`, not `.groupby()` (Polars v1+)
- Call `.collect()` before assigning to `result` if you built a LazyFrame
- Prefer `.alias(...)` over renaming afterward
- For window functions, use `.over("col")` on an expression inside `.with_columns(...)`

Examples:

Q: total revenue per region
A: result = sales.group_by("region").agg(pl.col("amount").sum().alias("revenue"))

Q: orders above 100 placed in 2024
A: result = orders.filter((pl.col("amount") > 100) & (pl.col("year") == 2024))

Q: each customer's name with their total order value
A: totals = orders.group_by("customer_id").agg(pl.col("amount").sum().alias("total"))
result = customers.join(totals, on="customer_id", how="inner")

Q: 7-day rolling average of amount per store
A: result = sales.with_columns(pl.col("amount").rolling_mean(window_size=7).over("store").alias("avg_7d"))
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

llm = LLM(
    model=MODEL_NAME,
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
    stop=["```\n", "\n\n\n"],
)


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
def chat(payload: ChatRequest) -> ChatResponse:
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

    outputs = llm.chat(messages, sampling_params)
    response = outputs[0].outputs[0].text

    return ChatResponse(response=strip_code_fence(response))
