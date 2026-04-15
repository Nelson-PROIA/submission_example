import json

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)


class ChatRequest(BaseModel):
    message: str
    schema: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    messages = [
        {
            "role": "system",
            "content": (
                "You generate only valid Python Polars code.\n"
                "Return code only.\n"
                "Do not use markdown fences.\n"
                "Assume the benchmark executor already provides dataset paths and executes the code.\n"
                "Always assign the final Polars DataFrame to a variable named result.\n"
                "Use Polars idioms and prefer concise, correct code.\n"
                f"Available datasets:\n{json.dumps(payload.schema, ensure_ascii=False)}"
            ),
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
        max_new_tokens=384,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))