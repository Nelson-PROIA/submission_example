from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to("cuda")

print("cuda:", torch.cuda.is_available())
print("device:", model.device)
print(next(model.parameters()).device)

class ChatRequest(BaseModel):
    message: str
    schema: dict

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. that generates polars code "
                "Answer strictly in python code, without anything else:\n"
                "Here are the schemas of available data"
                f"{json.dumps(payload.schema, ensure_ascii=False)}"
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
        max_new_tokens=512,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return ChatResponse(response=response)