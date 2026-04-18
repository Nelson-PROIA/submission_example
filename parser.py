import re

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
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "[INST]",
    "[/INST]",
    "</s>",
    "<s>",
)


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
