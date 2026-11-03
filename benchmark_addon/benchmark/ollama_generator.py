"""
Minimal Ollama client for the baseline LMs (llama2:7b, llama3:8b,
gemma3:12b, ...). Needs `ollama serve` running and `ollama pull <model>`.

Decoding follows the paper's baseline setup: greedy (temperature 0),
max 100 new tokens for open QA, 50 for closed-set tasks.
"""
import requests

PAPER_RETRIEVAL_PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
)
PAPER_PLAIN_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_paragraph(ctxs: list[dict]) -> str:
    """Same layout as the official baseline script: '[1] title\\ntext'."""
    return "\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(ctxs, 1))


class OllamaGenerator:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 template: str = "chat", timeout: float = 300.0):
        if template not in ("chat", "alpaca"):
            raise ValueError("template must be 'chat' or 'alpaca'")
        self.model, self.host, self.template, self.timeout = model, host.rstrip("/"), template, timeout

    def _options(self, max_new_tokens: int) -> dict:
        return {"temperature": 0.0, "num_predict": max_new_tokens, "seed": 0}

    def generate(self, instruction: str, ctxs: list[dict] | None,
                 max_new_tokens: int = 100) -> str:
        """`ctxs=None` -> no-retrieval baseline; list -> retrieval baseline."""
        if self.template == "alpaca":
            # Paper-faithful raw prompt, bypasses the model's chat template.
            prompt = (PAPER_RETRIEVAL_PROMPT.format(paragraph=format_paragraph(ctxs),
                                                    instruction=instruction)
                      if ctxs else PAPER_PLAIN_PROMPT.format(instruction=instruction))
            r = requests.post(f"{self.host}/api/generate", timeout=self.timeout, json={
                "model": self.model, "prompt": prompt, "raw": True, "stream": False,
                "options": self._options(max_new_tokens)})
            r.raise_for_status()
            return r.json()["response"].strip()

        # 'chat': the model's own template (what the paper does for
        # instruction-tuned LMs: "official instruction format ... if available").
        content = (f"Use the following retrieved passages if they are helpful.\n\n"
                   f"{format_paragraph(ctxs)}\n\n{instruction}") if ctxs else instruction
        r = requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False,
            "messages": [{"role": "user", "content": content}],
            "options": self._options(max_new_tokens)})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
