"""Experimental MLX inference backend for fine-tuned Self-RAG 7B QLoRA.

This module provides `MLXSelfRAGGenerator`, a drop-in generator backend for EvoRAG
that executes the Task 12E fine-tuned QLoRA adapter on Apple Silicon Metal GPU.

Key features:
1. Loads 4-bit Self-RAG base model + fine-tuned LoRA adapter via MLX-LM.
2. Preserves all 16 Self-RAG special tokens as atomic token IDs (32000-32015).
3. Enforces token-level stopping criteria on utility tokens ([Utility:1]..[Utility:5])
   and retrieval decision probes without truncating partial special tokens.
4. Records inference latency, token counts, peak memory, and stopping reasons.
5. Fully compatible with EvoRAGController and self_rag_answer pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    HAS_MLX = True
except ImportError:
    mx = None
    mlx_lm = None
    load = None
    stream_generate = None
    HAS_MLX = False

# Self-RAG Special Token Constants
RETRIEVE_NO = "[No Retrieval]"
RETRIEVE_YES = "[Retrieval]"
RETRIEVE_CONTINUE = "[Continue to Use Evidence]"
ISREL_IRRELEVANT = "[Irrelevant]"
ISREL_RELEVANT = "[Relevant]"
ISSUP_FULLY = "[Fully supported]"
ISSUP_PARTIALLY = "[Partially supported]"
ISSUP_NONE = "[No support / Contradictory]"

# Token IDs in selfrag_llama2_7b vocabulary
SPECIAL_TOKEN_IDS: Dict[str, int] = {
    RETRIEVE_NO: 32000,
    RETRIEVE_YES: 32001,
    RETRIEVE_CONTINUE: 32002,
    ISREL_IRRELEVANT: 32003,
    ISREL_RELEVANT: 32004,
    "<paragraph>": 32005,
    "</paragraph>": 32006,
    "[Utility:1]": 32007,
    "[Utility:2]": 32008,
    "[Utility:3]": 32009,
    "[Utility:4]": 32010,
    "[Utility:5]": 32011,
    ISSUP_FULLY: 32012,
    ISSUP_PARTIALLY: 32013,
    ISSUP_NONE: 32014,
    "<pad>": 32015,
}

UTILITY_TOKEN_IDS: Set[int] = {32007, 32008, 32009, 32010, 32011}
RETRIEVAL_DECISION_IDS: Set[int] = {32000, 32001}
EOS_TOKEN_ID: int = 2

DEFAULT_BASE_MODEL = "models/mlx/selfrag_llama2_7b_4bit"
DEFAULT_ADAPTER_DIR = "training/qlora_dbms/adapters/full_dbms_v1"


def format_prompt(instruction: str, paragraph: Optional[str] = None, history: str = "") -> str:
    """Canonical Self-RAG prompt formatting from the official model card."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{history}"
    if paragraph is not None:
        if not history.endswith(RETRIEVE_YES):
            prompt += RETRIEVE_YES
        prompt += f"<paragraph>{paragraph}</paragraph>"
    return prompt


class MLXSelfRAGGenerator:
    """Experimental Self-RAG Generator using Apple Silicon MLX-LM with QLoRA adapter."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_BASE_MODEL,
        adapter_path: Optional[str | Path] = DEFAULT_ADAPTER_DIR,
        max_tokens: int = 250,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_on_utility: bool = True,
        stop_on_retrieval_probe: bool = True,
    ):
        if not HAS_MLX:
            raise ImportError(
                "MLX and MLX-LM are required to use MLXSelfRAGGenerator. "
                "Please run within an Apple Silicon environment with MLX installed."
            )
        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_on_utility = stop_on_utility
        self.stop_on_retrieval_probe = stop_on_retrieval_probe

        # Telemetry & Diagnostics
        self.load_duration_seconds: float = 0.0
        self.memory_after_load_gb: float = 0.0
        self.total_generations: int = 0
        self.generation_latencies_ms: List[float] = []
        self.token_counts: List[int] = []
        self.stop_reasons: Dict[str, int] = {}

        self._load_model()

    def _load_model(self):
        """Load 4-bit base model and optional LoRA adapter."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Base MLX model directory not found: {self.model_path}")

        t0 = time.perf_counter()
        mx.reset_peak_memory()

        adapter_str = str(self.adapter_path) if (self.adapter_path and self.adapter_path.exists()) else None
        self._model, self._tokenizer = load(str(self.model_path), adapter_path=adapter_str)

        self.load_duration_seconds = time.perf_counter() - t0
        self.memory_after_load_gb = mx.get_peak_memory() / 1e9

        # Validate special tokens in tokenizer
        for name, expected_id in SPECIAL_TOKEN_IDS.items():
            encoded = self._tokenizer.encode(name, add_special_tokens=False)
            if not (len(encoded) == 1 and encoded[0] == expected_id):
                raise ValueError(
                    f"Special token integrity mismatch for {name}: expected [{expected_id}], got {encoded}"
                )

    def generate(
        self,
        instruction: str,
        paragraph: Optional[str] = None,
        history: str = "",
    ) -> str:
        """Single generation matching the SelfRAGGenerator interface."""
        prompt = format_prompt(instruction, paragraph, history)
        t_start = time.perf_counter()

        text_pieces: List[str] = []
        token_count = 0
        stop_reason = "max_tokens"

        # Determine generation stopping rules
        is_retrieval_probe = (paragraph is None and not history)

        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
        ):
            token_count += 1
            text_pieces.append(resp.text)
            tok_id = resp.token

            # 1. Retrieval decision probe: stop if model decides [Retrieval]
            if is_retrieval_probe and self.stop_on_retrieval_probe and tok_id == SPECIAL_TOKEN_IDS[RETRIEVE_YES]:
                stop_reason = "retrieval_decision_retrieval"
                break

            # 2. Complete utility token emitted: stop generation immediately
            if self.stop_on_utility and tok_id in UTILITY_TOKEN_IDS:
                stop_reason = f"utility_token_{tok_id}"
                break

            # 3. Standard EOS
            if tok_id == EOS_TOKEN_ID:
                stop_reason = "eos"
                break

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.total_generations += 1
        self.generation_latencies_ms.append(latency_ms)
        self.token_counts.append(token_count)
        self.stop_reasons[stop_reason] = self.stop_reasons.get(stop_reason, 0) + 1

        raw_output = "".join(text_pieces)
        return raw_output

    def generate_batch(
        self,
        instruction: str,
        paragraphs: List[str],
        history: str = "",
    ) -> List[str]:
        """Generate one candidate continuation per passage."""
        return [self.generate(instruction, p, history) for p in paragraphs]

    def get_metrics(self) -> Dict[str, Any]:
        """Return runtime telemetry summary."""
        avg_lat = (
            sum(self.generation_latencies_ms) / len(self.generation_latencies_ms)
            if self.generation_latencies_ms
            else 0.0
        )
        return {
            "model_path": str(self.model_path),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "load_duration_seconds": round(self.load_duration_seconds, 2),
            "memory_after_load_gb": round(self.memory_after_load_gb, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
            "total_generations": self.total_generations,
            "average_latency_ms": round(avg_lat, 2),
            "stop_reasons": self.stop_reasons,
        }

