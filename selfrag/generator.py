"""
Generator backends for selfrag/selfrag_llama2_7b, one interface.

The official checkpoint is a plain LlamaForCausalLM (model_type=llama,
vocab_size 32016 = llama-2's 32000 + the 16 reflection tokens), so it runs
under two interchangeable backends:

  * llama.cpp (DEFAULT) -- a local GGUF conversion, Metal-accelerated on
    Apple silicon, CPU elsewhere. This is the only way to run the
    checkpoint on a Mac: vLLM ships Linux+NVIDIA builds only, there is no
    macOS wheel at all (that is the metadata error pip shows when it
    tries to install it here).
  * vLLM -- loads the HF checkpoint selfrag/selfrag_llama2_7b directly.
    Linux + NVIDIA GPU only (~14GB+ VRAM for fp16).

Select the backend with the SELFRAG_BACKEND env var ("llamacpp" |
"vllm", default "llamacpp"). Prompt format and sampling settings are
copied verbatim from the official model card's usage example, because the
checkpoint was fine-tuned to this exact template -- deviating from it
(e.g. different instruction wrapper, different passage delimiter) means
the reflection tokens it emits are no longer meaningful.
"""
import os

MODEL_NAME = "selfrag/selfrag_llama2_7b"

# Local GGUF default: download with
#   hf download QuantFactory/selfrag_llama2_7b-GGUF selfrag_llama2_7b.Q4_K_M.gguf --local-dir models/
# Q4_K_M is the right quality/size tradeoff for an 18GB Mac; Q8_0
# (6.8 GB) is in the same repo if you want maximum fidelity instead.
DEFAULT_GGUF = "models/selfrag_llama2_7b.Q4_K_M.gguf"

# Sampling settings from the official model card, identical across
# backends: greedy decoding, no nucleus cutoff, 150 new tokens, and
# critically skip_special_tokens=False -- without it the reflection tokens
# are stripped from the output before you ever see them.
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 150


def format_prompt(instruction: str, paragraph: str | None = None) -> str:
    """Exact template from the official model card."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    if paragraph is not None:
        prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
    return prompt


class SelfRAGGenerator:
    """Unified generator over both backends. `model` is a .gguf path
    (llamacpp backend) or a Hugging Face id (vllm backend)."""

    def __init__(self, model: str | None = None, *,
                 n_ctx: int = 4096, n_gpu_layers: int = -1,
                 dtype: str = "half", download_dir: str | None = None):
        backend = os.environ.get("SELFRAG_BACKEND", "llamacpp")
        self._backend = backend
        if backend == "llamacpp":
            self._init_llamacpp(model or DEFAULT_GGUF, n_ctx, n_gpu_layers)
        elif backend == "vllm":
            self._init_vllm(model or MODEL_NAME, dtype, download_dir)
        else:
            raise ValueError(
                f"Unknown SELFRAG_BACKEND={backend!r}; use 'llamacpp' or 'vllm'")

    # --- backend constructors ---

    def _init_llamacpp(self, model_path: str, n_ctx: int, n_gpu_layers: int):
        from llama_cpp import Llama

        # n_gpu_layers=-1 offloads every layer to Metal on Apple silicon
        # (set 0 to force CPU). n_ctx is the checkpoint's native 4096.
        self._llm = Llama(
            model_path=model_path, n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers, verbose=False,
        )

    def _init_vllm(self, model_name: str, dtype: str, download_dir: str | None):
        from vllm import LLM, SamplingParams

        self._llm = LLM(model_name, download_dir=download_dir, dtype=dtype)
        self._sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            skip_special_tokens=False,
        )

    # --- the interface pipeline.py depends on ---

    def generate(self, instruction: str, paragraph: str | None = None) -> str:
        """Single forward generation. Returns raw text INCLUDING reflection
        tokens -- callers parse those out via reflection_tokens.py."""
        prompt = format_prompt(instruction, paragraph)
        if self._backend == "llamacpp":
            out = self._llm(
                prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                echo=False,
            )
            # The low-level completion API detokenizes every normal vocab
            # token; reflection tokens are plain added tokens (not flagged
            # special in added_tokens.json), so they come through as text.
            # Stripping only happens in the chat-format layer, which we
            # deliberately do not use.
            return out["choices"][0]["text"]
        outputs = self._llm.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(self, instruction: str, paragraphs: list[str]) -> list[str]:
        """Generate one candidate continuation per passage. vLLM does this
        in a single batched call -- this is the 'parallel generation across
        retrieved passages' step from the paper. The llama.cpp backend
        completes sequentially instead; k <= 5 passages at Q4 on an
        M-series chip is a few seconds each, fine for a demo trace."""
        if self._backend == "llamacpp":
            return [self.generate(instruction, p) for p in paragraphs]
        prompts = [format_prompt(instruction, p) for p in paragraphs]
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [o.outputs[0].text for o in outputs]
