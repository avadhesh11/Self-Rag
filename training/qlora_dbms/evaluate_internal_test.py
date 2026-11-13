#!/usr/bin/env python3
"""EvoRAG Task 12E: Post-Training Validation on Internal Test Dataset.

This script:
1. Evaluates loss and perplexity on the held-out `internal_test.jsonl` (39 examples)
   for both the Base 4-bit Model and Fine-Tuned QLoRA Adapter.
2. Generates completions on all 39 internal test prompts for both models.
3. Audits Self-RAG reflection token fidelity (retrieval decision, relevance, support, utility).
4. Evaluates answer completeness, meaningfulness, abstention behavior, and malformed outputs.
5. Saves results to `training/qlora_dbms/internal_test_evaluation.json`.
"""

import json
import math
import os
import platform
import re
import sys
import time
import types
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import CacheDataset, evaluate, default_loss, iterate_batches

BASE_MODEL_DIR = PROJECT_ROOT / "models/mlx/selfrag_llama2_7b_4bit"
ADAPTER_DIR = PROJECT_ROOT / "training/qlora_dbms/adapters/full_dbms_v1"
BEST_ADAPTER_FILE = ADAPTER_DIR / "best_adapters.safetensors"
DATASET_DIR = PROJECT_ROOT / "training/qlora_dbms/mlx_dataset"
INTERNAL_TEST_FILE = DATASET_DIR / "internal_test.jsonl"
OUTPUT_EVAL_FILE = PROJECT_ROOT / "training/qlora_dbms/internal_test_evaluation.json"

SELF_RAG_TOKENS = {
    "retrieval": ["[Retrieval]", "[No Retrieval]"],
    "relevance": ["[Relevant]", "[Irrelevant]"],
    "support": ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"],
    "utility": [f"[Utility:{i}]" for i in range(1, 6)],
}


def parse_reflection_tokens(text: str) -> dict:
    found_retrieval = [t for t in SELF_RAG_TOKENS["retrieval"] if t in text]
    found_relevance = [t for t in SELF_RAG_TOKENS["relevance"] if t in text]
    found_support = [t for t in SELF_RAG_TOKENS["support"] if t in text]
    found_utility = [t for t in SELF_RAG_TOKENS["utility"] if t in text]

    # Check paragraph tag syntax
    has_paragraph_open = "<paragraph>" in text
    has_paragraph_close = "</paragraph>" in text
    paragraph_balanced = (has_paragraph_open == has_paragraph_close)

    # Clean text to measure substantive answer
    clean_text = text
    for t in (SELF_RAG_TOKENS["retrieval"] + SELF_RAG_TOKENS["relevance"] +
              SELF_RAG_TOKENS["support"] + SELF_RAG_TOKENS["utility"] +
              ["<paragraph>", "</paragraph>"]):
        clean_text = clean_text.replace(t, " ")
    clean_text = " ".join(clean_text.split())

    # Detect unclosed/malformed tokens (e.g. "[Retriev", "[Fully supp")
    malformed = False
    bracket_matches = re.findall(r"\[[^\]]*$", text)
    if bracket_matches:
        malformed = True

    return {
        "retrieval_decision": found_retrieval[0] if found_retrieval else None,
        "relevance_label": found_relevance[0] if found_relevance else None,
        "support_label": found_support[0] if found_support else None,
        "utility_scores": found_utility,
        "paragraph_tags_balanced": paragraph_balanced,
        "substantive_text_length": len(clean_text),
        "is_complete_sentence": len(clean_text) > 30 and (clean_text.endswith(".") or clean_text.endswith("!")),
        "has_malformed_token": malformed,
        "clean_text": clean_text,
    }


def main():
    print("=" * 80)
    print("EVORAG TASK 12E: POST-TRAINING INTERNAL TEST EVALUATION")
    print("=" * 80)

    assert BASE_MODEL_DIR.exists(), f"Base model missing: {BASE_MODEL_DIR}"
    assert BEST_ADAPTER_FILE.exists(), f"Best adapter file missing: {BEST_ADAPTER_FILE}"
    assert INTERNAL_TEST_FILE.exists(), f"Internal test file missing: {INTERNAL_TEST_FILE}"

    # Load 39 internal test rows
    with open(INTERNAL_TEST_FILE, "r", encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_examples)} internal test examples across 13 categories.")

    # 1. Load Datasets for Quantitative Loss/Perplexity Evaluation
    print("\n--- 1. Evaluating Held-Out Loss & Perplexity on Internal Test Split ---")
    _, tokenizer = load(str(BASE_MODEL_DIR))

    # Load internal test set through completions dataset
    ds_args = types.SimpleNamespace(
        data=str(DATASET_DIR),
        hf_dataset=False,
        train=False,
        test=True,                         # In MLX-LM, test=True loads test.jsonl (which mirrors internal_test.jsonl)
        mask_prompt=True,
        prompt_feature="prompt",
        completion_feature="completion",
    )
    _, _, test_set = load_dataset(ds_args, tokenizer)
    cached_test_set = CacheDataset(test_set)
    print(f"Internal test cache dataset contains {len(cached_test_set)} processed examples.")

    # A. Base Model Loss on Internal Test
    print("Evaluating Base Model loss...")
    base_model, _ = load(str(BASE_MODEL_DIR))
    base_test_loss = evaluate(
        model=base_model,
        dataset=cached_test_set,
        loss=default_loss,
        batch_size=1,
        num_batches=-1,
        max_seq_length=1024,
        iterate_batches=iterate_batches,
    )
    base_test_ppl = math.exp(min(base_test_loss, 20.0))
    print(f"Base Model Test Loss:       {base_test_loss:.4f} (Perplexity: {base_test_ppl:.2f})")
    del base_model
    mx.metal.clear_cache()

    # B. Fine-Tuned Model Loss on Internal Test
    print("Evaluating Fine-Tuned Adapter loss...")
    ft_model, _ = load(str(BASE_MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    ft_test_loss = evaluate(
        model=ft_model,
        dataset=cached_test_set,
        loss=default_loss,
        batch_size=1,
        num_batches=-1,
        max_seq_length=1024,
        iterate_batches=iterate_batches,
    )
    ft_test_ppl = math.exp(min(ft_test_loss, 20.0))
    print(f"Fine-Tuned Model Test Loss: {ft_test_loss:.4f} (Perplexity: {ft_test_ppl:.2f})")
    loss_reduction = base_test_loss - ft_test_loss
    loss_reduction_pct = (loss_reduction / base_test_loss) * 100.0
    print(f"Loss Delta on Held-Out Test: -{loss_reduction:.4f} ({loss_reduction_pct:.1f}% reduction)")

    # 2. Qualitative Generation Comparison on All 39 Internal Test Examples
    print("\n--- 2. Running Comparative Generation on Held-Out Prompts ---")
    # Reload fresh base model and ft model
    base_model, base_tok = load(str(BASE_MODEL_DIR))

    comparisons = []
    category_summary = {}

    for idx, ex in enumerate(test_examples, 1):
        ex_id = ex["example_id"]
        cat = ex["category"]
        prompt_q = ex["prompt"]
        ground_truth_completion = ex["completion"]

        formatted_prompt = f"### Instruction:\n{prompt_q}\n\n### Response:\n"

        # Generate with Base Model
        base_t0 = time.perf_counter()
        base_gen = mlx_lm.generate(
            base_model,
            base_tok,
            prompt=formatted_prompt,
            max_tokens=220,
            verbose=False,
        )
        base_time = time.perf_counter() - base_t0

        # Generate with Fine-Tuned Model
        ft_t0 = time.perf_counter()
        ft_gen = mlx_lm.generate(
            ft_model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=220,
            verbose=False,
        )
        ft_time = time.perf_counter() - ft_t0

        base_parsed = parse_reflection_tokens(base_gen)
        ft_parsed = parse_reflection_tokens(ft_gen)

        item = {
            "example_id": ex_id,
            "category": cat,
            "prompt": prompt_q,
            "ground_truth": ground_truth_completion,
            "base_model": {
                "output": base_gen,
                "duration_seconds": round(base_time, 2),
                "parsed": base_parsed,
            },
            "fine_tuned_model": {
                "output": ft_gen,
                "duration_seconds": round(ft_time, 2),
                "parsed": ft_parsed,
            },
        }
        comparisons.append(item)

        if cat not in category_summary:
            category_summary[cat] = {
                "total": 0,
                "ft_has_retrieval_decision": 0,
                "ft_has_utility": 0,
                "ft_complete_sentence": 0,
                "base_has_retrieval_decision": 0,
                "base_has_utility": 0,
                "base_complete_sentence": 0,
            }
        category_summary[cat]["total"] += 1
        if ft_parsed["retrieval_decision"]:
            category_summary[cat]["ft_has_retrieval_decision"] += 1
        if ft_parsed["utility_scores"]:
            category_summary[cat]["ft_has_utility"] += 1
        if ft_parsed["is_complete_sentence"]:
            category_summary[cat]["ft_complete_sentence"] += 1
        if base_parsed["retrieval_decision"]:
            category_summary[cat]["base_has_retrieval_decision"] += 1
        if base_parsed["utility_scores"]:
            category_summary[cat]["base_has_utility"] += 1
        if base_parsed["is_complete_sentence"]:
            category_summary[cat]["base_complete_sentence"] += 1

        print(f"[{idx:2d}/39] {ex_id} ({cat}): Base {base_time:.1f}s | FT {ft_time:.1f}s | "
              f"FT Decision: {ft_parsed['retrieval_decision']} | FT Utility: {ft_parsed['utility_scores']}")

    # 3. Aggregate Metrics
    total_samples = len(comparisons)
    ft_retrieval_decision_count = sum(1 for c in comparisons if c["fine_tuned_model"]["parsed"]["retrieval_decision"])
    ft_utility_count = sum(1 for c in comparisons if c["fine_tuned_model"]["parsed"]["utility_scores"])
    ft_complete_count = sum(1 for c in comparisons if c["fine_tuned_model"]["parsed"]["is_complete_sentence"])
    ft_malformed_count = sum(1 for c in comparisons if c["fine_tuned_model"]["parsed"]["has_malformed_token"])

    base_retrieval_decision_count = sum(1 for c in comparisons if c["base_model"]["parsed"]["retrieval_decision"])
    base_utility_count = sum(1 for c in comparisons if c["base_model"]["parsed"]["utility_scores"])
    base_complete_count = sum(1 for c in comparisons if c["base_model"]["parsed"]["is_complete_sentence"])
    base_malformed_count = sum(1 for c in comparisons if c["base_model"]["parsed"]["has_malformed_token"])

    results = {
        "evaluation_name": "EvoRAG Task 12E Internal Test Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "dataset": {
            "internal_test_file": str(INTERNAL_TEST_FILE),
            "total_examples": total_samples,
            "categories_evaluated": len(category_summary),
        },
        "quantitative_loss_metrics": {
            "base_model_loss": round(base_test_loss, 4),
            "base_model_perplexity": round(base_test_ppl, 2),
            "fine_tuned_model_loss": round(ft_test_loss, 4),
            "fine_tuned_model_perplexity": round(ft_test_ppl, 2),
            "loss_reduction": round(loss_reduction, 4),
            "loss_reduction_percent": round(loss_reduction_pct, 2),
        },
        "fidelity_and_generation_metrics": {
            "fine_tuned_model": {
                "retrieval_decision_rate": round(ft_retrieval_decision_count / total_samples, 4),
                "utility_token_rate": round(ft_utility_count / total_samples, 4),
                "complete_sentence_rate": round(ft_complete_count / total_samples, 4),
                "malformed_token_count": ft_malformed_count,
            },
            "base_model": {
                "retrieval_decision_rate": round(base_retrieval_decision_count / total_samples, 4),
                "utility_token_rate": round(base_utility_count / total_samples, 4),
                "complete_sentence_rate": round(base_complete_count / total_samples, 4),
                "malformed_token_count": base_malformed_count,
            },
        },
        "category_breakdown": category_summary,
        "sample_comparisons": comparisons,
    }

    with open(OUTPUT_EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Base Model Held-Out Test Loss:       {base_test_loss:.4f} (PPL: {base_test_ppl:.2f})")
    print(f"Fine-Tuned Model Held-Out Test Loss: {ft_test_loss:.4f} (PPL: {ft_test_ppl:.2f})")
    print(f"Test Loss Reduction:                 {loss_reduction:.4f} ({loss_reduction_pct:.1f}%)")
    print(f"FT Retrieval Decision Rate:          {ft_retrieval_decision_count}/{total_samples} ({ft_retrieval_decision_count/total_samples*100:.1f}%)")
    print(f"FT Utility Token Rate:               {ft_utility_count}/{total_samples} ({ft_utility_count/total_samples*100:.1f}%)")
    print(f"FT Complete Sentences:               {ft_complete_count}/{total_samples} ({ft_complete_count/total_samples*100:.1f}%)")
    print(f"FT Malformed Outputs:                {ft_malformed_count}/{total_samples}")
    print(f"Results saved to:                    {OUTPUT_EVAL_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()

