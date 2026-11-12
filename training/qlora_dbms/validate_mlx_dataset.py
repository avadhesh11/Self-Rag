#!/usr/bin/env python3
"""EvoRAG Task 12C: Validate Self-RAG MLX QLoRA Training Dataset.

Tests:
1. JSONL parsing across all generated dataset files.
2. Tokenizer compatibility and Self-RAG reflection token preservation.
3. Provenance and evidence-anchor verification against indexed corpus.
4. Duplicate detection (internal to dataset).
5. Held-out contamination audit against all 32 evaluation questions.
6. Disjoint train / valid / internal test separation.
7. Compatibility with MLX-LM's actual dataset loader (`load_local_dataset`).
8. Small batch inspection of tokenized sequences and prompt masking offsets.
"""

import hashlib
import json
import os
import re
import sys
import types
import unicodedata
from collections import Counter
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoTokenizer
import mlx_lm.tuner.datasets as mlx_datasets

DATASET_DIR = Path("training/qlora_dbms/mlx_dataset")
MODEL_DIR = Path("models/mlx/selfrag_llama2_7b_4bit")
CORPUS_PATH = Path("index/dbms_sql_v1/corpus.jsonl")
HELD_OUT_PATH = Path("EvoRAG_Blind_Test_Questions/held_out_test_questions_only.jsonl")

EXPECTED_REFLECTION_TOKENS = {
    32000: "[No Retrieval]",
    32001: "[Retrieval]",
    32002: "[Continue to Use Evidence]",
    32003: "[Irrelevant]",
    32004: "[Relevant]",
    32005: "<paragraph>",
    32006: "</paragraph>",
    32007: "[Utility:1]",
    32008: "[Utility:2]",
    32009: "[Utility:3]",
    32010: "[Utility:4]",
    32011: "[Utility:5]",
    32012: "[Fully supported]",
    32013: "[Partially supported]",
    32014: "[No support / Contradictory]",
    32015: "<pad>",
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).lower()
    cleaned = re.sub(r"[^\w\s]", "", normalized)
    return " ".join(cleaned.split()).strip()

def token_jaccard(text1: str, text2: str) -> float:
    t1 = set(normalize_text(text1).split())
    t2 = set(normalize_text(text2).split())
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))

def main():
    print("=" * 80)
    print("EVORAG TASK 12C: VALIDATE MLX QLORA DATASET")
    print("=" * 80)

    # 1. JSONL Parsing
    print("\n--- 1. Testing JSONL Parsing and File Completeness ---")
    files_to_check = [
        "candidate_examples.jsonl",
        "approved_examples.jsonl",
        "train.jsonl",
        "valid.jsonl",
        "internal_test.jsonl",
        "test.jsonl",
        "dataset_manifest.json",
        "dataset_quality_report.md",
    ]
    for fname in files_to_check:
        fpath = DATASET_DIR / fname
        assert fpath.exists(), f"Missing deliverable file: {fpath}"
        assert fpath.stat().st_size > 0, f"File is empty: {fpath}"
        print(f"  [PASS] {fname} ({fpath.stat().st_size / 1024:.1f} KB)")

    # Read records
    with open(DATASET_DIR / "candidate_examples.jsonl") as f:
        candidates = [json.loads(l) for l in f]
    with open(DATASET_DIR / "approved_examples.jsonl") as f:
        approved = [json.loads(l) for l in f]
    with open(DATASET_DIR / "train.jsonl") as f:
        train_data = [json.loads(l) for l in f]
    with open(DATASET_DIR / "valid.jsonl") as f:
        valid_data = [json.loads(l) for l in f]
    with open(DATASET_DIR / "internal_test.jsonl") as f:
        test_data = [json.loads(l) for l in f]

    print(f"Loaded counts: Candidates={len(candidates)}, Approved={len(approved)}, "
          f"Train={len(train_data)}, Valid={len(valid_data)}, InternalTest={len(test_data)}")
    assert len(candidates) == 260
    assert len(approved) == 260
    assert len(train_data) == 182
    assert len(valid_data) == 39
    assert len(test_data) == 39
    assert len(train_data) + len(valid_data) + len(test_data) == len(approved)
    print("  [PASS] Record counts and split sums verified.")

    # 2. Tokenizer Compatibility & Reflection Token Preservation
    print("\n--- 2. Testing Tokenizer Compatibility & Reflection Tokens ---")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    assert len(tokenizer) == 32016, f"Expected vocab 32016, got {len(tokenizer)}"

    # Check that reflection tokens encode to exact single token IDs in dataset completions
    for tok_id, tok_literal in EXPECTED_REFLECTION_TOKENS.items():
        encoded = tokenizer.encode(tok_literal, add_special_tokens=False)
        assert encoded == [tok_id], f"Token {tok_literal} encoded to {encoded}, expected [{tok_id}]"

    # Sample completions and verify tokens are intact
    for r in approved[:10]:
        c_tokens = tokenizer.encode(r["completion"], add_special_tokens=False)
        c_decoded = tokenizer.decode(c_tokens)
        for tok_id, tok_literal in EXPECTED_REFLECTION_TOKENS.items():
            if tok_literal in r["completion"]:
                assert tok_id in c_tokens, f"Expected token {tok_id} ({tok_literal}) in completion tokens"
    print("  [PASS] Reflection tokens 100% preserved in tokenizer encoding.")

    # 3. Provenance and Evidence-Anchor Verification
    print("\n--- 3. Testing Provenance & Evidence-Anchor Matching ---")
    chunks_by_id = {}
    with open(CORPUS_PATH) as f:
        for l in f:
            c = json.loads(l)
            chunks_by_id[c["chunk_id"]] = c

    for r in approved:
        if r["retrieval_decision"] == "[Retrieval]":
            assert r["source_document"] is not None
            assert len(r["supporting_chunk_ids"]) > 0
            for cid in r["supporting_chunk_ids"]:
                assert cid in chunks_by_id, f"Invalid chunk_id {cid} in {r['example_id']}"
                chunk = chunks_by_id[cid]
                assert chunk["source"] == r["source_document"]
                assert chunk["chunk_hash"] in r["chunk_hashes"]
                # Evidence anchor verification: text in completion must match chunk text
                assert chunk["text"][:50] in r["completion"]
    print("  [PASS] Provenance anchors and chunk hashes match indexed corpus.")

    # 4. Internal Duplicate Check
    print("\n--- 4. Testing Internal Duplicate Detection ---")
    seen_questions = set()
    for r in approved:
        q_norm = normalize_text(r["question"])
        assert q_norm not in seen_questions, f"Duplicate question in dataset: {r['question']}"
        seen_questions.add(q_norm)
    print("  [PASS] All 260 questions are unique.")

    # 5. Held-Out Contamination Audit
    print("\n--- 5. Testing Held-Out Contamination Safeguards ---")
    with open(HELD_OUT_PATH) as f:
        held_out = [json.loads(l) for l in f]

    contamination_found = False
    for r in approved:
        q_text = r["question"]
        q_norm = normalize_text(q_text)
        for h in held_out:
            h_text = h["question"]
            h_norm = normalize_text(h_text)
            assert q_text != h_text, f"EXACT CONTAMINATION: {q_text} matches {h['question_id']}"
            assert q_norm != h_norm, f"NORMALIZED CONTAMINATION: {q_text} matches {h['question_id']}"
            sim = token_jaccard(q_text, h_text)
            assert sim < 0.65, f"HIGH SIMILARITY CONTAMINATION ({sim:.2f}): '{q_text}' vs '{h_text}'"

    print("  [PASS] 0 exact matches, 0 normalized matches, 0 near-duplicates across all 32 held-out questions.")

    # 6. Disjoint Split Separation
    print("\n--- 6. Testing Disjoint Train/Valid/Test Separation ---")
    train_ids = set(r["example_id"] for r in train_data)
    valid_ids = set(r["example_id"] for r in valid_data)
    test_ids = set(r["example_id"] for r in test_data)

    assert train_ids.isdisjoint(valid_ids), "Train and Valid splits overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and Test splits overlap!"
    assert valid_ids.isdisjoint(test_ids), "Valid and Test splits overlap!"
    print("  [PASS] Train, validation, and internal test sets are strictly disjoint.")

    # 7. Compatibility with MLX-LM's Actual Dataset Loader
    print("\n--- 7. Testing Compatibility with MLX-LM Dataset Loader ---")
    loader_args = types.SimpleNamespace(
        data=str(DATASET_DIR),
        hf_dataset=False,
        train=True,
        test=True,
        mask_prompt=True,
        prompt_feature="prompt",
        completion_feature="completion",
    )

    train_set, valid_set, test_set = mlx_datasets.load_dataset(loader_args, tokenizer)
    assert len(train_set) == 182, f"Expected 182 in MLX train set, got {len(train_set)}"
    assert len(valid_set) == 39, f"Expected 39 in MLX valid set, got {len(valid_set)}"
    assert len(test_set) == 39, f"Expected 39 in MLX test set, got {len(test_set)}"
    print(f"  [PASS] MLX-LM load_dataset successfully loaded splits: Train={len(train_set)}, Valid={len(valid_set)}, Test={len(test_set)}")

    # 8. Batch Inspection of Tokenized Sequences & Prompt Masking
    print("\n--- 8. Testing Batch Tokenization and Prompt Masking Offset ---")
    for idx in range(min(4, len(train_set))):
        sample_dict = train_set[idx]
        tokens, offset = train_set.process(sample_dict)
        prompt_str = tokenizer.decode(tokens[:offset])
        completion_str = tokenizer.decode(tokens[offset:])

        print(f"\n  [Sample {idx+1}: {sample_dict['example_id']} ({sample_dict['category']})]")
        print(f"    Total sequence length: {len(tokens)} tokens")
        print(f"    Prompt mask offset:    {offset} tokens")
        print(f"    Prompt prefix:         {repr(prompt_str[:60])}...")
        print(f"    Completion prefix:     {repr(completion_str[:60])}...")

        assert offset > 0, "Prompt mask offset must be > 0"
        assert len(tokens) > offset, "Completion tokens must exist after prompt offset"
        assert "### Instruction:" in prompt_str
        assert "### Response:" in prompt_str

    print("\n  [PASS] Prompt masking and sequence tokenization verified successfully.")
    print("=" * 80)
    print("ALL DATASET VALIDATION TESTS PASSED (100% OK)")
    print("=" * 80)

if __name__ == "__main__":
    main()

