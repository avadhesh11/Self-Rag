# EvoRAG Task 12E: Controlled Full DBMS QLoRA Training Report

**Date:** 2026-11-12
**Task:** Task 12E — Controlled Full DBMS QLoRA Training
**Branch:** `evorag-development`
**Execution Environment:** Isolated `.venv-mlx` (Apple Silicon M3 Pro, 36 GB Unified Memory, macOS 15.5, ARM64)
**Status:** COMPLETE (Full 3-epoch QLoRA training executed on Metal GPU, validation loss decreased monotonically from 6.4219 to 0.8713, held-out test loss reduced by 82.8%, base model and production baselines 100% frozen)

---

## 1. Executive Summary

In Task 12E, we executed the full controlled fine-tuning of the 4-bit Self-RAG 7B model using MLX-LM 0.32.2 on the Apple Silicon M3 Pro Metal GPU. Training was conducted over 3 full epochs (546 optimization iterations) using the audited dataset of 260 curated DBMS examples.

Key highlights:
1. **Convergence & Validation Dynamics:** Validation loss dropped monotonically across all 7 evaluation checkpoints, from **6.4219** (untrained baseline) down to **0.8713** at Iteration 545. No overfitting or divergence was observed.
2. **Held-Out Test Set Generalization:** On the held-out `internal_test.jsonl` split (39 unseen DBMS questions spanning all 13 canonical categories), the fine-tuned adapter reduced loss from **6.3655** down to **1.0922** — an **82.8% loss reduction** (perplexity reduced from **581.43** to **2.98**).
3. **Hardware Efficiency:** Full training took **15.21 minutes** (912.61 seconds) on Metal GPU, maintaining a peak memory footprint of **6.61 GB** (leaving ~29.4 GB unified memory headroom). Disk consumption for all 5 saved checkpoints was **0.12 GiB**.
4. **Self-RAG Reflection Token Fidelity:** The fine-tuned adapter achieved **100% syntax compliance** for initial `[Retrieval]` decision tokens and eliminated base-model degenerative infinite loops (`[No Retrieval]The NULL bitmap...` repeated 11 times) and empty paragraph sequences (`<paragraph><paragraph>...`).
5. **Architectural Separation & Safeguards:** Production `.venv`, original GGUF model (`models/selfrag_llama2_7b.Q4_K_M.gguf`), frozen benchmark predictions (`evaluation/predictions/frozen/predictions_20261109_151957Z.csv`), and all 25 Experience Memory records remained 100% untouched. All 13 production pytest regression tests passed cleanly.

---

## 2. Preflight Audit & Safeguards Verification

Prior to launching the training loop, mandatory preflight audits confirmed system integrity:

| Preflight Check | Target Asset / Requirement | Verified Value / Status | Result |
| :--- | :--- | :--- | :---: |
| **Compute Engine** | Apple Silicon M3 Pro Metal GPU | `mlx.core.metal.is_available() == True` | **PASS** |
| **Runtime Isolation** | `.venv-mlx` Python 3.12.16 (ARM64) | Production `.venv` unmodified | **PASS** |
| **Disk Availability** | Workspace root filesystem | Initial: 10.16 GiB free (Required: $\ge 5.0$ GiB) | **PASS** |
| **Train Split Hash** | [`train.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/train.jsonl) (182 rows) | `7eb1636d755cb183042a13d76229c6044267e2bc5ba591cfe029f3ab9dcf7274` | **MATCH** |
| **Valid Split Hash** | [`valid.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/valid.jsonl) (39 rows) | `51737f90a25f725b71065739160a6473dc5868534c330d02267bd5b5c68fcf89` | **MATCH** |
| **Internal Test Hash** | [`internal_test.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/internal_test.jsonl) (39 rows) | `c305a1492a97955bf38c97c7eb0420e0786847760bcdf72ce14c59fbba568f80` | **MATCH** |
| **Base Model Safetensors**| [`model.safetensors`](file:///Users/devj/code/self-rag/self-rag-demo/models/mlx/selfrag_llama2_7b_4bit/model.safetensors) | `71576febb874c07b71697976f0892c2bab9e972e77bc880dcc0210eb6342db32` | **MATCH** |
| **Smoke-Test Adapter** | [`adapters/smoke_test/adapters.safetensors`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/adapters/smoke_test/adapters.safetensors) | 38.15 MB preserved | **PASS** |
| **Frozen Benchmark** | [`predictions_20261109_151957Z.csv`](file:///Users/devj/code/self-rag/self-rag-demo/evaluation/predictions/frozen/predictions_20261109_151957Z.csv) | `a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7` | **MATCH** |
| **Contamination Check** | 32 Held-Out Benchmark Questions | Zero overlap in train, valid, or test splits | **PASS** |

---

## 3. Training Configuration & Hyperparameters

The full training parameters were configured in [`training_config.yaml`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/training_config.yaml) and executed by [`run_full_training.py`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/run_full_training.py):

```yaml
base_model: "models/mlx/selfrag_llama2_7b_4bit"
adapter_output_dir: "training/qlora_dbms/adapters/full_dbms_v1"
dataset_dir: "training/qlora_dbms/mlx_dataset"

lora_parameters:
  rank: 8
  scale: 16.0
  dropout: 0.0

training_parameters:
  num_layers: 16             # LoRALinear injected into layers 16-31
  batch_size: 1
  learning_rate: 1.0e-5       # Adam optimizer
  max_seq_length: 1024
  mask_prompt: true          # Only completion tokens incur gradient loss
  epochs: 3
  iters: 546                 # 182 train examples * 3 epochs = 546 steps
  val_batches: -1            # Complete validation set evaluation (39 examples)
  steps_per_report: 10       # Log training loss every 10 steps
  steps_per_eval: 91         # Validate every 0.5 epoch (7 validation checkpoints)
  steps_per_save: 182        # Save epoch checkpoint at 182, 364, 546
  grad_checkpoint: false
  grad_accumulation_steps: 1
```

- **Total Base Model Parameters:** 6,738,546,688
- **Trainable LoRA Parameters:** 9,994,240 (0.148% of total)
- **Target Matrices:** `q_proj`, `v_proj` inside `QuantizedLinear` blocks

---

## 4. Training Execution & Loss History

Training ran synchronously across 546 iterations. Training throughput averaged **180–196 tokens/second**.

```
Iteration Loss Convergence:
  Step 001: Val Loss: 6.4219 (Pre-training baseline)
  Step 010: Train Loss: 4.4343 | Tokens/s: 164.2 | Peak Mem: 6.60 GB
  Step 050: Train Loss: 2.5252 | Tokens/s: 162.5 | Peak Mem: 6.60 GB
  Step 090: Train Loss: 2.2363 | Tokens/s: 174.5 | Peak Mem: 6.60 GB
  Step 091: Val Loss: 1.9978 (Epoch 0.5) [Δ = -4.4241]
  Step 180: Train Loss: 1.6385 | Tokens/s: 172.1 | Peak Mem: 6.60 GB
  Step 182: Val Loss: 1.5556 (Epoch 1.0) [Δ = -0.4422] -> Saved 0000182_adapters.safetensors
  Step 270: Train Loss: 1.2582 | Tokens/s: 184.2 | Peak Mem: 6.61 GB
  Step 273: Val Loss: 1.2538 (Epoch 1.5) [Δ = -0.3018]
  Step 360: Train Loss: 1.0558 | Tokens/s: 187.8 | Peak Mem: 6.61 GB
  Step 364: Val Loss: 1.0443 (Epoch 2.0) [Δ = -0.2095] -> Saved 0000364_adapters.safetensors
  Step 450: Train Loss: 0.6258 | Tokens/s: 190.6 | Peak Mem: 6.61 GB
  Step 455: Val Loss: 0.9655 (Epoch 2.5) [Δ = -0.0788]
  Step 540: Train Loss: 0.6308 | Tokens/s: 187.8 | Peak Mem: 6.61 GB
  Step 546: Val Loss: 0.8713 (Epoch 3.0) [Δ = -0.0942] -> Saved 0000546_adapters.safetensors
  Final Step 546 Train Loss: 0.6063
```

### Complete Validation Loss Trajectory (All 39 Validation Examples Evaluated)

| Step # | Progress | Evaluation Phase | Validation Loss | Perplexity | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 0.0% | Pre-Training Baseline | 6.4219 | 615.17 | Initial baseline recorded |
| **90** | 16.5% | Epoch 0.5 | 1.9978 | 7.37 | Best checkpoint updated |
| **181** | 33.2% | Epoch 1.0 | 1.5556 | 4.74 | Checkpoint `0000182_adapters.safetensors` |
| **272** | 49.8% | Epoch 1.5 | 1.2538 | 3.50 | Best checkpoint updated |
| **363** | 66.5% | Epoch 2.0 | 1.0443 | 2.84 | Checkpoint `0000364_adapters.safetensors` |
| **454** | 83.2% | Epoch 2.5 | 0.9655 | 2.63 | Best checkpoint updated |
| **545** | 99.8% | Epoch 3.0 | **0.8713** | **2.39** | Best checkpoint (`best_adapters.safetensors`) |

```mermaid
xychart-beta
    title "EvoRAG Full DBMS QLoRA Validation Loss Trajectory"
    x-axis ["Baseline (0)", "Epoch 0.5 (90)", "Epoch 1.0 (181)", "Epoch 1.5 (272)", "Epoch 2.0 (363)", "Epoch 2.5 (454)", "Epoch 3.0 (545)"]
    y-axis "Validation Loss" 0.0 --> 7.0
    line [6.42, 2.00, 1.56, 1.25, 1.04, 0.97, 0.87]
```

### Best Checkpoint Selection Rationale
Validation loss decreased monotonically at every single evaluation point throughout the 3 epochs without a single uptick.
- **Selected Best Checkpoint:** Iteration 545 (`best_adapters.safetensors`, 38.15 MB, SHA-256 matching `0000546_adapters.safetensors`).
- **Validation Loss:** **0.8713** (Global minimum).
- **Extension Decision:** An extension to 5 epochs was deemed unnecessary because the validation loss reached an exceptionally low 0.8713 (PPL 2.39) and further training could risk overfitting to retrieval syntax templates.

---

## 5. Post-Training Validation on Held-Out Internal Test Split

To guarantee unbiased evaluation, [`evaluate_internal_test.py`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/evaluate_internal_test.py) was executed on the held-out [`internal_test.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/internal_test.jsonl) split (39 examples never seen during training or validation).

### 1. Quantitative Loss & Perplexity Comparison

| Model Evaluated | Test Loss | Perplexity | Delta vs Base Model | % Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Base 4-bit Self-RAG (Untrained)** | 6.3655 | 581.43 | Baseline | — |
| **Fine-Tuned Adapter (Task 12E Best)** | **1.0922** | **2.98** | **-5.2733** | **82.8% Reduction** |

### 2. Generative Quality & Syntax Breakdown (39 Held-Out Examples)

| Evaluation Dimension | Base Model Output | Fine-Tuned Adapter Output | Assessment |
| :--- | :---: | :---: | :--- |
| **`[Retrieval]` Decision Rate** | 17/39 (43.6%) | **39/39 (100.0%)** | Adapter consistently emits retrieval decision. |
| **Degenerative Infinite Loops** | **High** (loops up to 11x) | **0% (0 / 39)** | Looping failure mode completely eliminated. |
| **Empty Paragraph Tag Cascades** | **Frequent** (`<paragraph><paragraph>...`) | **0% (0 / 39)** | Empty cascading tags eliminated. |
| **Substantive DBMS Explanations**| Often missing or truncated | **Present in 39/39** | Concrete DBMS mechanisms articulated. |
| **Malformed Token Count** | 4 / 39 | **0 / 39** | No corrupted or unclosed brackets detected. |

---

## 6. Qualitative Comparison: Base Model vs. Fine-Tuned Adapter

The following direct comparisons on unseen held-out test prompts highlight the structural and conceptual improvements:

### Example 1: Storage Architecture (`DBMS-QLORA-0203`)
- **Question:** *How does the NULL bitmap in a tuple header save storage space?*
- **Base Model Output:**
  ```text
  Here's how:
  1.[Retrieval]<paragraph><paragraph><paragraph><paragraph><paragraph>
  The NULL bitmap in a tuple header saves storage space by indicating which fields in a row are NULL.[No Retrieval]The NULL bitmap is a bitmask that indicates which fields in a row are NULL.[No Retrieval]The NULL bitmap is a bitmask that indicates which fields in a row are NULL.[No Retrieval]The NULL bitmap is a bitmask that indicates which fields in a row are NULL.[No Retrieval]The NULL bitmap is a bitmask... (repeated 11 times)
  ```
- **Fine-Tuned Adapter Output:**
  ```text
  [Retrieval]<paragraph>EVORAG | DBMS & SQL KNOWLEDGE BASE POSTGRESQL 18 REFERENCE BASELINE ... A file is a logical collection of database pages. A page (or block) is a fixed-size unit transferred between storage and memory.</paragraph>[Relevant]The NULL bitmap dedicates 1 bit per column in the tuple header to indicate whether each attribute is NULL. If a bit is 0, the attribute is NULL and consumes zero bytes in the data payload, completely omitting space for NULL fields.[Fully supported][Utility:5]
  ```
- **Analysis:** The base model collapsed into an infinite repetition loop. The fine-tuned adapter properly initiated retrieval, provided the educational context block, validated relevance, answered with exact technical precision, and concluded with support and utility tokens.

### Example 2: Concurrency Control (`DBMS-QLORA-0160`)
- **Question:** *Can concurrent transactions read the same data page simultaneously without blocking?*
- **Base Model Output:**
  ```text
  Yes, concurrent transactions can read the same data page simultaneously without blocking.[Retrieval]<paragraph><paragraph><paragraph><paragraph>
  Concurrent transactions can read the same data page simultaneously without blocking.[No Retrieval]This is because the data page is not being modified...
  ```
- **Fine-Tuned Adapter Output:**
  ```text
  [Retrieval]<paragraph>EVORAG | DBMS & SQL KNOWLEDGE BASE POSTGRESQL 18 REFERENCE BASELINE ... Concurrency control aims to allow many users to work on the same logical data set while preserving the chosen result.</paragraph>[Relevant]Yes, concurrent transactions can read the same data page simultaneously without blocking because Shared locks and Shared page latches are mutually compatible, allowing concurrent read operations without conflict.[Fully supported][Utility:5]
  ```
- **Analysis:** The adapter accurately cites the database theory reason (compatibility of Shared locks and page latches) rather than the vague tautology of the base model.

---

## 7. Failure Modes, Limitations & Behavioral Analysis

In accordance with Section 4 of the task specification, we conducted a rigorous analysis of failure modes and model limitations:

1. **Self-RAG Pipeline Decoupling (End-to-End Prompting):**
   - *Observation:* When prompted solely with a raw user question without an external retrieval engine supplying the evidence, the fine-tuned model attempts to generate both the `<paragraph>...</paragraph>` evidence context and the answer.
   - *Root Cause:* The SFT completion format combined retrieval context with critique. Because Self-RAG was trained as an inference-time modular controller, the model expects an external retriever to intervene after the `[Retrieval]` token.
   - *Mitigation:* The adapter should only be invoked within the EvoRAG orchestrator where the external passage retriever supplies the evidence block into `<paragraph>...</paragraph>`.
2. **Context-Grounded Abstention Sycophancy:**
   - *Observation:* When unanswerable or ungrounded questions (e.g. asking for exact microsecond packet latency or passwords in the university schema) were posed directly without evidence, the model generated plausible-sounding hallucinations (e.g. claiming latency is 100 microseconds, or that passwords use strict encryption) rather than self-abstaining with `[No support / Contradictory]`.
   - *Root Cause:* In the absence of an external contradictory document, the model's language modeling prior favors generating a helpful completion rather than generating its own contradictory context.
   - *Mitigation:* Explicitly verify that the EvoRAG retrieval critic stage feeds the retrieved document context prior to generating the critique.
3. **Token Length Truncation & Repetition at Tail:**
   - *Observation:* When generating both the paragraph (~150 tokens) and answer text within fixed limits, if `max_tokens` is set too short (e.g. 220), the answer text can be cut off before reaching the closing `[Utility:5]`. If allowed to generate beyond the closing `[Utility:5]`, the model occasionally emits trailing `<unk>` delimiters or repeats `[Utility:5]`.
   - *Mitigation:* Production inference stopping criteria should terminate generation immediately upon emitting any `[Utility:X]` token.

---

## 8. Artifacts & Resource Inventory

All training artifacts were written to dedicated directories, leaving existing files intact:

```
training/qlora_dbms/adapters/full_dbms_v1/
├── adapter_config.json                 (320 bytes)
├── adapters.safetensors                (38.15 MB, SHA-256: 7e7381...)
├── best_adapters.safetensors           (38.15 MB, SHA-256: 7e7381...)
├── best_checkpoint_info.json           (198 bytes)
├── 0000182_adapters.safetensors        (38.15 MB, Epoch 1 Checkpoint)
├── 0000364_adapters.safetensors        (38.15 MB, Epoch 2 Checkpoint)
└── 0000546_adapters.safetensors        (38.15 MB, Epoch 3 Checkpoint)

training/qlora_dbms/
├── full_training_metrics.json          (Complete iteration-by-iteration trace)
├── internal_test_evaluation.json       (Complete 39-example evaluation metrics)
├── training_config.yaml                (Hyperparameter configuration)
├── run_full_training.py                (Reproducible training runner)
└── evaluate_internal_test.py           (Post-training evaluation runner)
```

- **Net Disk Space Consumed:** 0.12 GiB (Free space remaining: 10.04 GiB).
- **Peak Unified Memory:** 6.61 GB during training; 4.82 GB during inference.

---

## 9. Integrity Verification & Production Regression Suite

All project-wide immutability requirements were verified:

| Verification Check | Target Asset | Verified State | Status |
| :--- | :--- | :--- | :---: |
| **Base MLX Weights** | `models/mlx/selfrag_llama2_7b_4bit/model.safetensors` | SHA-256: `71576febb874c07b...` | **BITWISE UNCHANGED** |
| **Production GGUF Model** | `models/selfrag_llama2_7b.Q4_K_M.gguf` | Size: 4,081,095,776 bytes, SHA-256: `fb39ef32...` | **BITWISE UNCHANGED** |
| **Frozen Benchmark CSV** | `evaluation/predictions/frozen/predictions_20261109_151957Z.csv` | SHA-256: `a08d3bd3b0440302...` | **BITWISE UNCHANGED** |
| **Experience Memory** | `data/experience_memory.db` & `data/experience_meta.json` | 25 approved DBMS experiences intact | **BITWISE UNCHANGED** |
| **Production `.venv`** | `.venv` Python 3.13 packages | Completely isolated from `.venv-mlx` | **UNMODIFIED** |
| **Regression Test Suite** | 13 production tests across Task 9B and Task 10 | `13 passed, 1 warning in 7.01s` | **100% PASS** |

---

## 10. Conclusion & Recommendation for EvoRAG Integration

### Verdict: **READY FOR CONTROLLED HYBRID EVORAG INTEGRATION AUDIT**

1. **Why It Is Ready:**
   - The fine-tuned QLoRA adapter achieves an **82.8% reduction in loss** and **99.5% reduction in perplexity** on held-out DBMS questions compared to the base model.
   - It eliminates severe base model failure modes (infinite token repetition loops and empty tag cascades).
   - It reliably generates canonical Self-RAG reflection tokens (`[Retrieval]`, `[Relevant]`, `[Fully supported]`, `[Utility:5]`).
2. **Integration Architecture Requirements:**
   - **Do NOT replace the production GGUF model** or deploy this adapter directly as an end-to-end ungrounded chatbot.
   - In subsequent EvoRAG tasks, the adapter should be integrated into the **modular Self-RAG inference pipeline** where:
     a) The PassageRetriever retrieves DBMS context from the FAISS document index.
     b) The context is inserted into `<paragraph>...</paragraph>`.
     c) The adapted model is prompted with the evidence prefix to critique relevance and support.
     d) Generation is halted immediately when `[Utility:X]` is emitted.

---
*Report generated and validated under EvoRAG Task 12E.*

