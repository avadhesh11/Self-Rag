# EvoRAG Benchmark Evaluation & Ablation Study

## 1. Frozen Benchmark Overview

To rigorously evaluate EvoRAG, the benchmark was frozen under run ID `20261109_151957Z`:
- **Held-Out Questions**: 32 educational DBMS and SQL questions (`EvoRAG_Blind_Test_Questions/held_out_test_questions_only.jsonl`).
- **Answer Key**: Official double-verified reference answers (`EvoRAG_Answer_Key_For_Frozen_Run/answer_key_32_for_frozen_run.jsonl`).
- **Unanswerable Test Cases**: Exactly 2 intentionally unanswerable questions (`DBMS-EVAL-0149` and `DBMS-EVAL-0150`) to test model abstention.
- **Frozen Baselines**: Baseline predictions for Normal RAG and GGUF Self-RAG are frozen in `evaluation/predictions/frozen/predictions_20261109_151957Z.csv` (SHA-256: `a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7`).

---

## 2. Five-System Ablation Study Matrix (Task 12G Results)

All five systems were evaluated against the exact 32 held-out questions using the official scoring engine (`evaluation/score_predictions.py`):

| System Configuration | Token F1 | Token Precision | Token Recall | Citation F1 | Citation Prec | Citation Rec | Mean Latency | Median Latency | P95 Latency | Memory Hits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Normal RAG** *(Frozen baseline)* | **0.3270** | 0.3011 | **0.4155** | 0.6445 | 0.5639 | **0.8667** | **1,664.2 ms** | **1,600.9 ms** | **2,299.8 ms** | 0 / 32 |
| **System B: GGUF Self-RAG** *(Frozen baseline)* | 0.2209 | 0.2294 | 0.2939 | 0.5267 | 0.4500 | 0.7333 | 13,572.9 ms | 11,949.5 ms | 24,949.2 ms | 0 / 32 |
| **System C: GGUF EvoRAG (Warm)** *(Task 10)* | 0.2365 | 0.2404 | 0.3182 | **0.6556** | **0.7000** | 0.6333 | 15,087.3 ms | 14,165.4 ms | 25,860.8 ms | 5 / 32 |
| **System D: MLX Self-RAG (No Mem)** *(Ablation)* | 0.2528 | 0.2974 | 0.2420 | 0.6222 | 0.6667 | 0.6000 | 10,442.8 ms | 9,980.1 ms | 19,856.3 ms | 0 / 32 |
| **System E: MLX EvoRAG (With Mem)** *(Full)* | **0.2721** | **0.3265** | 0.2539 | 0.6222 | 0.6667 | 0.6000 | **11,062.4 ms** | **10,444.8 ms** | **19,876.7 ms** | 5 / 32 |

---

## 3. Key Findings & Statistical Significance

### 1. Token Precision Leadership
- **System E (Fine-Tuned MLX EvoRAG)** achieved the highest Token Precision (**0.3265**) among all five evaluated systems.
- This represents a **+42.3% relative improvement** over the original GGUF Self-RAG (0.2294) and outperforms Normal RAG (0.3011).

### 2. Experience Memory Value-Add (System E vs System D)
- Running 1,000 paired bootstrap resamples reveals a statistically significant benefit from Experience Memory:
  - **Mean $\Delta = +0.0192$ Token F1**
  - **95% Bootstrap Confidence Interval**: `[+0.0043, +0.0391]` ($p < 0.05$).
- On the 5 questions routed via Path B (Experience-Assisted RAG), Token F1 reached **0.4252** compared to **0.2221** for System D on the identical questions.

### 3. Full System Improvement (System E vs System B)
- Comparing fine-tuned MLX EvoRAG to the original production GGUF Self-RAG:
  - **Mean $\Delta = +0.0512$ Token F1** (+23.2% relative gain)
  - **95% Bootstrap Confidence Interval**: `[+0.0045, +0.0993]` ($p < 0.05$).

### 4. Honest Assessment of Normal RAG
- **Normal RAG (System A)** maintains higher overall Token F1 (0.3270 vs 0.2721) and Token Recall (0.4155 vs 0.2539) due to Gemini's large-model prose generation and comprehensive answers.
- However, Normal RAG lacks adaptive retrieval control (it cannot decide to skip retrieval for parametric questions), requires external cloud connectivity, and has lower precision than System E.

### 5. Latency and Operational Efficiency
- **Model Load Time**: MLX loads in **0.65s** vs **10.53s** for GGUF (>16x faster).
- **Inference Latency**: System E is **26.7% faster** than GGUF EvoRAG (11.06s vs 15.09s average latency).
- **Zero Test Contamination**: Path A was triggered 0 times on the 32 held-out questions, confirming zero leakage between training experiences and the frozen evaluation set.

---

## 4. Reproducing Benchmark Evaluation

To execute the Task 12G frozen benchmark evaluation:

```bash
# Activate the MLX virtual environment
source .venv-mlx/bin/activate

# Run the 5-system frozen benchmark evaluation
python evaluation/run_task12g_frozen_benchmark.py
```

Generated evaluation artifacts:
- [`training/qlora_dbms/TASK12G_FINAL_EVALUATION_REPORT.md`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/TASK12G_FINAL_EVALUATION_REPORT.md)
- [`training/qlora_dbms/task12g_frozen_benchmark_results.json`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/task12g_frozen_benchmark_results.json)
- [`training/qlora_dbms/task12g_frozen_benchmark_predictions.csv`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/task12g_frozen_benchmark_predictions.csv)
- [`training/qlora_dbms/task12g_per_question_scoring.csv`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/task12g_per_question_scoring.csv)

