# PDF testbench

Full setup + results summary and recreate commands: see `testbench.md` at
the repo root. This file covers the retrieval specifics.

55 questions over 4 fictional PDFs (+ the Self-RAG paper) in `data/testbench/`.
Regenerate everything (PDFs + questions.jsonl; works from any cwd) with:

    python benchmark_addon/testbench/build_testbench.py

    # 1. index the testbench PDFs + the paper (small chunks -> retrieval must discriminate)
    python index/build_index.py \
        --pdf benchmark_addon/data/testbench/*.pdf data/2310.11511v1.pdf \
        --out benchmark_addon/index --target-words 40 --overlap-words 0

    # 2. run (resumable) and print the table  -- from benchmark_addon/ so the
    #    defaults (questions.jsonl, index/, results dir) resolve correctly
    cd benchmark_addon
    python -m benchmark.run_benchmark run \
        --systems selfrag ollama:llama2:7b ollama:llama3:8b ollama:gemma3:12b \
        --retrieval both --k 5

    # 3. table again later / inspect raw outputs
    python -m benchmark.run_benchmark report
    # per-question results: benchmark_results_pdf/<system>__<type>.jsonl

## Retrieval hit-rate (verified 2026-10-30)

With the all-MiniLM-L6-v2 embedder, k=5, 40-word chunks: **96% (43/45)** of
gold evidence phrases land in the top-5 passages — 100% on short/multi/tf/mc.
The two misses are both `paper`-type questions against the 350-chunk paper
corpus, where MiniLM's ranking falls short of the paper's Contriever-MSMARCO:

- `pp01` ("1,399" PopQA long-tail queries): the evidence chunk ranks 7th.
- `pp04` ("Contriever" retriever): the sentence reads "Contriever-MS MARCO"
  after PDF cleaning, so the token "retriever" never appears in the chunk and
  MiniLM ranks it far down.

Known fixes if it ever matters: raise `--k`, or swap the embedder for
Contriever-MSMARCO (the paper's retriever) when building the index.
