"""
Benchmark Self-RAG 7B against Ollama models, in the style of Table 2 of the paper.

Two sources:
  --source pdf     (default) your PDF testbench: testbench/questions.jsonl +
                   data/testbench/*.pdf, retrieved with YOUR FAISS index.
  --source paper   the official Self-RAG eval files (popqa/tqa/pubhealth/arc)
                   with pre-retrieved Wikipedia passages, if you can obtain them.

  python -m benchmark.run_benchmark run \
      --systems selfrag ollama:llama2:7b ollama:llama3:8b ollama:gemma3:12b \
      --retrieval both
  python -m benchmark.run_benchmark report
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Make both the addon itself and the repo root (selfrag/, retriever/, ...)
# importable no matter which directory the command is run from.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "benchmark_addon"))

from benchmark.metrics import score
from benchmark.tasks import (PDF_LABELS, PDF_TYPES, TASKS, attach_retrieval,
                             load_pdf_testbench, load_task, retrieval_hit)

LABELS = {**{t: lab for t, (_, lab) in TASKS.items()}, **PDF_LABELS}
COL_ORDER = ["popqa", "triviaqa", "pubhealth", "arc_c"] + PDF_TYPES


def build_systems(args) -> list:
    from benchmark.systems import OllamaSystem, SelfRAGSystem
    systems = []
    for spec in args.systems:
        if spec == "selfrag":
            from selfrag.generator import SelfRAGGenerator
            gen = SelfRAGGenerator(args.selfrag_model)
            systems.append(SelfRAGSystem(gen, k=args.k, mode=args.selfrag_mode))
        elif spec.startswith("ollama:"):
            model = spec.split(":", 1)[1]                 # keeps the tag, e.g. llama3:8b
            for ret in {"on": [True], "off": [False], "both": [False, True]}[args.retrieval]:
                systems.append(OllamaSystem(model, ret, k=args.k, template=args.template,
                                            host=args.ollama_host))
        else:
            raise SystemExit(f"unknown system spec: {spec}")
    return systems


def load_items(args, out_dir: Path) -> dict[str, list[dict]]:
    if args.source == "paper":
        return {t: load_task(t, args.data_dir, args.limit, args.seed) for t in args.tasks}
    by_type = load_pdf_testbench(args.questions, args.types, args.limit, args.seed)
    attach_retrieval(by_type, args.index, args.k, str(out_dir / f"retrieval_k{args.k}.json"),
                     refresh=args.refresh_retrieval)
    return by_type


def run(args):
    out_dir = Path(args.out_dir or ("benchmark_results_pdf" if args.source == "pdf"
                                    else "benchmark_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    by_task = load_items(args, out_dir)
    systems = build_systems(args)
    for task, items in by_task.items():
        for sys_ in systems:
            fp = out_dir / f"{sys_.name.replace(':', '_').replace('/', '_')}__{task}.jsonl"
            done = {json.loads(l)["id"] for l in fp.open()} if fp.exists() else set()
            todo = [it for it in items if it["id"] not in done]
            print(f"[{task}] {sys_.name}: {len(done)} done, {len(todo)} to go")
            with fp.open("a") as f:
                for n, it in enumerate(todo, 1):
                    t0 = time.perf_counter()
                    try:
                        pred = sys_.answer(it)
                    except Exception as e:                # keep long runs alive
                        print(f"  ! item {it['id']}: {e}")
                        continue
                    row = {"id": it["id"], "task": task, "system": sys_.name,
                           "question": it.get("question"), "prediction": pred,
                           "golds": it["golds"], "retrieval_hit": retrieval_hit(it),
                           "latency": round(time.perf_counter() - t0, 3),
                           **score(task, pred, it["golds"])}
                    f.write(json.dumps(row) + "\n"); f.flush()
                    if n % 25 == 0:
                        print(f"  {n}/{len(todo)}")
    report(args)


def report(args):
    out_dir = Path(args.out_dir or ("benchmark_results_pdf" if getattr(args, "source", "pdf") == "pdf"
                                    else "benchmark_results"))
    table: dict[str, dict] = {}
    hits: dict[str, dict] = {}
    for fp in sorted(out_dir.glob("*__*.jsonl")):
        rows = [json.loads(l) for l in fp.open()]
        if not rows:
            continue
        s, t = rows[0]["system"], rows[0]["task"]
        table.setdefault(s, {})[t] = (100 * sum(r["acc"] for r in rows) / len(rows), len(rows))
        for r in rows:
            if r.get("retrieval_hit") is not None:
                hits.setdefault(t, {})[r["id"]] = r["retrieval_hit"]
    cols = [t for t in COL_ORDER if any(t in res for res in table.values())]
    w = 11
    head = f"{'System':<26}" + "".join(f"{LABELS[t]:>{w}}" for t in cols) + f"{'Avg':>{w}}"
    print("\n" + head + "\n" + "-" * len(head))
    for s, res in table.items():
        vals = [res[t][0] for t in cols if t in res]
        cells = "".join(f"{res[t][0]:>{w}.1f}" if t in res else f"{'--':>{w}}" for t in cols)
        avg = f"{sum(vals) / len(vals):>{w}.1f}" if vals else f"{'--':>{w}}"
        print(f"{s:<26}{cells}{avg}")
    ns = "".join(f"{(next((res[t][1] for res in table.values() if t in res), '')):>{w}}" for t in cols)
    print(f"{'(n questions)':<26}{ns}")
    if hits:
        print("\nRetrieval hit-rate (gold evidence phrase inside the k retrieved passages):")
        for t, d in hits.items():
            print(f"  {LABELS[t]:<8} {100 * sum(d.values()) / len(d):5.1f}%  (n={len(d)})")
    print("\nacc: substring/label match (paper metric); Unans = % that abstain instead of "
          "hallucinating (phrase heuristic: spot check the `prediction` fields). "
          "Rows without '+ret' saw no passages.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default=None)
        p.add_argument("--source", choices=["pdf", "paper"], default="pdf")
    r = sub.choices["run"]
    r.add_argument("--systems", nargs="+", required=True,
                   help="selfrag | ollama:<model[:tag]> ...")
    r.add_argument("--retrieval", choices=["on", "off", "both"], default="on",
                   help="for Ollama systems: with/without the retrieved passages")
    r.add_argument("--template", choices=["chat", "alpaca"], default="chat")
    r.add_argument("--selfrag-mode", choices=["adaptive", "always"], default="adaptive")
    r.add_argument("--selfrag-model",
                   default=str(_ROOT / "models/selfrag_llama2_7b.Q4_K_M.gguf"))
    r.add_argument("--ollama-host", default="http://localhost:11434")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--limit", type=int, default=None, help="random subsample per task/type")
    r.add_argument("--seed", type=int, default=0)
    # pdf source
    r.add_argument("--questions", default="testbench/questions.jsonl")
    r.add_argument("--index", default="index")
    r.add_argument("--types", nargs="+", default=None, choices=PDF_TYPES)
    r.add_argument("--refresh-retrieval", action="store_true")
    # paper source
    r.add_argument("--data-dir", default="eval_data")
    r.add_argument("--tasks", nargs="+", default=["popqa", "pubhealth", "arc_c"], choices=list(TASKS))
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
