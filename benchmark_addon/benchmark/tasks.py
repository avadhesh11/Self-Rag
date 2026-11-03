"""
Loaders for the evaluation files released with the official Self-RAG repo
(https://drive.google.com/file/d/1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb). Every
file already contains the top-k passages retrieved by Contriever-MSMARCO
from Wikipedia under `ctxs`, so no Wikipedia index is needed here.

Each loader returns items of the form:
    {id, task, instruction, golds, ctxs: [{title, text}, ...]}
`instruction` follows paper Table 5 / official TASK_INST (no instruction for
open-domain QA).
"""
import ast
import json
import random
from pathlib import Path

ARC_INST = ("Given four answer candidates, A, B, C and D, choose the best answer "
            "choice. Please answer with the capitalized alphabet only, without "
            "adding any extra phrase or period.")
PUB_INST = ("Is the following statement correct or not? Say true if it's correct; "
            "otherwise, say false. Don't capitalize or add periods, just say "
            "\"true\" or \"false\".")

# task -> (default file name in eval_data/, label used in the results table)
# NOTE: file names follow the official Drive folder; rename here or pass
# --data-dir if yours differ.
TASKS = {
    "popqa":     ("popqa_longtail_w_gs.jsonl",      "PopQA"),
    "triviaqa":  ("triviaqa_test_w_gs.jsonl",       "TQA"),
    "pubhealth": ("health_claims_processed.jsonl",  "Pub"),
    "arc_c":     ("arc_challenge_processed.jsonl",  "ARC"),
}


def _read(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _golds(row: dict) -> list[str]:
    if "possible_answers" in row:                       # PopQA stores a stringified list
        v = row["possible_answers"]
        return ast.literal_eval(v) if isinstance(v, str) else list(v)
    for key in ("golds", "answers", "answer", "output"):
        if key in row:
            v = row[key]
            return [v] if isinstance(v, str) else list(v)
    if "answerKey" in row:
        return [row["answerKey"]]
    raise KeyError(f"no gold answer field in row keys: {list(row)}")


def _arc_options(row: dict) -> str:
    ch = row["choices"]
    labels = {}
    for lab, txt in zip(ch["label"], ch["text"]):
        lab = {"1": "A", "2": "B", "3": "C", "4": "D"}.get(lab, lab)
        labels[lab] = txt
    labels.setdefault("D", "")
    out = "".join(f"\n{k}: {labels[k]}" for k in "ABCD")
    if "E" in labels:
        out += f"\nE: {labels['E']}"
    return out


def load_task(task: str, data_dir: str, limit: int | None = None,
              seed: int = 0) -> list[dict]:
    fname, _ = TASKS[task]
    rows = _read(Path(data_dir) / fname)
    items = []
    for i, row in enumerate(rows):
        question = row.get("question") or row.get("instruction")
        if task == "arc_c":
            instruction = f"{ARC_INST}\n\n### Input:\n{question}{_arc_options(row)}"
        elif task == "pubhealth":
            instruction = f"{PUB_INST}\n\n### Input:\n{question}"
        else:
            instruction = question                       # open-domain QA: no instruction
        ctxs = row.get("ctxs") or row.get("contexts") or row.get("top_contexts") or []
        items.append({"id": row.get("id", i), "task": task,
                      "instruction": instruction, "golds": _golds(row),
                      "ctxs": [{"title": c.get("title", ""), "text": c["text"]} for c in ctxs]})
    if limit and limit < len(items):                     # reproducible subsample
        items = random.Random(seed).sample(items, limit)
    return items


# ===========================================================================
# PDF testbench (replacement for the unavailable Drive eval files)
# ===========================================================================
PDF_TYPES = ["short", "multi", "tf", "mc", "unanswerable", "general", "paper"]
PDF_LABELS = {"short": "Short", "multi": "Multi", "tf": "T/F", "mc": "MC",
              "unanswerable": "Unans", "general": "General", "paper": "Paper"}


def load_pdf_testbench(path: str = "testbench/questions.jsonl",
                       types: list[str] | None = None,
                       limit: int | None = None, seed: int = 0) -> dict[str, list[dict]]:
    """Return {type: [items]}. `ctxs` is empty until attach_retrieval()."""
    by_type: dict[str, list[dict]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if types and r["type"] not in types:
            continue
        by_type.setdefault(r["type"], []).append({
            "id": r["id"], "task": r["type"], "question": r["question"],
            "instruction": r.get("instruction") or r["question"],
            "golds": r["golds"], "evidence": r.get("evidence", []), "ctxs": []})
    if limit:
        for t, items in by_type.items():
            if limit < len(items):
                by_type[t] = random.Random(seed).sample(items, limit)
    return by_type


def attach_retrieval(by_type: dict[str, list[dict]], index_dir: str, k: int,
                     cache_path: str, refresh: bool = False) -> None:
    """Retrieve top-k passages ONCE per question with your own PassageRetriever
    and cache them, so every system (Self-RAG and every Ollama model) reads
    byte-identical passages. Differences between systems are then purely the
    generator / pipeline, never the retriever."""
    cache_file = Path(cache_path)
    cache = {} if refresh or not cache_file.exists() else json.loads(cache_file.read_text())
    retriever = None
    for items in by_type.values():
        for it in items:
            if it["id"] not in cache:
                if retriever is None:
                    from retriever.retriever import PassageRetriever
                    retriever = PassageRetriever(index_dir)
                cache[it["id"]] = [
                    {"title": f"{p.get('source', '?')} p.{p.get('page', '?')}", "text": p["text"]}
                    for p in retriever.search(it["question"], k=k)]
            it["ctxs"] = cache[it["id"]][:k]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))


def retrieval_hit(item: dict) -> int | None:
    """1 if every evidence phrase appears in a retrieved passage (whitespace /
    case-insensitive). None when the question has no evidence (unanswerable,
    general knowledge)."""
    if not item.get("evidence"):
        return None
    blob = " ".join(" ".join(c["text"].lower().split()) for c in item["ctxs"])
    return int(all(" ".join(e.lower().split()) in blob for e in item["evidence"]))
