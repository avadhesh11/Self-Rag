"""
Scoring for the "short-form" and "closed-set" columns of Table 2 in the
Self-RAG paper (PopQA, TriviaQA, PubHealth, ARC-Challenge).

`match` is copied from the official repo (retrieval_lm/metrics.py): the
prediction counts as correct if ANY gold answer appears in it as a plain
substring (paper Sec. 4.1: "whether gold answers are included in the model
generations instead of strictly requiring exact matching").
"""
import re
import string


def match(prediction: str, golds: list[str]) -> int:
    """Official metric: case-sensitive substring containment."""
    return int(any(g in prediction for g in golds))


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def match_normalized(prediction: str, golds: list[str]) -> int:
    """Lenient variant (lowercase, strip punctuation/articles). Reported
    alongside the official number so chat models aren't penalised for casing."""
    p = _normalize(prediction)
    return int(any(_normalize(g) and _normalize(g) in p for g in golds))


# ---- closed-set post-processing (paper App. B.2: models format answers
# ---- in "slightly different ways", so predictions are string-matched back
# ---- onto the label set) -------------------------------------------------

def extract_choice(prediction: str) -> str | None:
    """ARC-C: first standalone capital A-D, e.g. 'B', '(B)', 'Answer: B'."""
    m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", prediction)
    return m.group(1) if m else None


def extract_truth(prediction: str) -> str | None:
    """PubHealth: 'true' / 'false' (lowercased, per the paper)."""
    p = prediction.lower().replace("supports", "true").replace("refutes", "false")
    m = re.search(r"\b(true|false)\b", p)
    return m.group(1) if m else None


_ABSTAIN = ("not mentioned", "not specified", "not provided", "not stated",
            "does not say", "doesn't say", "does not mention", "doesn't mention",
            "does not contain", "do not contain", "doesn't contain", "no information",
            "not contained", "cannot be determined", "can't be determined",
            "cannot determine", "can't determine", "cannot find", "can't find",
            "not enough information", "i don't know", "i do not know", "i'm not sure",
            "i am not sure", "i'm sorry", "i am sorry", "unable to", "not able to",
            "not available", "not in the", "no mention", "isn't mentioned",
            "is not mentioned", "isn't specified", "isn't provided", "don't have",
            "do not have", "unclear", "not clear", "not familiar", "cannot answer",
            "can't answer", "cannot provide", "can't provide", "not mention",
            "not specify", "not state", "not include", "no data")


def abstains(prediction: str) -> bool:
    """Heuristic: did the model decline / say the info isn't available?
    Phrase-based, so spot check a few outputs (see README of the testbench)."""
    p = prediction.lower().replace("\u2019", "'")
    return any(ph in p for ph in _ABSTAIN)


def score(task: str, prediction: str, golds: list[str]) -> dict:
    """Return per-item scores. `acc` is the number that goes in the table."""
    if task == "unanswerable":                  # correct = does NOT hallucinate
        ok = int(abstains(prediction))
        return {"acc": ok, "acc_norm": ok, "parsed": None}
    task = {"mc": "arc_c", "tf": "pubhealth"}.get(task, task)
    if task == "arc_c":
        gold = golds[0]
        if gold in "1234":                      # some dumps use 1-4 labels
            gold = "ABCD"[int(gold) - 1]
        pred = extract_choice(prediction)
        ok = int(pred == gold)
        return {"acc": ok, "acc_norm": ok, "parsed": pred}
    if task == "pubhealth":
        pred = extract_truth(prediction)
        ok = int(pred == golds[0].lower())
        return {"acc": ok, "acc_norm": ok, "parsed": pred}
    return {"acc": match(prediction, golds),
            "acc_norm": match_normalized(prediction, golds),
            "parsed": None}
