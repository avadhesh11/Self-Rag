"""
Reflection-token vocabulary used by selfrag/selfrag_llama2_7b, and small
helpers to parse them out of generated text / logprobs.

These exact bracketed strings are what the checkpoint was fine-tuned to
produce (see the paper's Appendix tables and the official model card's
usage example). Getting these strings exactly right matters: the model
was trained with these literal token strings, so any deviation means the
logprob you're reading off no longer corresponds to what the model was
taught to signal.
"""

RETRIEVE_YES = "[Retrieval]"
RETRIEVE_NO = "[No Retrieval]"
RETRIEVE_CONTINUE = "[Continue to Use Evidence]"
RETRIEVE_TOKENS = [RETRIEVE_YES, RETRIEVE_NO, RETRIEVE_CONTINUE]

ISREL_RELEVANT = "[Relevant]"
ISREL_IRRELEVANT = "[Irrelevant]"
ISREL_TOKENS = [ISREL_RELEVANT, ISREL_IRRELEVANT]

ISSUP_FULL = "[Fully supported]"
ISSUP_PARTIAL = "[Partially supported]"
ISSUP_NONE = "[No support / Contradictory]"
ISSUP_TOKENS = [ISSUP_FULL, ISSUP_PARTIAL, ISSUP_NONE]

ISUSE_TOKENS = [f"[Utility:{i}]" for i in range(1, 6)]

# Numeric weights used ONLY to combine critique signals into a single score
# for candidate selection in this demo. The paper's actual segment-level
# scoring is a weighted log-probability combination over the full
# reflection-token sequence (see Section 3.3 / Appendix), which is more
# involved than this. This is a documented simplification, not a
# reproduction of their exact formula -- good enough to demonstrate the
# control loop, not to reproduce paper benchmark numbers.
ISREL_WEIGHT = {ISREL_RELEVANT: 1.0, ISREL_IRRELEVANT: -1.0}
ISSUP_WEIGHT = {ISSUP_FULL: 1.0, ISSUP_PARTIAL: 0.4, ISSUP_NONE: -1.0}


def strip_reflection_tokens(text: str) -> str:
    """Remove all reflection-token markup, leaving only the natural-language
    answer, for final display to the user."""
    all_tokens = RETRIEVE_TOKENS + ISREL_TOKENS + ISSUP_TOKENS + ISUSE_TOKENS
    cleaned = text
    for tok in all_tokens:
        cleaned = cleaned.replace(tok, "")
    # collapse leftover whitespace from removed tokens
    return " ".join(cleaned.split())


def find_first(text: str, candidates: list[str]) -> str | None:
    """Return whichever candidate token appears first in text, or None."""
    positions = [(text.find(tok), tok) for tok in candidates if tok in text]
    if not positions:
        return None
    positions.sort(key=lambda p: p[0])
    return positions[0][1]


def parse_utility_score(text: str) -> int | None:
    tok = find_first(text, ISUSE_TOKENS)
    if tok is None:
        return None
    return int(tok.split(":")[1].rstrip("]"))


def critique_score(isrel: str | None, issup: str | None, isuse: int | None) -> float:
    """Combine the three critique signals into one scalar for ranking
    candidate continuations. See module docstring re: simplification."""
    score = 0.0
    if isrel is not None:
        score += ISREL_WEIGHT.get(isrel, 0.0)
    if issup is not None:
        score += ISSUP_WEIGHT.get(issup, 0.0)
    if isuse is not None:
        score += (isuse - 3) / 2.0  # center utility 1..5 around 0, range -1..1
    return score
