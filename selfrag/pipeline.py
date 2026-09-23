"""
The actual Self-RAG algorithm (simplified to single-segment generation,
suitable for short-answer queries against a small demo document).

Full paper algorithm, recursive over multiple output segments:
    for each segment:
        decide whether to retrieve (or continue using current evidence)
        if retrieving: generate one candidate continuation per passage
        score every candidate on ISREL / ISSUP / ISUSE
        keep the best-scoring candidate, move to next segment

What we implement here: exactly one segment, i.e. one retrieval decision
and one round of candidate generation/scoring, which is sufficient to
demonstrate every piece of the control loop (adaptive retrieval, passage
filtering, support checking, utility-weighted selection) without the
added complexity of multi-segment long-form generation. This scoping
choice is called out explicitly rather than silently -- extending to
multi-segment generation is a natural next feature to add on top.
"""
from dataclasses import dataclass, field

from .query_normalizer import normalize_query
from .reflection_tokens import (
    RETRIEVE_NO,
    RETRIEVE_YES,
    find_first,
    parse_utility_score,
    strip_reflection_tokens,
    critique_score,
    ISREL_TOKENS,
    ISSUP_TOKENS,
)


_REFUSAL_PHRASES = (
    "i'm sorry",
    "i am sorry",
    "not sure what you are asking",
    "not sure what you are referring to",
    "not familiar with the term",
    "provide more context or clarify",
    "don't have any information",
    "do not have any information",
)


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def _all_candidates_refuse(candidates: list["Candidate"]) -> bool:
    return bool(candidates) and all(_is_refusal(c.answer_text) for c in candidates)


@dataclass
class Candidate:
    passage: dict
    raw_text: str
    isrel: str | None
    issup: str | None
    isuse: int | None
    score: float
    answer_text: str = field(init=False)

    def __post_init__(self):
        self.answer_text = strip_reflection_tokens(self.raw_text)


@dataclass
class SelfRAGResult:
    query: str
    retrieved: bool
    candidates: list[Candidate]
    selected: Candidate | None
    final_answer: str
    decision: str | None = None  # [Retrieval] / [No Retrieval], or None
    normalized_query: str | None = None


def _decide_retrieve(generator, query: str) -> tuple[bool, str | None]:
    """Ask the model, with no passage attached, whether it wants to
    retrieve. Greedy decoding (temperature=0) makes this a single
    deterministic forward pass -- the first reflection token it emits is
    its retrieval decision.

    Returns (should_retrieve, decision_token). Three observed behaviors of
    the checkpoint (verified live, see README):

    * emits [Retrieval]  -> retrieve
    * emits [No Retrieval] -> don't retrieve
    * emits no retrieval token at all but answers directly (typically
      ending in [Utility:N], the model scoring its own answer) -> don't
      retrieve either. Overriding the model here would be worse, not
      safer: the Self-RAG contract is that the model owns this decision,
      and forcing retrieval for e.g. arithmetic just wastes a round on
      passages the model will then correctly judge [No support /
      Contradictory].
    """
    raw = generator.generate(query, paragraph=None)
    decision = find_first(raw, [RETRIEVE_YES, RETRIEVE_NO])
    return (decision == RETRIEVE_YES, decision)


def _score_candidate(passage: dict, raw_text: str) -> Candidate:
    isrel = find_first(raw_text, ISREL_TOKENS)
    issup = find_first(raw_text, ISSUP_TOKENS)
    isuse = parse_utility_score(raw_text)
    score = critique_score(isrel, issup, isuse)
    return Candidate(
        passage=passage, raw_text=raw_text,
        isrel=isrel, issup=issup, isuse=isuse, score=score,
    )


def self_rag_answer(query: str, generator, retriever, k: int = 5) -> SelfRAGResult:
    # Normalize the query: decouple search terms from the generation instruction
    search_query, instruction = normalize_query(query)

    should_retrieve, decision = _decide_retrieve(generator, instruction)
    if not should_retrieve:
        raw = generator.generate(instruction, paragraph=None)
        answer = strip_reflection_tokens(raw)
        return SelfRAGResult(query=query, retrieved=False, candidates=[],
                              selected=None, final_answer=answer, decision=decision,
                              normalized_query=instruction)

    passages = retriever.search(search_query, k=k)
    raw_outputs = generator.generate_batch(instruction, [p["text"] for p in passages])

    candidates = [_score_candidate(p, raw) for p, raw in zip(passages, raw_outputs)]

    # Refusal fallback: if every candidate generated a refusal/clarification response
    # ("I'm sorry, but I'm not sure..."), re-prompt with an explicit extraction directive.
    if _all_candidates_refuse(candidates) and passages:
        fallback_inst = f"Summarize the key facts about {search_query} stated in the passage:"
        fallback_outputs = generator.generate_batch(fallback_inst, [p["text"] for p in passages])
        fallback_candidates = [_score_candidate(p, raw) for p, raw in zip(passages, fallback_outputs)]
        if any(c.issup != "[No support / Contradictory]" for c in fallback_candidates):
            candidates = fallback_candidates

    # Discard candidates whose passage the model itself judged irrelevant,
    # UNLESS every candidate was judged irrelevant (then fall back to the
    # single best-scoring one so the demo still returns something).
    relevant = [c for c in candidates if c.isrel != "[Irrelevant]"]
    pool = relevant if relevant else candidates

    best = max(pool, key=lambda c: c.score)
    return SelfRAGResult(query=query, retrieved=True, candidates=candidates,
                          selected=best, final_answer=best.answer_text,
                          decision=decision, normalized_query=instruction)
