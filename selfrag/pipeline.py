"""
Self-RAG inference per the paper (Section 3.1 / Algorithm 1): multi-segment
generation with adaptive, reflection-token-driven document retrieval.

The paper's control loop, one iteration per output segment:
    for each segment:
        predict Retrieve given (x, y<t)
            -> [Retrieval]              fetch NEW documents and answer from them
            -> [Continue to Use Evidence]  reuse the evidence already in context
            -> [No Retrieval]              answer without passages and stop

Each [Retrieval] segment is the paper's tree-decoding step: retrieve k
passages, generate one candidate continuation per passage in parallel,
critique every candidate on ISREL / ISSUP / ISUSE, and keep the best-scoring
one. The model then emits the decision token for the NEXT segment at the
end of its output (the training data stream is "[Retrieval]<paragraph>..
</paragraph>[Relevant]sentence[Fully supported][Continue to Use Evidence]
sentence..."), which is exactly what we parse here -- the same mechanism as
the official repo's run_long_form_static.py, which cuts each prediction at
the next retrieval token.

Segment 1's initial retrieval decision uses the natural greedy probe
(verified live on this checkpoint, see README/MEMORY); subsequent decisions
come from the trailing decision token parsed above.

Documented simplifications vs. the paper (kept explicit, same policy as the
previous single-segment version):
  * Segment-level beam search (Section 3.3) is not implemented: we keep one
    best segment per step, not a beam of B.
  * Continue segments reuse the vetted evidence in context without
    re-attaching it per passage, matching the training-data format. The
    official repo's variant (re-running the per-passage candidate batch)
    is a possible extension.
  * Candidate scoring is the linear ISREL/ISSUP/ISUSE combination, not the
    logprob-weighted sum (see reflection_tokens.py).
"""
from dataclasses import dataclass, field

from .query_normalizer import normalize_query
from .reflection_tokens import (
    RETRIEVE_CONTINUE,
    RETRIEVE_NO,
    RETRIEVE_YES,
    RETRIEVE_TOKENS,
    ISREL_IRRELEVANT,
    ISREL_TOKENS,
    ISSUP_NONE,
    ISSUP_TOKENS,
    find_first,
    parse_utility_score,
    strip_reflection_tokens,
    critique_score,
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
class Segment:
    """One output segment from the Algorithm 1 loop, plus its trace."""
    index: int
    decision: str | None              # retrieval decision that opened this segment
    evidence: list[dict]              # passages this segment generated candidates over
    candidates: list[Candidate]
    selected: Candidate | None
    answer_text: str                  # this segment's stripped contribution
    reused_evidence: bool = False     # True for [Continue to Use Evidence] segments


@dataclass
class SelfRAGResult:
    query: str
    retrieved: bool
    candidates: list[Candidate]       # segment 1's candidates (kept for the
    selected: Candidate | None        # existing trace/metrics consumers)
    final_answer: str
    decision: str | None = None  # [Retrieval] / [No Retrieval], or None
    normalized_query: str | None = None
    segments: list[Segment] = field(default_factory=list)


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


def _generate_and_score(instruction: str, passages: list[dict], generator,
                        history: str, search_query: str) -> tuple[list[Candidate], Candidate]:
    """One [Retrieval] segment of the paper's algorithm: generate a candidate
    continuation per passage in parallel, critique each on ISREL/ISSUP/ISUSE,
    apply the refusal fallback and relevance filter, and return
    (all candidates, selected best)."""
    raw_outputs = generator.generate_batch(
        instruction, [p["text"] for p in passages], history=history)
    candidates = [_score_candidate(p, raw) for p, raw in zip(passages, raw_outputs)]

    # Refusal fallback: if every candidate generated a refusal/clarification response
    # ("I'm sorry, but I'm not sure..."), re-prompt with an explicit extraction directive.
    if _all_candidates_refuse(candidates) and passages:
        fallback_inst = f"Summarize the key facts about {search_query} stated in the passage:"
        fallback_outputs = generator.generate_batch(
            fallback_inst, [p["text"] for p in passages], history=history)
        fallback_candidates = [_score_candidate(p, raw) for p, raw in zip(passages, fallback_outputs)]
        if any(c.issup != ISSUP_NONE for c in fallback_candidates):
            candidates = fallback_candidates

    # Discard candidates whose passage the model itself judged irrelevant,
    # UNLESS every candidate was judged irrelevant (then fall back to the
    # single best-scoring one so the demo still returns something).
    relevant = [c for c in candidates if c.isrel != ISREL_IRRELEVANT]
    pool = relevant if relevant else candidates

    best = max(pool, key=lambda c: c.score)
    return candidates, best


def _split_next_decision(raw: str) -> tuple[str, str | None, str]:
    """Split `raw` at its FIRST retrieval decision token:
    (head, decision_token, tail). The checkpoint emits decision tokens in
    order as it wants to change behavior ([Continue to Use Evidence] to
    extend the answer on current evidence, [Retrieval] to fetch new
    documents, [No Retrieval] to finish); processing them left to right
    preserves the model's full decision sequence, and mirrors the official
    repo cutting each prediction at its retrieval token. The tail is
    re-parsed by the caller, so one output can carry several decisions
    (e.g. "[Continue to Use Evidence]sentence[No Retrieval]")."""
    pos, token = None, None
    for tok in RETRIEVE_TOKENS:
        p = raw.find(tok)
        if p != -1 and (pos is None or p < pos):
            pos, token = p, tok
    if token is None:
        return raw, None, ""
    return raw[:pos], token, raw[pos + len(token):]


def _retrieve_new(retriever, search_query: str, k: int,
                  used_ids: set) -> list[dict]:
    """Fetch the next batch of passages for a [Retrieval] segment, skipping
    chunks already used in earlier segments so each retrieval round actually
    fetches NEW documents (and the trace shows which ones)."""
    found = retriever.search(search_query, k=k + len(used_ids))
    return [p for p in found if p.get("chunk_id", p["text"]) not in used_ids][:k]


def self_rag_answer(query: str, generator, retriever, k: int = 5,
                    max_segments: int = 3) -> SelfRAGResult:
    # Normalize the query: decouple search terms from the generation instruction
    search_query, instruction = normalize_query(query)

    # --- Segment 1: the model's own retrieval decision ---
    should_retrieve, decision = _decide_retrieve(generator, instruction)
    if not should_retrieve:
        raw = generator.generate(instruction, paragraph=None)
        answer = strip_reflection_tokens(raw)
        return SelfRAGResult(query=query, retrieved=False, candidates=[],
                             selected=None, final_answer=answer, decision=decision,
                             normalized_query=instruction)

    # --- Segment 1: retrieve + parallel candidates + critique + select ---
    passages = retriever.search(search_query, k=k)
    candidates, best = _generate_and_score(
        instruction, passages, generator, history="", search_query=search_query)

    segments: list[Segment] = []
    answer_stream: list[str] = []  # stripped text per segment, joined at the end
    used_ids = {p.get("chunk_id", p["text"]) for p in passages}
    evidence = passages

    def record(index, dec, ev, cands, sel, text, reused=False):
        segments.append(Segment(index=index, decision=dec, evidence=ev,
                                candidates=cands, selected=sel,
                                answer_text=text, reused_evidence=reused))

    # `ctx` is the model's view of everything after "### Response:" so far
    # (its own outputs + the [Retrieval]<paragraph> markers we attached),
    # i.e. the stream that seeds the next segment's prompt. `out` is the
    # latest segment's raw output.
    ctx = f"{RETRIEVE_YES}<paragraph>{best.passage['text']}</paragraph>"
    out = best.raw_text
    record(1, decision, evidence, candidates, best, strip_reflection_tokens(out))
    answer_stream.append(strip_reflection_tokens(out))  # provisional; revised below if a decision follows

    # --- Segments 2..max_segments: the paper's loop over (x, y<t) ---
    seg_idx = 2
    while seg_idx <= max_segments:
        head, next_decision, tail = _split_next_decision(out)
        if next_decision is None or next_decision == RETRIEVE_NO:
            break  # model chose to finish; head + tail are already in answer_stream

        # The decision token closes the previous segment: keep only its head
        # (for [Retrieval] the tail is displaced by the paragraph we attach;
        # for [Continue] the tail becomes its own segment below).
        answer_stream[-1] = strip_reflection_tokens(head)
        segments[-1].answer_text = answer_stream[-1]

        if next_decision == RETRIEVE_YES:
            fresh = _retrieve_new(retriever, search_query, k, used_ids)
            if not fresh:
                break  # nothing new left in the corpus -- stop here
            used_ids |= {p.get("chunk_id", p["text"]) for p in fresh}
            ctx += head + RETRIEVE_YES
            cands, sel = _generate_and_score(
                instruction, fresh, generator, history=ctx, search_query=search_query)
            ctx += f"<paragraph>{sel.passage['text']}</paragraph>"
            out = sel.raw_text
            evidence = fresh
            record(seg_idx, RETRIEVE_YES, evidence, cands, sel,
                   strip_reflection_tokens(out))
        else:  # [Continue to Use Evidence]: reuse evidence already in context
            ctx += head + RETRIEVE_CONTINUE
            if tail.strip():
                out = tail  # the model already continued inline
            else:
                out = generator.generate(instruction, paragraph=None, history=ctx)
            record(seg_idx, RETRIEVE_CONTINUE, evidence, [], None,
                   strip_reflection_tokens(out), reused=True)
        answer_stream.append(strip_reflection_tokens(out))
        seg_idx += 1

    final = " ".join(p for p in answer_stream if p).strip()
    if not final:  # model emitted only reflection tokens -- degrade gracefully
        final = strip_reflection_tokens(best.raw_text)

    return SelfRAGResult(query=query, retrieved=True, candidates=candidates,
                         selected=best, final_answer=final, decision=decision,
                         normalized_query=instruction, segments=segments)
