"""
Uniform `System.answer(item) -> str` wrappers so Self-RAG and every Ollama
baseline are driven by identical data, prompts-per-task, and scoring.
"""
from selfrag.pipeline import self_rag_answer, _generate_and_score
from selfrag.reflection_tokens import strip_reflection_tokens

from .ollama_generator import OllamaGenerator

MAX_NEW_TOKENS = {"pubhealth": 50, "arc_c": 50, "tf": 50, "mc": 50}   # others: 100


class PrecomputedRetriever:
    """Duck-types PassageRetriever.search() but returns the passages that ship
    with each benchmark item (Contriever top-k over Wikipedia), so your own
    FAISS/PDF index is not involved in the benchmark."""

    def __init__(self):
        self._passages: list[dict] = []

    def set_item(self, item: dict):
        self._passages = [
            {"text": f"{c['title']}\n{c['text']}", "page": c["title"],
             "source": "wikipedia", "chunk_id": f"{item['id']}:{i}",
             "retrieval_score": 0.0}
            for i, c in enumerate(item["ctxs"])]

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self._passages[:k]


class SelfRAGSystem:
    """Your existing Self-RAG pipeline, fed pre-retrieved passages.
    mode='adaptive': the model decides whether to retrieve (paper default).
    mode='always'  : force retrieval (fair match for always-retrieve baselines)."""

    def __init__(self, generator, k: int = 5, mode: str = "adaptive"):
        self.gen, self.k, self.mode = generator, k, mode
        self.retriever = PrecomputedRetriever()
        self.name = f"selfrag-{mode}"

    def answer(self, item: dict) -> str:
        self.retriever.set_item(item)
        if self.mode == "always":
            passages = self.retriever.search(item["instruction"], self.k)
            _, best = _generate_and_score(item["instruction"], passages, self.gen,
                                          history="", search_query=item["instruction"])
            return strip_reflection_tokens(best.raw_text).strip()
        res = self_rag_answer(item["instruction"], self.gen, self.retriever,
                              k=self.k, max_segments=1, normalize=False)
        return res.final_answer.strip()


class OllamaSystem:
    def __init__(self, model: str, retrieval: bool, k: int = 5,
                 template: str = "chat", host: str = "http://localhost:11434"):
        self.gen = OllamaGenerator(model, host=host, template=template)
        self.retrieval, self.k = retrieval, k
        self.name = f"{model}{'+ret' if retrieval else ''}"

    def answer(self, item: dict) -> str:
        ctxs = item["ctxs"][: self.k] if self.retrieval else None
        return self.gen.generate(item["instruction"], ctxs,
                                 max_new_tokens=MAX_NEW_TOKENS.get(item["task"], 100))
