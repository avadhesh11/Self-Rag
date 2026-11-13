"""EvoRAG: Self-RAG with Hybrid Experience Learning."""
from .config import ExperienceMemoryConfig
from .result import ExperienceCandidate, ValidationResult, EvoRAGResult
from .controller import EvoRAGController, MemoryIndexRebuildError
from .memory.retriever import ExperienceAugmentedRetriever
try:
    from .mlx_generator import MLXSelfRAGGenerator
except ImportError:
    MLXSelfRAGGenerator = None

__all__ = [
    "ExperienceMemoryConfig",
    "ExperienceCandidate",
    "ValidationResult",
    "EvoRAGResult",
    "EvoRAGController",
    "MemoryIndexRebuildError",
    "ExperienceAugmentedRetriever",
    "MLXSelfRAGGenerator",
]
