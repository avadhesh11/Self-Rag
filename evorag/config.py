"""Configuration constants and classes for EvoRAG Experience Memory."""
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_TOP_K = 3
DEFAULT_DB_PATH = "data/experience_memory.db"
DEFAULT_INDEX_DIR = "data/experience_index"


@dataclass
class ExperienceMemoryConfig:
    """Configuration settings for Experience Memory storage, indexing, and retrieval."""
    db_path: str = DEFAULT_DB_PATH
    index_dir: str = DEFAULT_INDEX_DIR
    embed_model_name: str = DEFAULT_EMBED_MODEL
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    top_k: int = DEFAULT_TOP_K

