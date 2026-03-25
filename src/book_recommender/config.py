"""Environment-aware configuration for the Book Recommender Agent."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """All environment-specific settings for the Book Recommender Agent."""

    catalog: str
    schema_name: str
    volume: str
    llm_endpoint: str
    embedding_endpoint: str
    warehouse_id: str
    vector_search_endpoint: str
    vector_search_index: str = "book_recommender_vs_index"
    chunk_size: int = 2000
    chunk_overlap: int = 200
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.95

    @property
    def full_schema_name(self) -> str:
        """Fully-qualified schema: <catalog>.<schema>."""
        return f"{self.catalog}.{self.schema_name}"

    @property
    def full_volume_path(self) -> str:
        """FUSE path to the summaries volume."""
        return f"/Volumes/{self.catalog}/{self.schema_name}/{self.volume}"

    @property
    def full_vs_index_name(self) -> str:
        """Fully-qualified Vector Search index: <catalog>.<schema>.<index>."""
        return f"{self.full_schema_name}.{self.vector_search_index}"

    @classmethod
    def from_yaml(cls, config_path: str | Path, env: str = "dev") -> ProjectConfig:
        """Load config for *env* from *config_path*."""
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if env not in raw:
            raise ValueError(
                f"Environment '{env}' not found in {config_path}. "
                f"Available: {list(raw.keys())}"
            )
        return cls(**raw[env])


def load_config(env: str | None = None) -> ProjectConfig:
    """Load ProjectConfig for the active environment.

    Search order:
    1. Bundled inside the installed package (works on Databricks after wheel install).
    2. Root of the source tree (works in local dev without installing the wheel).
    """
    if env is None:
        env = get_env()

    # 1. Prefer the copy baked into the wheel via importlib.resources
    try:
        from importlib.resources import files  # noqa: PLC0415

        config_path = files("book_recommender").joinpath("project_config.yml")
        with config_path.open() as f:
            raw = yaml.safe_load(f)
        if env not in raw:
            raise ValueError(
                f"Environment '{env}' not found in bundled project_config.yml. "
                f"Available: {list(raw.keys())}"
            )
        return ProjectConfig(**raw[env])
    except (FileNotFoundError, TypeError):
        pass

    # 2. Fallback: walk up from __file__ (local dev, editable install)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "project_config.yml"
        if candidate.exists():
            return ProjectConfig.from_yaml(candidate, env=env)

    raise FileNotFoundError(
        "project_config.yml not found — ensure it is present at the project root "
        "or rebuild the wheel after adding it to src/book_recommender/."
    )


def get_env() -> str:
    """Return the active environment.

    Reads the 'env' Databricks widget when running on a cluster;
    falls back to the ENV environment variable, then 'dev'.
    """
    try:
        from pyspark.sql import SparkSession  # noqa: PLC0415

        spark = SparkSession.getActiveSession()
        if spark is not None:
            return spark.conf.get("env", "dev")
    except Exception:
        pass
    return os.getenv("ENV", "dev")
