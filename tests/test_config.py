"""Tests for ProjectConfig loading across all environments."""

import pytest

from book_recommender.config import ProjectConfig


SAMPLE_CONFIG = {
    "dev": {
        "catalog": "dev",
        "schema_name": "book_recommender",
        "volume": "summaries",
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "warehouse_id": "abc123",
        "vector_search_endpoint": "vs-endpoint-dev",
    },
    "stg": {
        "catalog": "stg",
        "schema_name": "book_recommender",
        "volume": "summaries",
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "warehouse_id": "def456",
        "vector_search_endpoint": "vs-endpoint-stg",
    },
    "prd": {
        "catalog": "prd",
        "schema_name": "book_recommender",
        "volume": "summaries",
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "warehouse_id": "ghi789",
        "vector_search_endpoint": "vs-endpoint-prd",
    },
}


@pytest.fixture
def config_file(tmp_path):
    """Write a temporary project_config.yml and return its path."""
    import yaml

    config_path = tmp_path / "project_config.yml"
    config_path.write_text(yaml.dump(SAMPLE_CONFIG))
    return config_path


@pytest.mark.parametrize("env", ["dev", "stg", "prd"])
def test_config_loads_for_all_environments(config_file, env):
    config = ProjectConfig.from_yaml(config_file, env=env)
    assert config.catalog == env
    assert config.schema_name == "book_recommender"
    assert config.volume == "summaries"


def test_full_schema_name(config_file):
    config = ProjectConfig.from_yaml(config_file, env="dev")
    assert config.full_schema_name == "dev.book_recommender"


def test_full_volume_path(config_file):
    config = ProjectConfig.from_yaml(config_file, env="dev")
    assert config.full_volume_path == "/Volumes/dev/book_recommender/summaries"


def test_default_model_params(config_file):
    config = ProjectConfig.from_yaml(config_file, env="dev")
    assert config.temperature == 0.7
    assert config.max_tokens == 2000
    assert config.top_p == 0.95


def test_invalid_environment_raises(config_file):
    with pytest.raises(ValueError, match="Environment 'prod' not found"):
        ProjectConfig.from_yaml(config_file, env="prod")
