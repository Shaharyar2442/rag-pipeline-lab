

import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Loading YAML config file and return as dict.

    
    Returns:
        Dict with keys: dataset_path, embedding_model, generation_model,
                        api_provider, api_key, top_k, index_path.
    """
    if config_path:
        path = Path(config_path)
    else:
        # Trying config.yaml first, then fall back to config.example.yaml
        path = Path.cwd() / "config" / "config.yaml"
        if not path.exists():
            path = Path.cwd() / "config" / "config.example.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setting the default values for optional fields
    config.setdefault("top_k", 5)
    config.setdefault("api_provider", "groq")
    config.setdefault("index_path", "index/crag_index")
    config.setdefault("embedding_model", "all-MiniLM-L6-v2")

    return config
