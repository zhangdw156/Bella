from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None
    bfcl_registry_name: str
    bfcl_project_root: str | None


def load_settings(env_path: Path | None = None) -> Settings:
    """
    Load configuration from .env and environment variables.
    Priority:
    - Explicit env_path if provided
    - Default to Bella project root /.env
    """
    if env_path is None:
        # Bella project root is the directory containing pyproject.toml
        project_root = Path(__file__).resolve().parents[2]
        env_path = project_root / ".env"

    load_dotenv(dotenv_path=env_path, override=False)

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    bfcl_registry_name = os.getenv("BFCL_REGISTRY_NAME", "").strip() or "bella-mvp"
    bfcl_project_root = os.getenv("BFCL_PROJECT_ROOT", "").strip() or None

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in .env or environment.")
    if not openai_model:
        raise RuntimeError("OPENAI_MODEL must be set in .env or environment.")

    # Propagate critical env vars for BFCL / OpenAI handler
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if openai_base_url:
        # Only set when non-empty, otherwise let SDK use default
        os.environ["OPENAI_BASE_URL"] = openai_base_url

    if bfcl_project_root:
        os.environ["BFCL_PROJECT_ROOT"] = bfcl_project_root

    return Settings(
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        bfcl_registry_name=bfcl_registry_name,
        bfcl_project_root=bfcl_project_root,
    )

