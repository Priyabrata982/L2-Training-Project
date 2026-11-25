"""Configuration settings for the Loan Application Risk Assistant."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    use_fallback_llm: bool = False

    confidence_threshold: float = 0.65
    max_dti_ratio: float = 0.50
    max_resubmit_attempts: int = 3

    knowledge_base_dir: Path = PROJECT_ROOT / "knowledge_base"
    database_path: Path = PROJECT_ROOT / "data" / "loan_reviews.db"
    checkpoint_db_path: Path = PROJECT_ROOT / "data" / "langgraph_checkpoints.db"


settings = Settings()
