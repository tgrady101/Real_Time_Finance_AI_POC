"""Pipeline configuration module."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


@dataclass
class PipelineConfig:
    """Ingestion pipeline configuration."""
    
    # GCP Settings
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899")
    region: str = "us-central1"
    gcs_bucket: str = os.getenv(
        "GCS_BUCKET_NAME",
        f"{os.getenv('GOOGLE_CLOUD_PROJECT', 'project-4b3d3288-7603-4755-899')}-finance-data-bucket"
    )
    
    # API Settings
    ninja_api_key: str = os.getenv("NINJA_API_KEY", "")
    ninja_base_url: str = "https://api.api-ninjas.com/v1"
    api_delay: float = 0.5
    
    # Embedding Settings
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 5
    embedding_requests_per_minute: int = 300
    
    # Chunking Settings
    max_chunk_size: int = 2000
    chunk_overlap: int = 200
    
    # Checkpoint Settings
    checkpoint_save_interval: int = 50  # Save every N batches (250 chunks)
    max_checkpoint_backups: int = 3
    
    # Directories
    base_dir: Path = Path(__file__).parent.parent
    output_dir: Path = base_dir / "downloaded_earnings_calls"
    checkpoint_dir: Path = base_dir / "checkpoints"
    embeddings_dir: Path = base_dir / "embeddings_cache"  # Individual embedding files
    hybrid_output_dir: Path = base_dir / "hybrid_embeddings"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.hybrid_output_dir.mkdir(parents=True, exist_ok=True)
