"""Hybrid Vector Search Ingestion Pipeline.

Modular, crash-resistant pipeline for ingesting S&P 500 earnings calls
into Vertex AI Vector Search with hybrid embeddings.

Modules:
- config: Shared configuration
- checkpoint: Atomic checkpoint management with corruption recovery
- embeddings: Dense embedding generation with per-batch persistence
- bm25: Sparse BM25 encoding
- chunker: Transcript chunking
- downloader: Transcript downloading
- export: JSONL export and GCS upload
"""

from .config import PipelineConfig
from .checkpoint import CheckpointManager
from .embeddings import EmbeddingGenerator
from .bm25 import BM25SparseEncoder
from .chunker import TranscriptChunker
from .downloader import TranscriptDownloader
from .export import export_to_jsonl, upload_to_gcs, save_chunk_content

__all__ = [
    'PipelineConfig',
    'CheckpointManager', 
    'EmbeddingGenerator',
    'BM25SparseEncoder',
    'TranscriptChunker',
    'TranscriptDownloader',
    'export_to_jsonl',
    'upload_to_gcs',
    'save_chunk_content',
]
