#!/usr/bin/env python
"""Hybrid Vector Search Ingestion Pipeline - Main Orchestrator.

Crash-resistant pipeline with stage-based resumption:

Stages:
1. download  - Download transcripts (cached)
2. chunk     - Create chunks from transcripts
3. embed     - Generate dense embeddings (with per-batch checkpoints)
4. sparse    - Generate BM25 sparse embeddings
5. export    - Export to JSONL
6. upload    - Upload to GCS

Usage:
    # ALWAYS activate venv first (from project root)
    & .\\.venv\\Scripts\\Activate.ps1
    cd ingestion
    
    # Full pipeline
    python run_pipeline.py
    
    # Resume from last checkpoint
    python run_pipeline.py --resume
    
    # Skip download (use cached transcripts)
    python run_pipeline.py --skip-download
    
    # Start from a specific stage
    python run_pipeline.py --start-stage embed
    
    # Test with limited companies
    python run_pipeline.py --limit 10
    
    # Clear all checkpoints and start fresh
    python run_pipeline.py --clear
    
    # Generate and upload chunk content only (skip embeddings)
    python run_pipeline.py --skip-download --chunks-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    PipelineConfig,
    CheckpointManager,
    EmbeddingGenerator,
    BM25SparseEncoder,
    TranscriptChunker,
    TranscriptDownloader,
    save_chunk_content,
)
from pipeline.bm25 import add_sparse_embeddings
from pipeline.downloader import get_sp500_tickers
from pipeline.export import export_to_jsonl, upload_to_gcs


STAGES = ['download', 'chunk', 'embed', 'sparse', 'export', 'upload']


def save_chunks_cache(chunks: list, config: PipelineConfig) -> None:
    """Save chunks to cache file for resumption."""
    cache_file = config.checkpoint_dir / "chunks_cache.json"
    # Only save metadata, not content (too large)
    chunks_meta = [
        {k: v for k, v in c.items() if k != 'content' and k != 'embedding' and k != 'sparse_embedding'}
        for c in chunks
    ]
    with open(cache_file, 'w') as f:
        json.dump(chunks_meta, f)
    print(f"   💾 Saved {len(chunks)} chunks metadata")


def load_chunks_cache(config: PipelineConfig) -> list:
    """Load chunks metadata from cache."""
    cache_file = config.checkpoint_dir / "chunks_cache.json"
    if not cache_file.exists():
        return []
    with open(cache_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Hybrid Vector Search Ingestion Pipeline')
    parser.add_argument('--year', type=int, default=2025, help='Year to fetch')
    parser.add_argument('--quarters', type=str, default='2,3', help='Quarters (comma-separated)')
    parser.add_argument('--limit', type=int, help='Limit companies (for testing)')
    parser.add_argument('--skip-download', action='store_true', help='Skip transcript download')
    parser.add_argument('--skip-upload', action='store_true', help='Skip GCS upload')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--start-stage', type=str, choices=STAGES, help='Start from specific stage')
    parser.add_argument('--clear', action='store_true', help='Clear all checkpoints')
    parser.add_argument('--chunks-only', action='store_true',
                        help='Only generate and upload chunk content (skip embeddings)')
    args = parser.parse_args()
    
    quarters = [int(q.strip()) for q in args.quarters.split(',')]
    config = PipelineConfig()
    
    print("=" * 70)
    print("  Hybrid Vector Search Ingestion Pipeline (v2 - Crash Resistant)")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Project: {config.project_id}")
    print(f"   Bucket: {config.gcs_bucket}")
    print(f"   Embedding Model: {config.embedding_model}")
    print(f"   Embedding Dimensions: {config.embedding_dimensions}")
    print(f"   Target: Q{'/Q'.join(map(str, quarters))} {args.year}")
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        config.checkpoint_dir,
        config.embeddings_dir,
        f"embeddings_{args.year}",
    )
    
    # Clear checkpoints if requested
    if args.clear:
        print("\n🧹 Clearing all checkpoints...")
        checkpoint_mgr.clear()
        chunks_cache = config.checkpoint_dir / "chunks_cache.json"
        if chunks_cache.exists():
            chunks_cache.unlink()
        print("   ✓ Checkpoints cleared")
        if not args.start_stage:
            print("\n💡 Use --start-stage to begin from a specific stage, or run without flags for full pipeline")
            return
    
    # Show checkpoint stats
    stats = checkpoint_mgr.get_stats()
    if stats['embeddings_count'] > 0:
        print(f"\n📊 Existing checkpoint:")
        print(f"   Embeddings cached: {stats['embeddings_count']:,}")
        print(f"   Cache size: {stats['embeddings_size_mb']:.1f} MB")
    
    # Check API key
    if not config.ninja_api_key and not args.skip_download:
        print("\n❌ NINJA_API_KEY not set. Use --skip-download if transcripts exist.")
        sys.exit(1)
    
    # Determine starting stage
    start_stage = args.start_stage or ('download' if not args.skip_download else 'chunk')
    if args.resume:
        processed_ids, stage = checkpoint_mgr.load_state()
        if stage == 'embeddings_complete':
            start_stage = 'sparse'
        elif processed_ids:
            start_stage = 'embed'
    
    start_idx = STAGES.index(start_stage)
    print(f"\n🚀 Starting from stage: {start_stage}")
    
    # =========================================================================
    # Stage 1: Download / Load Transcripts
    # =========================================================================
    transcripts = []
    tickers = get_sp500_tickers()
    
    if not tickers:
        print("\n❌ Could not fetch S&P 500 tickers")
        sys.exit(1)
    
    if args.limit:
        tickers = tickers[:args.limit]
        print(f"\n⚠️  Limited to {args.limit} companies")
    
    downloader = TranscriptDownloader(config)
    
    if start_idx <= STAGES.index('download'):
        if args.skip_download:
            transcripts = downloader.load_from_cache(tickers, args.year, quarters)
        else:
            transcripts = downloader.download_all(tickers, args.year, quarters)
    else:
        transcripts = downloader.load_from_cache(tickers, args.year, quarters)
    
    if not transcripts:
        print("\n❌ No transcripts found")
        sys.exit(1)
    
    # =========================================================================
    # Stage 2: Chunk Transcripts
    # =========================================================================
    chunks = []
    chunker = TranscriptChunker(config)
    
    if start_idx <= STAGES.index('chunk'):
        chunks = chunker.chunk_all(transcripts)
        save_chunks_cache(chunks, config)
    else:
        # Reload chunks with content from transcripts
        chunks_meta = load_chunks_cache(config)
        if not chunks_meta:
            # Need to rechunk
            chunks = chunker.chunk_all(transcripts)
            save_chunks_cache(chunks, config)
        else:
            # Rebuild full chunks from transcripts
            chunks = chunker.chunk_all(transcripts)
    
    if not chunks:
        print("\n❌ No chunks created")
        sys.exit(1)
    
    # =========================================================================
    # Chunks-only mode: Skip embeddings, just save chunk content
    # =========================================================================
    if args.chunks_only:
        print("\n⏭️  Skipping embeddings (--chunks-only mode)")
        
        # Save chunk content
        chunks_file = config.hybrid_output_dir / "chunk_content.json"
        save_chunk_content(chunks, chunks_file)
        
        # Upload to GCS
        if not args.skip_upload:
            upload_to_gcs(chunks_file, config, "embeddings/chunk_content.json")
        
        print("\n" + "=" * 70)
        print("✅ Chunks-Only Pipeline Complete!")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   Transcripts: {len(transcripts):,}")
        print(f"   Chunks: {len(chunks):,}")
        print(f"   Output: {chunks_file}")
        return
    
    # =========================================================================
    # Stage 3: Generate Dense Embeddings
    # =========================================================================
    if start_idx <= STAGES.index('embed'):
        processed_ids, _ = checkpoint_mgr.load_state()
        
        embedding_gen = EmbeddingGenerator(config, checkpoint_mgr)
        _processed_ids = embedding_gen.generate(chunks, processed_ids)  # noqa: F841
    
    # Load embeddings from cache
    embedding_gen = EmbeddingGenerator(config, checkpoint_mgr)
    chunks = embedding_gen.apply_embeddings(chunks)
    
    # =========================================================================
    # Stage 4: Generate Sparse Embeddings
    # =========================================================================
    bm25_path = config.hybrid_output_dir / "bm25_encoder.json"
    
    if start_idx <= STAGES.index('sparse'):
        # Fit BM25 on corpus
        corpus = [c['content'] for c in chunks]
        bm25 = BM25SparseEncoder()
        bm25.fit(corpus)
        bm25.save(str(bm25_path))
        
        # Add sparse embeddings
        chunks = add_sparse_embeddings(chunks, bm25)
    else:
        # Load existing BM25
        if bm25_path.exists():
            bm25 = BM25SparseEncoder.from_file(str(bm25_path))
            chunks = add_sparse_embeddings(chunks, bm25)
        else:
            corpus = [c['content'] for c in chunks]
            bm25 = BM25SparseEncoder()
            bm25.fit(corpus)
            bm25.save(str(bm25_path))
            chunks = add_sparse_embeddings(chunks, bm25)
    
    # =========================================================================
    # Stage 5: Export to JSONL
    # =========================================================================
    if start_idx <= STAGES.index('export'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = config.hybrid_output_dir / f"hybrid_embeddings_{timestamp}.jsonl"
        export_to_jsonl(chunks, output_file)
    else:
        # Find latest export
        exports = list(config.hybrid_output_dir.glob("hybrid_embeddings_*.jsonl"))
        if exports:
            output_file = max(exports, key=lambda p: p.stat().st_mtime)
            print(f"\n📂 Using existing export: {output_file.name}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = config.hybrid_output_dir / f"hybrid_embeddings_{timestamp}.jsonl"
            export_to_jsonl(chunks, output_file)
    
    # Save chunk content (for retrieval display)
    chunks_file = config.hybrid_output_dir / "chunk_content.json"
    save_chunk_content(chunks, chunks_file)
    
    # =========================================================================
    # Stage 6: Upload to GCS
    # =========================================================================
    if start_idx <= STAGES.index('upload') and not args.skip_upload:
        # Upload embeddings JSONL
        upload_to_gcs(output_file, config, "embeddings/hybrid_embeddings.jsonl")
        # Upload BM25 encoder
        upload_to_gcs(bm25_path, config, "embeddings/bm25_encoder.json")
        # Upload chunk content
        upload_to_gcs(chunks_file, config, "embeddings/chunk_content.json")
    else:
        print("\n⏭️  Skipping GCS upload")
    
    # =========================================================================
    # Cleanup (only on full success)
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ Pipeline Complete!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Transcripts: {len(transcripts):,}")
    print(f"   Chunks: {len(chunks):,}")
    print(f"   Dense dimensions: {config.embedding_dimensions}")
    print(f"   BM25 vocabulary: {bm25.vocab_size:,} terms")
    print(f"   Output: {output_file}")
    
    # Ask before clearing checkpoints
    print(f"\n💡 Checkpoint files preserved in {config.embeddings_dir}")
    print(f"   Run with --clear to remove them for a fresh start")


if __name__ == "__main__":
    main()
