"""Export utilities for Vector Search."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from google.cloud import storage

from .config import PipelineConfig


def export_to_jsonl(chunks: List[Dict], output_path: Path) -> str:
    """Export to Vector Search JSONL format with hybrid embeddings.
    
    Output format:
    {
        "id": "AAPL_Q3_2025_chunk_1",
        "embedding": [0.1, 0.2, ...],  # Dense (3072-dim)
        "sparse_embedding": {"values": [...], "dimensions": [...]},
        "restricts": [
            {"namespace": "ticker", "allow": ["AAPL"]},
            {"namespace": "company", "allow": ["apple"]},
            {"namespace": "quarter", "allow": ["Q3_2025"]}
        ],
        "numeric_restricts": [{"namespace": "year", "value_int": 2025}],
        "crowding_tag": "AAPL"
    }
    """
    total = len(chunks)
    print(f"\n💾 Exporting {total:,} documents to JSONL...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i, chunk in enumerate(chunks):
            # Progress every 10000
            if (i + 1) % 10000 == 0:
                pct = (i + 1) / total * 100
                print(f"   [{i + 1:,}/{total:,}] ({pct:.1f}%)")
            # Build token restricts for categorical filtering
            restricts = []
            
            if chunk.get('ticker'):
                restricts.append({
                    "namespace": "ticker",
                    "allow": [chunk['ticker'].upper()]
                })
            
            # Add company name for natural language filtering
            if chunk.get('company'):
                company = chunk['company'].strip()
                company_normalized = company.lower().replace(',', '').replace('.', '')
                # Remove common suffixes
                for suffix in [' inc', ' corp', ' corporation', ' company', ' co', ' ltd', ' llc', ' plc']:
                    if company_normalized.endswith(suffix):
                        company_normalized = company_normalized[:-len(suffix)].strip()
                restricts.append({
                    "namespace": "company",
                    "allow": [company_normalized]
                })
            
            # Combine quarter and year
            quarter = chunk.get('quarter', '')
            year = chunk.get('year', '')
            if quarter and year:
                if not str(quarter).startswith('Q'):
                    quarter = f"Q{quarter}"
                quarter_year = f"{str(quarter).upper()}_{year}"
                restricts.append({
                    "namespace": "quarter",
                    "allow": [quarter_year]
                })
            
            # Numeric restricts for range queries
            numeric_restricts = []
            if year:
                try:
                    numeric_restricts.append({
                        "namespace": "year",
                        "value_int": int(year)
                    })
                except (ValueError, TypeError):
                    pass
            
            doc = {
                "id": chunk['id'],
                "embedding": chunk['embedding'],
                "sparse_embedding": chunk['sparse_embedding'],
                "restricts": restricts,
                "crowding_tag": chunk.get('ticker', ''),
            }
            
            if numeric_restricts:
                doc["numeric_restricts"] = numeric_restricts
            
            f.write(json.dumps(doc) + '\n')
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Exported {len(chunks):,} documents ({size_mb:.1f} MB)")
    return str(output_path)


def save_chunk_content(chunks: List[Dict], output_path: Path) -> str:
    """Save chunk content (id → text mapping) for retrieval at query time.
    
    This creates a JSON file mapping chunk IDs to their text content and metadata,
    which is needed to display search results since Vector Search only stores embeddings.
    """
    print(f"\n💾 Saving chunk content...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunk_data = {}
    for chunk in chunks:
        chunk_data[chunk['id']] = {
            'content': chunk['content'],
            'ticker': chunk.get('ticker', ''),
            'company': chunk.get('company', ''),
            'quarter': chunk.get('quarter', ''),
            'year': chunk.get('year', ''),
            'speaker': chunk.get('speaker', ''),
        }
    
    with open(output_path, 'w') as f:
        json.dump(chunk_data, f)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved {len(chunk_data):,} chunks ({size_mb:.1f} MB)")
    return str(output_path)


def upload_to_gcs(local_path: Path, config: PipelineConfig, gcs_path: str | None = None) -> str:
    """Upload a file to GCS bucket.
    
    Args:
        local_path: Path to local file
        config: Pipeline configuration
        gcs_path: Optional custom GCS path (default: embeddings/{filename})
    """
    print(f"\n☁️  Uploading to GCS...")
    
    client = storage.Client(project=config.project_id)
    bucket = client.bucket(config.gcs_bucket)
    
    if gcs_path is None:
        gcs_path = f"embeddings/{local_path.name}"
    
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    
    gcs_uri = f"gs://{config.gcs_bucket}/{gcs_path}"
    print(f"   ✓ Uploaded to {gcs_uri}")
    return gcs_uri
