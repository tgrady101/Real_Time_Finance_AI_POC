"""S&P 500 Earnings Call Ingestion Script.

Downloads earnings call transcripts for all S&P 500 companies (last 2 quarters)
and imports them into Vertex AI Discovery Engine Data Store for RAG.

Uses API Ninjas for transcript data (paid tier required: $39/month).

Usage:
    # Set API key first
    $env:NINJA_API_KEY = "your-api-key"
    
    # Run ingestion
    cd ingestion
    python sp500_earnings_ingestion.py
    
    # Or specify quarters explicitly
    python sp500_earnings_ingestion.py --year 2025 --quarters 2,3
"""

import os
import sys
import json
import re
import time
import requests
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899")
DATA_STORE_ID = os.getenv("DATA_STORE_ID", "earnings-call-datastore")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{GCP_PROJECT_ID}-finance-data-bucket")
DATA_STORE_LOCATION = "global"

# Local directories
OUTPUT_DIR = Path(__file__).parent / "downloaded_earnings_calls"
CHUNKED_DIR = Path(__file__).parent / "chunked_earnings_calls"

# Chunking configuration (optimized for RAG)
MAX_CHUNK_SIZE = 6000  # Characters per chunk
CHUNK_OVERLAP = 300    # Character overlap between chunks for context continuity

# API Configuration
API_NINJAS_KEY = os.getenv("NINJA_API_KEY", "")
API_NINJAS_BASE_URL = "https://api.api-ninjas.com/v1"

# Rate limiting
API_DELAY_SECONDS = 0.5  # Delay between API requests


def get_sp500_tickers() -> List[Tuple[str, str]]:
    """Get list of all S&P 500 tickers and company names from Wikipedia.
    
    Returns:
        List of tuples: (ticker, company_name)
    """
    print("\n📋 Fetching S&P 500 constituent list from Wikipedia...")
    
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        tables = pd.read_html(url, storage_options=storage_options)
        
        # Find the constituents table (usually first, ~500 rows)
        df = None
        for table in tables:
            if len(table) > 450 and len(table) < 520:
                df = table
                break
        
        if df is None:
            raise ValueError("Could not find S&P 500 constituents table")
        
        # Extract ticker and company name
        tickers = []
        for _, row in df.iterrows():
            # Try common column names
            ticker = None
            name = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'symbol' in col_lower or 'ticker' in col_lower:
                    ticker = str(row[col]).strip().upper()
                elif 'security' in col_lower or 'company' in col_lower:
                    name = str(row[col]).strip()
            
            # Fallback to positional
            if ticker is None:
                ticker = str(row.iloc[0]).strip().upper()
            if name is None:
                name = str(row.iloc[1]).strip()
            
            if ticker and name and ticker != 'NAN' and len(ticker) <= 5:
                tickers.append((ticker, name))
        
        print(f"   ✓ Found {len(tickers)} S&P 500 companies")
        return tickers
        
    except Exception as e:
        print(f"   ✗ Error fetching S&P 500 list: {e}")
        return []


def sanitize_document_id(doc_id: str) -> str:
    """Sanitize document ID to match Vertex AI pattern: [a-zA-Z0-9-_]*"""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')


def fetch_earnings_transcript(ticker: str, year: int, quarter: int) -> Optional[str]:
    """Fetch earnings call transcript from API Ninjas.
    
    Args:
        ticker: Stock ticker symbol
        year: Year (e.g., 2025)
        quarter: Quarter (1-4)
        
    Returns:
        Transcript text if successful, None otherwise
    """
    if not API_NINJAS_KEY:
        return None
    
    url = f"{API_NINJAS_BASE_URL}/earningstranscript"
    headers = {'X-Api-Key': API_NINJAS_KEY}
    params = {
        'ticker': ticker,
        'year': str(year),
        'quarter': str(quarter)
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, dict) and 'transcript' in data:
            transcript_text = data.get('transcript', '')
            transcript_split = data.get('transcript_split', [])
            
            # Prefer speaker-split version if available
            if transcript_split and len(transcript_split) > 0:
                full_content = f"# {ticker} Earnings Call - Q{quarter} {year}\n\n"
                for segment in transcript_split:
                    speaker = segment.get('speaker', 'Unknown Speaker')
                    role = segment.get('role', '')
                    company = segment.get('company', '')
                    text = segment.get('text', '')
                    
                    if role and company:
                        full_content += f"**{speaker}** - {role} ({company}):\n{text}\n\n"
                    elif role:
                        full_content += f"**{speaker}** - {role}:\n{text}\n\n"
                    else:
                        full_content += f"**{speaker}**:\n{text}\n\n"
                return full_content
            elif transcript_text:
                return f"# {ticker} Earnings Call - Q{quarter} {year}\n\n{transcript_text}"
        
        return None
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 402:
            print(f"      ⚠️  Payment required - upgrade API Ninjas plan")
        elif e.response.status_code == 404:
            pass  # Transcript not found - expected for some companies
        return None
    except Exception:
        return None


def fetch_company_earnings(ticker: str, company_name: str, year: int, quarters: List[int]) -> List[Dict]:
    """Fetch earnings calls for a company for specified quarters.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        year: Year to fetch
        quarters: List of quarters to fetch (e.g., [2, 3])
        
    Returns:
        List of transcript dictionaries with metadata
    """
    transcripts = []
    
    for quarter in quarters:
        # Check if already downloaded
        filename = f"{ticker}_Q{quarter}_{year}.txt"
        file_path = OUTPUT_DIR / filename
        
        if file_path.exists():
            print(f"      ✓ Already downloaded: Q{quarter} {year}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            transcripts.append({
                'ticker': ticker,
                'company': company_name,
                'year': year,
                'quarter': quarter,
                'content': content,
                'file_path': str(file_path)
            })
            continue
        
        # Fetch from API
        content = fetch_earnings_transcript(ticker, year, quarter)
        
        if content:
            # Save locally
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"      ✓ Downloaded: Q{quarter} {year} ({len(content):,} chars)")
            
            transcripts.append({
                'ticker': ticker,
                'company': company_name,
                'year': year,
                'quarter': quarter,
                'content': content,
                'file_path': str(file_path)
            })
        else:
            print(f"      - No transcript: Q{quarter} {year}")
        
        # Rate limiting
        time.sleep(API_DELAY_SECONDS)
    
    return transcripts


def create_chunks(transcript: Dict) -> List[Dict]:
    """Create speaker-aware chunks from a transcript.
    
    Args:
        transcript: Dictionary with transcript data
        
    Returns:
        List of chunk dictionaries
    """
    content = transcript['content']
    ticker = transcript['ticker']
    company = transcript['company']
    year = transcript['year']
    quarter = transcript['quarter']
    
    # Split by speaker sections (marked with **)
    speaker_pattern = re.compile(r'\*\*([^*]+)\*\*[:\s-]*', re.MULTILINE)
    
    sections = []
    current_speaker = "Introduction"
    current_text = []
    
    for line in content.split('\n'):
        match = speaker_pattern.match(line.strip())
        if match:
            if current_text:
                sections.append({
                    'speaker': current_speaker,
                    'text': '\n'.join(current_text).strip()
                })
            current_speaker = match.group(1).strip()
            # Get the text after the speaker name on the same line
            remaining = speaker_pattern.sub('', line).strip()
            current_text = [remaining] if remaining else []
        else:
            current_text.append(line)
    
    # Add final section
    if current_text:
        sections.append({
            'speaker': current_speaker,
            'text': '\n'.join(current_text).strip()
        })
    
    # Create chunks from sections
    chunks = []
    chunk_num = 0
    
    for section in sections:
        speaker = section['speaker']
        text = section['text']
        
        if not text.strip():
            continue
        
        # If section is small enough, keep as single chunk
        if len(text) <= MAX_CHUNK_SIZE:
            chunk_num += 1
            chunks.append({
                'id': sanitize_document_id(f"{ticker}_Q{quarter}_{year}_chunk_{chunk_num}"),
                'ticker': ticker,
                'company': company,
                'year': year,
                'quarter': f"Q{quarter}",
                'speaker': speaker,
                'chunk_number': chunk_num,
                'content': text,
                'summary': f"{ticker} Q{quarter} {year} - {speaker}: {text[:200]}..."
            })
        else:
            # Split long sections with overlap
            for i in range(0, len(text), MAX_CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = text[i:i + MAX_CHUNK_SIZE]
                if chunk_text.strip():
                    chunk_num += 1
                    chunks.append({
                        'id': sanitize_document_id(f"{ticker}_Q{quarter}_{year}_chunk_{chunk_num}"),
                        'ticker': ticker,
                        'company': company,
                        'year': year,
                        'quarter': f"Q{quarter}",
                        'speaker': speaker,
                        'chunk_number': chunk_num,
                        'content': chunk_text,
                        'summary': f"{ticker} Q{quarter} {year} - {speaker}: {chunk_text[:200]}..."
                    })
    
    return chunks


def save_chunks_to_file(all_chunks: List[Dict]) -> str:
    """Save chunks to JSON file for later import.
    
    Args:
        all_chunks: List of chunk dictionaries
        
    Returns:
        Path to saved file
    """
    CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chunk_file = CHUNKED_DIR / f"sp500_earnings_chunks_{timestamp}.json"
    
    with open(chunk_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\n💾 Saved {len(all_chunks)} chunks to {chunk_file}")
    return str(chunk_file)


def upload_to_gcs(chunk_file: str) -> str:
    """Upload chunks file to GCS bucket.
    
    Args:
        chunk_file: Path to local chunk file
        
    Returns:
        GCS URI
    """
    from google.cloud import storage
    
    print(f"\n☁️  Uploading to GCS bucket: {GCS_BUCKET_NAME}")
    
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)
    
    blob_name = f"earnings_calls/{Path(chunk_file).name}"
    blob = bucket.blob(blob_name)
    
    blob.upload_from_filename(chunk_file)
    
    gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    print(f"   ✓ Uploaded to {gcs_uri}")
    
    return gcs_uri


def import_to_vertex_ai(all_chunks: List[Dict], start_batch: int = 0) -> None:
    """Import chunks to Vertex AI Discovery Engine using inline source.
    
    Args:
        all_chunks: List of chunk dictionaries
        start_batch: Batch number to start from (for resuming failed imports)
    """
    from google.cloud import discoveryengine_v1 as discoveryengine
    
    print(f"\n🔄 Importing {len(all_chunks)} chunks to Vertex AI Data Store...")
    print(f"   Project: {GCP_PROJECT_ID}")
    print(f"   Data Store: {DATA_STORE_ID}")
    
    client = discoveryengine.DocumentServiceClient()
    parent = f"projects/{GCP_PROJECT_ID}/locations/{DATA_STORE_LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}/branches/default_branch"
    
    # Convert chunks to Document objects
    documents = []
    for chunk in all_chunks:
        doc = discoveryengine.Document(
            id=chunk['id'],
            struct_data={
                'ticker': chunk['ticker'],
                'company': chunk['company'],
                'year': chunk['year'],
                'quarter': chunk['quarter'],
                'speaker': chunk['speaker'],
                'chunk_number': chunk['chunk_number'],
                'document_type': 'earnings_call_transcript',
                'summary': chunk['summary']
            },
            content=discoveryengine.Document.Content(
                mime_type="text/plain",
                raw_bytes=chunk['content'].encode('utf-8')
            )
        )
        documents.append(doc)
    
    # Batch import (max 100 documents per request)
    # Rate limit: 100 requests per minute = ~1.7 requests per second
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size
    requests_this_minute = 0
    minute_start = time.time()
    max_requests_per_minute = 90  # Stay under 100 limit
    
    failed_batches = []
    
    for i in range(start_batch * batch_size, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        # Rate limiting: if we've hit the limit, wait for the minute to reset
        if requests_this_minute >= max_requests_per_minute:
            elapsed = time.time() - minute_start
            if elapsed < 60:
                wait_time = 61 - elapsed
                print(f"   ⏳ Rate limit reached, waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
            requests_this_minute = 0
            minute_start = time.time()
        
        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            inline_source=discoveryengine.ImportDocumentsRequest.InlineSource(
                documents=batch
            ),
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
        )
        
        # Retry with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                operation = client.import_documents(request=request)
                print(f"   ✓ Batch {batch_num}/{total_batches}: {len(batch)} documents submitted")
                print(f"     Operation: {operation.operation.name}")
                requests_this_minute += 1
                break
            except Exception as e:
                if "429" in str(e) or "RATE_LIMIT" in str(e):
                    wait_time = (2 ** attempt) * 30  # 30s, 60s, 120s
                    print(f"   ⚠️ Rate limited on batch {batch_num}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    requests_this_minute = 0
                    minute_start = time.time()
                else:
                    print(f"   ✗ Batch {batch_num}/{total_batches}: {e}")
                    failed_batches.append(batch_num)
                    break
        else:
            print(f"   ✗ Batch {batch_num} failed after {max_retries} retries")
            failed_batches.append(batch_num)
    
    if failed_batches:
        print(f"\n⚠️  {len(failed_batches)} batches failed: {failed_batches[:10]}{'...' if len(failed_batches) > 10 else ''}")
        print(f"   Run with --resume-import to retry failed batches")
    
    print("\n✅ Import submitted! Monitor progress in Google Cloud Console:")
    print(f"   https://console.cloud.google.com/gen-app-builder/data-stores/{DATA_STORE_ID}")


def main():
    """Main function to orchestrate the S&P 500 earnings call ingestion."""
    parser = argparse.ArgumentParser(description='Ingest S&P 500 earnings calls to Vertex AI')
    parser.add_argument('--year', type=int, default=2025, help='Year to fetch (default: 2025)')
    parser.add_argument('--quarters', type=str, default='2,3', help='Quarters to fetch, comma-separated (default: 2,3)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of companies (for testing)')
    parser.add_argument('--skip-import', action='store_true', help='Skip Vertex AI import (just download)')
    args = parser.parse_args()
    
    quarters = [int(q.strip()) for q in args.quarters.split(',')]
    
    print("=" * 70)
    print("  S&P 500 Earnings Call Ingestion")
    print("=" * 70)
    print(f"\n📅 Target: Q{'/Q'.join(map(str, quarters))} {args.year}")
    
    # Check API key
    if not API_NINJAS_KEY:
        print("\n❌ ERROR: NINJA_API_KEY not set")
        print("\nTo get started:")
        print("1. Sign up at https://www.api-ninjas.com/register")
        print("2. Subscribe to Developer plan ($39/month)")
        print("3. Set environment variable:")
        print("   PowerShell: $env:NINJA_API_KEY = 'your-api-key'")
        print("   Or add to .env file: NINJA_API_KEY=your-api-key")
        sys.exit(1)
    
    print(f"✓ API key configured")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    if not tickers:
        print("❌ Failed to fetch S&P 500 list")
        sys.exit(1)
    
    if args.limit:
        tickers = tickers[:args.limit]
        print(f"\n⚠️  Limited to {args.limit} companies for testing")
    
    # Estimate API usage
    max_requests = len(tickers) * len(quarters)
    print(f"\n📊 Estimated API requests: {max_requests} (100K/month limit)")
    
    # Fetch transcripts for each company
    print(f"\n📥 Downloading earnings call transcripts...")
    
    all_transcripts = []
    success_count = 0
    fail_count = 0
    
    for idx, (ticker, company_name) in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] {ticker} - {company_name}")
        
        try:
            transcripts = fetch_company_earnings(ticker, company_name, args.year, quarters)
            if transcripts:
                all_transcripts.extend(transcripts)
                success_count += len(transcripts)
            else:
                fail_count += 1
        except Exception as e:
            print(f"      ✗ Error: {e}")
            fail_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"📊 Download Summary:")
    print(f"   ✓ Transcripts downloaded: {success_count}")
    print(f"   - Companies without transcripts: {fail_count}")
    
    if not all_transcripts:
        print("\n❌ No transcripts downloaded. Check API key and try again.")
        sys.exit(1)
    
    # Chunk transcripts
    print(f"\n✂️  Chunking {len(all_transcripts)} transcripts...")
    all_chunks = []
    for transcript in all_transcripts:
        chunks = create_chunks(transcript)
        all_chunks.extend(chunks)
        print(f"   {transcript['ticker']} Q{transcript['quarter']}: {len(chunks)} chunks")
    
    print(f"\n   Total chunks: {len(all_chunks)}")
    
    # Save chunks to file (function prints confirmation)
    _ = save_chunks_to_file(all_chunks)
    
    # Import to Vertex AI
    if not args.skip_import:
        import_to_vertex_ai(all_chunks)
    else:
        print("\n⏭️  Skipping Vertex AI import (--skip-import flag)")
    
    print("\n" + "=" * 70)
    print("✅ Ingestion Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
