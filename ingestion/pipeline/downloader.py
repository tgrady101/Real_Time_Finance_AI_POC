"""Transcript downloading from API Ninjas."""

import time
from typing import Dict, List, Optional, Tuple

import requests

from .config import PipelineConfig


def get_sp500_tickers() -> List[Tuple[str, str]]:
    """Get list of S&P 500 tickers from Wikipedia."""
    print("\n📋 Fetching S&P 500 constituent list...")
    
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        storage_options = {'User-Agent': 'Mozilla/5.0'}
        tables = pd.read_html(url, storage_options=storage_options)
        
        df = None
        for table in tables:
            if 450 < len(table) < 520:
                df = table
                break
        
        if df is None:
            raise ValueError("Could not find S&P 500 table")
        
        tickers = []
        for _, row in df.iterrows():
            ticker, name = None, None
            for col in df.columns:
                col_lower = str(col).lower()
                if 'symbol' in col_lower or 'ticker' in col_lower:
                    ticker = str(row[col]).strip().upper()
                elif 'security' in col_lower or 'company' in col_lower:
                    name = str(row[col]).strip()
            
            if ticker is None:
                ticker = str(row.iloc[0]).strip().upper()
            if name is None:
                name = str(row.iloc[1]).strip()
            
            if ticker and name and ticker != 'NAN' and len(ticker) <= 5:
                tickers.append((ticker, name))
        
        print(f"   ✓ Found {len(tickers)} S&P 500 companies")
        return tickers
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return []


class TranscriptDownloader:
    """Download earnings call transcripts from API Ninjas."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def _fetch_transcript(
        self,
        ticker: str,
        year: int,
        quarter: int,
    ) -> Optional[str]:
        """Fetch a single transcript from API."""
        if not self.config.ninja_api_key:
            return None
        
        url = f"{self.config.ninja_base_url}/earningstranscript"
        headers = {'X-Api-Key': self.config.ninja_api_key}
        params = {'ticker': ticker, 'year': str(year), 'quarter': str(quarter)}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and 'transcript' in data:
                transcript_split = data.get('transcript_split', [])
                
                if transcript_split:
                    content = f"# {ticker} Earnings Call - Q{quarter} {year}\n\n"
                    for seg in transcript_split:
                        speaker = seg.get('speaker', 'Unknown')
                        role = seg.get('role', '')
                        company = seg.get('company', '')
                        text = seg.get('text', '')
                        
                        if role and company:
                            content += f"**{speaker}** - {role} ({company}):\n{text}\n\n"
                        elif role:
                            content += f"**{speaker}** - {role}:\n{text}\n\n"
                        else:
                            content += f"**{speaker}**:\n{text}\n\n"
                    return content
                elif data.get('transcript'):
                    return f"# {ticker} Earnings Call - Q{quarter} {year}\n\n{data['transcript']}"
            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 402:
                print(f"      ⚠️  Payment required - upgrade API Ninjas plan")
            return None
        except Exception:
            return None
    
    def download_all(
        self,
        tickers: List[Tuple[str, str]],
        year: int,
        quarters: List[int],
    ) -> List[Dict]:
        """Download all transcripts."""
        print(f"\n📥 Downloading transcripts for {len(tickers)} companies...")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        all_transcripts = []
        
        for idx, (ticker, company) in enumerate(tickers, 1):
            print(f"\n[{idx}/{len(tickers)}] {ticker} - {company}")
            
            for quarter in quarters:
                filename = f"{ticker}_Q{quarter}_{year}.txt"
                file_path = self.config.output_dir / filename
                
                if file_path.exists():
                    print(f"      ✓ Cached: Q{quarter} {year}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    content = self._fetch_transcript(ticker, year, quarter)
                    if content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"      ✓ Downloaded: Q{quarter} {year}")
                    else:
                        print(f"      - Not found: Q{quarter} {year}")
                        continue
                    time.sleep(self.config.api_delay)
                
                all_transcripts.append({
                    'ticker': ticker,
                    'company': company,
                    'year': year,
                    'quarter': quarter,
                    'content': content,
                })
        
        print(f"\n   ✓ Total transcripts: {len(all_transcripts)}")
        return all_transcripts
    
    def load_from_cache(
        self,
        tickers: List[Tuple[str, str]],
        year: int,
        quarters: List[int],
    ) -> List[Dict]:
        """Load transcripts from cache without downloading."""
        print(f"\n📂 Loading transcripts from cache...")
        
        transcripts = []
        for ticker, company in tickers:
            for quarter in quarters:
                file_path = self.config.output_dir / f"{ticker}_Q{quarter}_{year}.txt"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        transcripts.append({
                            'ticker': ticker,
                            'company': company,
                            'year': year,
                            'quarter': quarter,
                            'content': f.read(),
                        })
        
        print(f"   ✓ Loaded {len(transcripts)} cached transcripts")
        return transcripts
