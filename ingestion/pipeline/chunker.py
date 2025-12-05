"""Transcript chunking with speaker awareness."""

import re
from typing import Dict, List

from .config import PipelineConfig


def sanitize_id(doc_id: str) -> str:
    """Sanitize document ID for Vector Search."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
    return re.sub(r'_+', '_', sanitized).strip('_')


class TranscriptChunker:
    """Create speaker-aware chunks from earnings call transcripts."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.speaker_pattern = re.compile(r'\*\*([^*]+)\*\*[:\s-]*', re.MULTILINE)
    
    def chunk_transcript(self, transcript: Dict) -> List[Dict]:
        """Chunk a single transcript."""
        ticker = transcript['ticker']
        company = transcript['company']
        year = transcript['year']
        quarter = transcript['quarter']
        content = transcript['content']
        
        # Split by speaker
        sections = []
        current_speaker = "Introduction"
        current_text: List[str] = []
        
        for line in content.split('\n'):
            match = self.speaker_pattern.match(line.strip())
            if match:
                if current_text:
                    sections.append({
                        'speaker': current_speaker,
                        'text': '\n'.join(current_text).strip()
                    })
                current_speaker = match.group(1).strip()
                remaining = self.speaker_pattern.sub('', line).strip()
                current_text = [remaining] if remaining else []
            else:
                current_text.append(line)
        
        if current_text:
            sections.append({
                'speaker': current_speaker,
                'text': '\n'.join(current_text).strip()
            })
        
        # Create chunks
        chunks = []
        chunk_num = 0
        
        for section in sections:
            text = section['text']
            if not text.strip():
                continue
            
            speaker = section['speaker']
            
            if len(text) <= self.config.max_chunk_size:
                chunk_num += 1
                chunks.append({
                    'id': sanitize_id(f"{ticker}_Q{quarter}_{year}_chunk_{chunk_num}"),
                    'ticker': ticker,
                    'company': company,
                    'year': year,
                    'quarter': f"Q{quarter}",
                    'speaker': speaker,
                    'chunk_number': chunk_num,
                    'content': text,
                })
            else:
                # Split large sections
                step = self.config.max_chunk_size - self.config.chunk_overlap
                for i in range(0, len(text), step):
                    chunk_text = text[i:i + self.config.max_chunk_size]
                    if chunk_text.strip():
                        chunk_num += 1
                        chunks.append({
                            'id': sanitize_id(f"{ticker}_Q{quarter}_{year}_chunk_{chunk_num}"),
                            'ticker': ticker,
                            'company': company,
                            'year': year,
                            'quarter': f"Q{quarter}",
                            'speaker': speaker,
                            'chunk_number': chunk_num,
                            'content': chunk_text,
                        })
        
        return chunks
    
    def chunk_all(self, transcripts: List[Dict]) -> List[Dict]:
        """Chunk all transcripts."""
        print(f"\n✂️  Chunking {len(transcripts)} transcripts...")
        
        all_chunks = []
        for transcript in transcripts:
            chunks = self.chunk_transcript(transcript)
            all_chunks.extend(chunks)
        
        print(f"   ✓ Created {len(all_chunks):,} chunks")
        return all_chunks
