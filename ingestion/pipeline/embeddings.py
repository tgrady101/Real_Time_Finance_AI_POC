"""Dense embedding generation with per-batch persistence.

Key features:
- Uses Google Gen AI SDK (not deprecated vertexai)
- Saves embeddings immediately after each batch (no memory accumulation)
- Automatic retry with exponential backoff
- Progress tracking with accurate ETAs
"""

import time
from typing import Dict, List, Set

from google import genai
from google.genai import types

from .config import PipelineConfig
from .checkpoint import CheckpointManager


class EmbeddingGenerator:
    """Generate dense embeddings with crash-resistant persistence."""
    
    def __init__(self, config: PipelineConfig, checkpoint_mgr: CheckpointManager):
        self.config = config
        self.checkpoint_mgr = checkpoint_mgr
        self.client = genai.Client(
            vertexai=True,
            project=config.project_id,
            location=config.region,
        )
        
        # Rate limiting state
        self._requests_this_minute = 0
        self._minute_start = time.time()
    
    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        if self._requests_this_minute >= self.config.embedding_requests_per_minute:
            elapsed = time.time() - self._minute_start
            if elapsed < 60:
                wait_time = 61 - elapsed
                print(f"   ⏳ Rate limit reached, waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
            self._requests_this_minute = 0
            self._minute_start = time.time()
    
    def _generate_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Generate embeddings for a batch with retry logic."""
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                response = self.client.models.embed_content(
                    model=self.config.embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type='RETRIEVAL_DOCUMENT',
                        output_dimensionality=self.config.embedding_dimensions,
                    ),
                )
                
                if not response.embeddings:
                    raise RuntimeError("No embeddings returned")
                
                embeddings = []
                for emb in response.embeddings:
                    if emb.values is None:
                        raise RuntimeError("Null embedding values")
                    embeddings.append(list(emb.values))
                
                self._requests_this_minute += 1
                return embeddings
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                    print(f"   ⚠️  Retry {attempt + 1}/{max_retries} in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise
        
        return []  # Should never reach here
    
    def generate(
        self,
        chunks: List[Dict],
        processed_ids: Set[str],
    ) -> Set[str]:
        """Generate embeddings for all chunks, saving each batch immediately.
        
        Args:
            chunks: List of chunk dictionaries with 'id' and 'content' keys
            processed_ids: Set of already processed chunk IDs (will be updated)
            
        Returns:
            Updated set of processed IDs
        """
        # Filter to unprocessed chunks
        chunks_to_process = [c for c in chunks if c['id'] not in processed_ids]
        
        if not chunks_to_process:
            print("   ✓ All chunks already have embeddings!")
            return processed_ids
        
        total_chunks = len(chunks)
        remaining = len(chunks_to_process)
        batch_size = self.config.embedding_batch_size
        total_batches = (remaining + batch_size - 1) // batch_size
        
        print(f"\n🧠 Generating {self.config.embedding_dimensions}-dim dense embeddings...")
        print(f"   Model: {self.config.embedding_model}")
        print(f"   Already processed: {len(processed_ids)}/{total_chunks}")
        print(f"   Remaining: {remaining} chunks in {total_batches} batches")
        
        # Estimate time
        est_minutes = total_batches * 0.5 / 60
        print(f"   Estimated time: {est_minutes:.1f} minutes")
        
        start_time = time.time()
        batches_since_save = 0
        
        for i in range(0, remaining, batch_size):
            batch = chunks_to_process[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Generate embeddings
                texts = [c['content'] for c in batch]
                embeddings = self._generate_batch(texts)
                
                # Save each embedding immediately (atomic per-file)
                batch_embeddings = {}
                for j, chunk in enumerate(batch):
                    batch_embeddings[chunk['id']] = embeddings[j]
                    processed_ids.add(chunk['id'])
                
                # Save batch to individual files
                self.checkpoint_mgr.save_embeddings_batch(batch_embeddings)
                batches_since_save += 1
                
                # Save state periodically
                if batches_since_save >= self.config.checkpoint_save_interval:
                    self.checkpoint_mgr.save_state(processed_ids, stage="embeddings")
                    batches_since_save = 0
                
                # Progress
                processed = len(processed_ids)
                pct = processed / total_chunks * 100
                elapsed = time.time() - start_time
                rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
                eta = (remaining - i - len(batch)) / rate if rate > 0 else 0
                
                print(f"   [{batch_num}/{total_batches}] {processed}/{total_chunks} ({pct:.1f}%) - ETA: {eta/60:.1f}m")
                
            except Exception as e:
                print(f"\n   ❌ Error at batch {batch_num}: {e}")
                print(f"   💾 Saving state before exit...")
                self.checkpoint_mgr.save_state(processed_ids, stage="embeddings")
                raise
        
        # Final state save
        self.checkpoint_mgr.save_state(processed_ids, stage="embeddings_complete")
        
        elapsed = time.time() - start_time
        print(f"\n   ✓ Generated {remaining} embeddings in {elapsed/60:.1f} minutes")
        
        return processed_ids
    
    def apply_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Load and apply embeddings to chunks from cache using parallel I/O."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"\n📦 Loading {len(chunks):,} embeddings from cache...")
        
        total = len(chunks)
        loaded = 0
        missing = []
        
        # Create a mapping for fast lookup
        chunk_map = {c['id']: c for c in chunks}
        chunk_ids = list(chunk_map.keys())
        
        # Load in parallel using thread pool (I/O bound)
        def load_one(chunk_id: str):
            return chunk_id, self.checkpoint_mgr.load_embedding(chunk_id)
        
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(load_one, cid): cid for cid in chunk_ids}
            
            for i, future in enumerate(as_completed(futures)):
                chunk_id, embedding = future.result()
                if embedding is not None:
                    chunk_map[chunk_id]['embedding'] = embedding
                    loaded += 1
                else:
                    missing.append(chunk_id)
                
                # Progress every 5000
                if (i + 1) % 5000 == 0 or (i + 1) == total:
                    pct = (i + 1) / total * 100
                    print(f"   [{i + 1:,}/{total:,}] ({pct:.1f}%)")
        
        if missing:
            print(f"   ⚠️  Missing {len(missing)} embeddings")
            raise RuntimeError(f"Missing embeddings for {len(missing)} chunks. Run embedding generation first.")
        
        print(f"   ✓ Loaded {loaded:,} embeddings")
        return chunks
