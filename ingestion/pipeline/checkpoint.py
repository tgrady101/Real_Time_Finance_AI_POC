"""Atomic checkpoint management with corruption recovery.

Key features:
- Atomic writes using temp files + rename (prevents corruption)
- Automatic backup rotation
- Corruption detection and recovery
- Separate metadata from large data files
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class CheckpointManager:
    """Manages checkpoints with atomic writes and corruption recovery.
    
    Checkpoint structure:
    - {job}_state.json: Small metadata file (processed IDs, timestamps)
    - embeddings_cache/{chunk_id}.json: Individual embedding files (crash-safe)
    - {job}_state.json.bak1, .bak2, .bak3: Rotating backups
    """
    
    def __init__(self, checkpoint_dir: Path, embeddings_dir: Path, job_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.embeddings_dir = embeddings_dir
        self.job_name = job_name
        self.state_file = checkpoint_dir / f"{job_name}_state.json"
        self.max_backups = 3
        
        # Ensure directories exist
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def _atomic_write(self, path: Path, data: Any) -> None:
        """Write file atomically using temp file + rename."""
        temp_path = path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic rename (safe on most filesystems)
            temp_path.replace(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _rotate_backups(self) -> None:
        """Rotate backup files."""
        for i in range(self.max_backups, 0, -1):
            old_backup = Path(f"{self.state_file}.bak{i}")
            if i == self.max_backups and old_backup.exists():
                old_backup.unlink()
            elif old_backup.exists():
                new_backup = Path(f"{self.state_file}.bak{i+1}")
                old_backup.rename(new_backup)
        
        # Current state becomes backup 1
        if self.state_file.exists():
            backup1 = Path(f"{self.state_file}.bak1")
            shutil.copy2(self.state_file, backup1)
    
    def save_state(self, processed_ids: Set[str], stage: str = "embeddings") -> None:
        """Save checkpoint state (just metadata, not embeddings)."""
        self._rotate_backups()
        
        state = {
            "job_name": self.job_name,
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "processed_count": len(processed_ids),
            "processed_ids": list(processed_ids),
        }
        self._atomic_write(self.state_file, state)
        print(f"   💾 State saved: {len(processed_ids)} processed")
    
    def save_embedding(self, chunk_id: str, embedding: List[float]) -> None:
        """Save a single embedding to its own file (atomic)."""
        # Use subdirectories to avoid too many files in one folder
        prefix = chunk_id[:4] if len(chunk_id) >= 4 else chunk_id
        subdir = self.embeddings_dir / prefix
        subdir.mkdir(parents=True, exist_ok=True)
        
        embedding_file = subdir / f"{chunk_id}.json"
        self._atomic_write(embedding_file, embedding)
    
    def save_embeddings_batch(self, embeddings: Dict[str, List[float]]) -> None:
        """Save a batch of embeddings atomically."""
        for chunk_id, embedding in embeddings.items():
            self.save_embedding(chunk_id, embedding)
    
    def load_state(self) -> Tuple[Set[str], str]:
        """Load checkpoint state, recovering from corruption if needed."""
        # Try main state file first
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                processed_ids = set(state.get('processed_ids', []))
                stage = state.get('stage', 'embeddings')
                print(f"   📂 Loaded state: {len(processed_ids)} processed from {state.get('timestamp', 'unknown')}")
                return processed_ids, stage
            except json.JSONDecodeError:
                print(f"   ⚠️  State file corrupted, trying backups...")
        
        # Try backups
        for i in range(1, self.max_backups + 1):
            backup = Path(f"{self.state_file}.bak{i}")
            if backup.exists():
                try:
                    with open(backup, 'r') as f:
                        state = json.load(f)
                    processed_ids = set(state.get('processed_ids', []))
                    stage = state.get('stage', 'embeddings')
                    print(f"   📂 Recovered from backup {i}: {len(processed_ids)} processed")
                    return processed_ids, stage
                except json.JSONDecodeError:
                    continue
        
        # No valid state found - scan embeddings directory
        print(f"   📂 No valid state, scanning embeddings directory...")
        return self._recover_from_embeddings(), 'embeddings'
    
    def _recover_from_embeddings(self) -> Set[str]:
        """Recover processed IDs by scanning embeddings directory."""
        processed_ids = set()
        
        if not self.embeddings_dir.exists():
            return processed_ids
        
        for subdir in self.embeddings_dir.iterdir():
            if subdir.is_dir():
                for embedding_file in subdir.glob("*.json"):
                    chunk_id = embedding_file.stem
                    # Verify it's valid
                    try:
                        with open(embedding_file, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            processed_ids.add(chunk_id)
                    except (json.JSONDecodeError, Exception):
                        # Corrupted file, remove it
                        embedding_file.unlink()
        
        print(f"   📂 Recovered {len(processed_ids)} embeddings from cache")
        return processed_ids
    
    def load_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Load a single embedding from cache."""
        prefix = chunk_id[:4] if len(chunk_id) >= 4 else chunk_id
        embedding_file = self.embeddings_dir / prefix / f"{chunk_id}.json"
        
        if not embedding_file.exists():
            return None
        
        try:
            with open(embedding_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Corrupted, remove it
            embedding_file.unlink()
            return None
    
    def load_all_embeddings(self, chunk_ids: List[str]) -> Dict[str, List[float]]:
        """Load all embeddings for given chunk IDs."""
        embeddings = {}
        for chunk_id in chunk_ids:
            emb = self.load_embedding(chunk_id)
            if emb is not None:
                embeddings[chunk_id] = emb
        return embeddings
    
    def clear(self) -> None:
        """Clear all checkpoint files."""
        # Clear state files
        if self.state_file.exists():
            self.state_file.unlink()
        for i in range(1, self.max_backups + 1):
            backup = Path(f"{self.state_file}.bak{i}")
            if backup.exists():
                backup.unlink()
        
        # Clear embeddings directory
        if self.embeddings_dir.exists():
            shutil.rmtree(self.embeddings_dir)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   🧹 Cleared all checkpoints")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        stats = {
            "state_file_exists": self.state_file.exists(),
            "embeddings_count": 0,
            "embeddings_size_mb": 0.0,
        }
        
        if self.embeddings_dir.exists():
            total_size = 0
            count = 0
            for subdir in self.embeddings_dir.iterdir():
                if subdir.is_dir():
                    for f in subdir.glob("*.json"):
                        count += 1
                        total_size += f.stat().st_size
            stats["embeddings_count"] = count
            stats["embeddings_size_mb"] = total_size / (1024 * 1024)
        
        return stats
