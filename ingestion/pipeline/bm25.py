"""BM25 sparse encoder for hybrid search.

Key features:
- Fits on corpus to build vocabulary and IDF scores
- Encodes documents to sparse vectors
- Saves/loads encoder state
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List


class BM25SparseEncoder:
    """BM25 sparse encoder for keyword matching in hybrid search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self._fitted: bool = False
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer with lowercasing."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]
    
    def fit(self, documents: List[str]) -> 'BM25SparseEncoder':
        """Fit BM25 on a corpus."""
        print(f"\n🔢 Fitting BM25 on {len(documents)} documents...")
        
        self.doc_count = len(documents)
        doc_freqs: Dict[str, int] = {}
        total_length = 0
        
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            total_length += len(tokens)
            seen = set()
            
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                if token not in seen:
                    doc_freqs[token] = doc_freqs.get(token, 0) + 1
                    seen.add(token)
            
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i + 1}/{len(documents)} documents...")
        
        self.avgdl = total_length / self.doc_count if self.doc_count > 0 else 0
        
        for token, df in doc_freqs.items():
            self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        
        self._fitted = True
        print(f"   ✓ Vocabulary size: {self.vocab_size:,}")
        print(f"   ✓ Average document length: {self.avgdl:.1f} tokens")
        
        return self
    
    def encode(self, text: str) -> Dict[str, List]:
        """Encode text to sparse BM25 vector."""
        if not self._fitted:
            raise ValueError("BM25 encoder not fitted. Call fit() first.")
        
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        
        tf: Dict[str, int] = {}
        for token in tokens:
            if token in self.vocab:
                tf[token] = tf.get(token, 0) + 1
        
        dimensions = []
        values = []
        
        for token, freq in tf.items():
            if token in self.idf:
                idf = self.idf[token]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf * (numerator / denominator)
                
                if score > 0:
                    dimensions.append(self.vocab[token])
                    values.append(float(score))
        
        if dimensions:
            sorted_pairs = sorted(zip(dimensions, values))
            dimensions, values = zip(*sorted_pairs)
            dimensions = list(dimensions)
            values = list(values)
        
        return {"values": values, "dimensions": dimensions}
    
    def encode_batch(self, texts: List[str]) -> List[Dict[str, List]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def save(self, path: str) -> None:
        """Save encoder to JSON file."""
        if not self._fitted:
            raise ValueError("Cannot save unfitted encoder")
        
        data = {
            "k1": self.k1,
            "b": self.b,
            "vocab": self.vocab,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "doc_count": self.doc_count,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"   💾 Saved BM25 encoder to {path}")
    
    def load(self, path: str) -> 'BM25SparseEncoder':
        """Load encoder from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.k1 = data["k1"]
        self.b = data["b"]
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.avgdl = data["avgdl"]
        self.doc_count = data["doc_count"]
        self._fitted = True
        
        print(f"   📂 Loaded BM25 encoder: {self.vocab_size:,} terms")
        return self
    
    @classmethod
    def from_file(cls, path: str) -> 'BM25SparseEncoder':
        """Create encoder from saved file."""
        encoder = cls()
        return encoder.load(path)


def add_sparse_embeddings(chunks: List[Dict], bm25: BM25SparseEncoder) -> List[Dict]:
    """Add sparse embeddings to chunks."""
    print(f"\n🔢 Generating BM25 sparse embeddings for {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        chunk['sparse_embedding'] = bm25.encode(chunk['content'])
        
        if (i + 1) % 10000 == 0:
            print(f"   Processed {i + 1}/{len(chunks)} chunks...")
    
    print(f"   ✓ Added sparse embeddings to {len(chunks)} chunks")
    return chunks
