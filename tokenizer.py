import os
import json
import re

class SimpleBPETokenizer:
    def __init__(self):
        # Base vocabulary is raw bytes (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Merges dictionary: (parent1, parent2) -> child
        self.merges = {}
        # Special tokens
        self.special_tokens = {"<|endoftext|>": 256}
        self.vocab[256] = b"<|endoftext|>"
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        # Encoding cache for performance
        self.cache = {}

    def train(self, text, vocab_size, max_text_len=200000, verbose=False):
        """Trains the token vocab using Byte-Pair Encoding on the input text.
        To avoid slow training in raw Python, we clip the text to max_text_len bytes.
        """
        assert vocab_size > 257, "vocab_size must be greater than 257"
        num_merges = vocab_size - 256 - len(self.special_tokens)
        
        # Convert text to UTF-8 bytes, truncated for speed
        raw_bytes = text.encode("utf-8")
        if max_text_len is not None and len(raw_bytes) > max_text_len:
            raw_bytes = raw_bytes[:max_text_len]
        
        # Convert to list of integers
        ids = list(raw_bytes)
        
        # Iteratively merge the most common pairs
        for i in range(num_merges):
            # Count pair counts
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
                
            if not counts:
                break # No more pairs to merge
                
            # Find the most frequent pair
            best_pair = max(counts, key=counts.get)
            
            # Create new token ID
            new_id = 256 + len(self.special_tokens) + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Replace the pair in ids list
            ids = self._merge(ids, best_pair, new_id)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id} (count: {counts[best_pair]})")

    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text, allowed_special=True):
        """Encodes text to a list of token integers with word-token caching."""
        if not text:
            return []
            
        # If input is bytes or string, ensure it's a string for regex split
        if isinstance(text, bytes):
            text_str = text.decode("utf-8", errors="replace")
        else:
            text_str = text
            
        # Split text into word-like chunks using Regex
        chunks = re.findall(r'\w+|[^\w\s]|\s+', text_str)
        
        # Apply the merges in order they were created (lower merged IDs first)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        
        # We will cache the tokenization of each chunk
        encoded_ids = []
        for chunk in chunks:
            if chunk in self.cache:
                encoded_ids.extend(self.cache[chunk])
                continue
                
            # If not in cache, encode it
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = list(chunk_bytes)
            
            for pair, new_id in sorted_merges:
                chunk_ids = self._merge(chunk_ids, pair, new_id)
                
            # Save to cache
            self.cache[chunk] = chunk_ids
            encoded_ids.extend(chunk_ids)
            
        return encoded_ids

    def decode(self, ids):
        """Decodes a list of token integers to a string."""
        byte_parts = []
        for idx in ids:
            if idx in self.vocab:
                byte_parts.append(self.vocab[idx])
            elif idx == 256:
                byte_parts.append(b"<|endoftext|>")
            else:
                raise ValueError(f"invalid token id {idx}")
                
        # Combine bytes and decode to string
        concatenated = b"".join(byte_parts)
        return concatenated.decode("utf-8", errors="replace")

    def save(self, file_path):
        """Saves the vocabulary and merges to file."""
        # Convert merges keys from tuple (int, int) to string "int,int" for JSON compatibility
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "merges": serializable_merges,
            "vocab_size": len(self.vocab)
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path):
        """Loads vocabulary and merges from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Load merges
        self.merges = {}
        for k, v in data["merges"].items():
            k_tuple = tuple(map(int, k.split(",")))
            self.merges[k_tuple] = v
            
        # Reconstruct vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab[256] = b"<|endoftext|>"
        
        # Sort merges by value to reconstruct vocab in correct order
        for pair, new_id in sorted(self.merges.items(), key=lambda x: x[1]):
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        # Clear cache when loading new merges
        self.cache = {}


if __name__ == "__main__":
    # Self-test the BPE Tokenizer
    sample_text = "hello world! hello world, this is a beautiful day. hello world!"
    tokenizer = SimpleBPETokenizer()
    print("Training BPE tokenizer...")
    tokenizer.train(sample_text, vocab_size=300, verbose=True)
    
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("\nOriginal text length:", len(sample_text))
    print("Encoded sequence length (tokens):", len(encoded))
    print("Decoded text matches original:", decoded == sample_text)
    print("Encoded sequence:", encoded)
    
    # Save and Load test
    temp_path = "test_tokenizer_model.json"
    tokenizer.save(temp_path)
    
    loader_token = SimpleBPETokenizer()
    loader_token.load(temp_path)
    print("Loads successfully and encodes match:", loader_token.encode(sample_text) == encoded)
    
    # Cleanup temp test file
    if os.path.exists(temp_path):
        os.remove(temp_path)
