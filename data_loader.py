import os
import urllib.request
import numpy as np
import torch
from tokenizer import SimpleBPETokenizer

# Dataset source URLs
DATASETS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "dante": "https://raw.githubusercontent.com/asperti/Dante/master/inferno.txt"
}

class DataLoader:
    def __init__(self, data_dir="data", block_size=256, batch_size=64, vocab_size=2048, dataset_name="shakespeare"):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name if dataset_name in DATASETS else "shakespeare"
        self.data_url = DATASETS[self.dataset_name]
        
        self.input_file_path = os.path.join(data_dir, f"input_{self.dataset_name}.txt")
        self.train_bin_path = os.path.join(data_dir, f"train_{self.dataset_name}.bin")
        self.val_bin_path = os.path.join(data_dir, f"val_{self.dataset_name}.bin")
        self.tokenizer_path = os.path.join(data_dir, f"tokenizer_{self.dataset_name}.json")
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load or initialize tokenizer
        self.tokenizer = SimpleBPETokenizer()
        
    def prepare_data(self, force=False):
        """Downloads input.txt, trains BPE tokenizer, and encodes into binary files."""
        # 1. Download dataset if not present
        if not os.path.exists(self.input_file_path) or force:
            print(f"Downloading {self.dataset_name} dataset from {self.data_url}...")
            urllib.request.urlretrieve(self.data_url, self.input_file_path)
            print("Download completed successfully.")
            
        with open(self.input_file_path, "r", encoding="utf-8") as f:
            data = f.read()
            
        print(f"Dataset loaded. Total length (characters): {len(data)}")
        
        # Split into train (90%) and validation (10%)
        n = len(data)
        train_text = data[:int(n*0.9)]
        val_text = data[int(n*0.9):]
        
        # 2. Train tokenizer on train portion
        if not os.path.exists(self.tokenizer_path) or force:
            print("Training Custom BPE Tokenizer on training split...")
            # We train on the first 20,000 chars of train text for quick Python execution for demo
            self.tokenizer.train(train_text, vocab_size=self.vocab_size, max_text_len=None, verbose=True)
            print(f"Saving trained tokenizer to {self.tokenizer_path}...")
            self.tokenizer.save(self.tokenizer_path)
        else:
            print(f"Loading existing tokenizer from {self.tokenizer_path}...")
            self.tokenizer.load(self.tokenizer_path)
            
        # 3. Tokenize train and val splits
        if not os.path.exists(self.train_bin_path) or not os.path.exists(self.val_bin_path) or force:
            print("Encoding splits into token IDs...")
            train_ids = self.tokenizer.encode(train_text)
            val_ids = self.tokenizer.encode(val_text)
            
            print(f"Train split has {len(train_ids):,} tokens")
            print(f"Validation split has {len(val_ids):,} tokens")
            
            # Export to binary files (uint16 is plenty for vocab < 65,536)
            train_ids = np.array(train_ids, dtype=np.uint16)
            val_ids = np.array(val_ids, dtype=np.uint16)
            
            train_ids.tofile(self.train_bin_path)
            val_ids.tofile(self.val_bin_path)
            print("Binary splits written successfully.")
        else:
            print("Pre-tokenized binary files already exist.")

    def get_batch_generator(self, split):
        """Creates standard memory-mapped batch generator for training/validation."""
        bin_path = self.train_bin_path if split == 'train' else self.val_bin_path
        
        # Check files
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at {bin_path}. Run prepare_data() first.")
            
        # Memory map the dataset file
        data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        def get_batch():
            # Grab random start indices
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
            return x, y
            
        return get_batch


if __name__ == "__main__":
    # Test data preparation for both datasets
    for d_name in ["shakespeare", "dante"]:
        print(f"\n--- Testing DataLoader with dataset: {d_name} ---")
        loader = DataLoader(data_dir="data", block_size=128, batch_size=8, vocab_size=512, dataset_name=d_name)
        loader.prepare_data(force=False)
        
        get_batch = loader.get_batch_generator('train')
        X, Y = get_batch()
        print("DataLoader validation passes shape checks:")
        print("X Shape (input batch):", X.shape)
        print("Y Shape (target batch):", Y.shape)
        
        # Check decoder
        sample_seq = X[0].tolist()
        print("Decoded sample string:\n", loader.tokenizer.decode(sample_seq[:40]))
