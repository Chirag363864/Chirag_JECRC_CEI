import os
import argparse
import torch
from model import GPTConfig, GPT
from tokenizer import SimpleBPETokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a trained Mini GPT-2 checkpoint")
    parser.add_argument("--checkpoint", type=str, default="out/best_checkpoint.pt", help="Path to checkpoint file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name ('shakespeare' or 'dante'). Used to auto-locate the tokenizer. "
                             "If not provided, inferred from checkpoint metadata.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Explicit path to tokenizer JSON. Overrides --dataset if provided.")
    parser.add_argument("--prompt", type=str, default="ROMEO:\nWhat light through yonder window breaks?", help="Input prompt seed string")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum number of tokens to generate per sample")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature scaling factor (>0)")
    parser.add_argument("--top_k", type=int, default=50, help="Filter to top-K tokens at each step (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Filter to top-P nucleus tokens (1.0 to disable)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, cuda, mps")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
        
    print(f"Loading generator on device: {device}")
    
    # 2. Check checkpoint existence
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}. Run train.py first.")
        
    # 3. Load model checkpoints
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    # Recover dataset name saved in checkpoint (fall back to CLI arg or 'shakespeare')
    ckpt_dataset = checkpoint.get('dataset', None)
    
    print("\nModel details recovered from checkpoint:")
    print(config)
    print(f"Trained at step: {checkpoint['iter_num']} | Validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # 4. Instantiate and load model weights
    model = GPT(config)
    
    # Load parameters (handling torch.compile prefixes if present)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 5. Resolve tokenizer path
    # Priority: explicit --tokenizer_path > --dataset flag > checkpoint metadata > default 'shakespeare'
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        dataset_name = args.dataset or ckpt_dataset or "shakespeare"
        # Try dataset-specific path first, then legacy fallback
        candidate = os.path.join("data", f"tokenizer_{dataset_name}.json")
        legacy   = os.path.join("data", "tokenizer.json")
        if os.path.exists(candidate):
            tokenizer_path = candidate
        elif os.path.exists(legacy):
            tokenizer_path = legacy
        else:
            raise FileNotFoundError(
                f"Tokenizer file not found. Tried:\n  {candidate}\n  {legacy}\n"
                "Run train.py or data_loader.py first to generate it."
            )

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train.py / data_loader.py first.")

    tokenizer = SimpleBPETokenizer()
    tokenizer.load(tokenizer_path)
    print(f"Tokenizer loaded from: {tokenizer_path}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    # 6. Encode seed prompt
    print(f"\nPrompt: '{args.prompt}'")
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        # Fallback if empty prompt
        prompt_ids = [256] # <|endoftext|>
        
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # 7. Generate multiple samples
    print(f"\nGenerating {args.num_samples} samples...")
    for s in range(args.num_samples):
        print(f"\n=================== Sample {s+1} ===================")
        with torch.no_grad():
            y = model.generate(
                x, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature, 
                top_k=args.top_k, 
                top_p=args.top_p
            )
            # Fetch generated tokens only
            gen_ids = y[0].tolist()
            text = tokenizer.decode(gen_ids)
            # Safe print to screen to avoid Unicode mapping errors in Windows terminal
            import sys
            try:
                print(text)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or 'utf-8'
                print(text.encode(encoding, errors='replace').decode(encoding))
        print("==================================================")
        
if __name__ == "__main__":
    main()
