from torch._library import fake_class_registry
import os
import csv
import time
import math
import argparse
import torch
from model import GPTConfig, GPT
from data_loader import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mini GPT-2 model from scratch")
    # File Paths
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for data files")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to save checkpoints")

    # Model Config
    parser.add_argument("--vocab_size", type=int, default=2048, help="Vocabulary size")
    parser.add_argument("--block_size", type=int, default=256, help="Context size")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_kv_head", type=int, default=None, help="Number of key/value heads for GQA/MQA (None for MHA)")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")

    # Dataset Selection
    parser.add_argument("--dataset", type=str, default="shakespeare", choices=["shakespeare", "dante"], help="Select dataset")

    # Architecture Flags
    parser.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"], help="Normalization type")
    parser.add_argument("--pos_emb_type", type=str, default="absolute", choices=["absolute", "rope"], help="Positional embeddings")
    parser.add_argument("--mlp_type", type=str, default="gelu", choices=["gelu", "swiglu"], help="MLP activation block type")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout percentage")

    # Training Config
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--max_iters", type=int, default=600, help="Total training iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay weight")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")

    # LR Decay Config
    parser.add_argument("--decay_lr", type=bool, default=True, help="Decay learning rate")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum decayed learning rate")

    # System Config
    parser.add_argument("--eval_interval", type=int, default=50, help="How often to evaluate val loss")
    parser.add_argument("--eval_iters", type=int, default=20, help="How many batches to average for evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, cuda, mps")
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile (requires PyTorch 2.0+)")

    return parser.parse_args()

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # Param dictionary of grad updates
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # 2D weight matrices decay, 1D biases and layerNorm shapes (weight/bias) do not.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    # Ensure correct optimizer setup
    use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer

def estimate_loss(model, eval_iters, get_batch_train, get_batch_val, ctx):
    """Averages losses over eval_iters batches to calculate stable logs."""
    out = {}
    model.eval()
    for split, get_batch in [('train', get_batch_train), ('val', get_batch_val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch()
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Setup Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # Handle Mixed Precision setup
    # Note: On CPU, only bfloat16 is supported for autocast. Fall back to float32 if not available.
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.amp.autocast(device_type='cpu', enabled=False)

    print(f"Using device: {device} (type: {device_type})")

    # Load dataset loader
    data_loader = DataLoader(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        dataset_name=args.dataset
    )

    # Prepare token data files
    data_loader.prepare_data(force=False)

    # Fetch batch generators
    get_batch_train = data_loader.get_batch_generator('train')
    get_batch_val = data_loader.get_batch_generator('val')

    # Model Config
    config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        norm_type=args.norm_type,
        pos_emb_type=args.pos_emb_type,
        mlp_type=args.mlp_type,
        n_kv_head=args.n_kv_head
    )

    print("Model hyperparameters selected:")
    print(config)

    # Instantiate Model
    model = GPT(config)
    model.to(device)

    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {num_params:,}")

    # Configure optimizer
    optimizer = configure_optimizers(
        model,
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device_type
    )

    # Optional PyTorch 2.0 compilation
    if args.compile:
        print("Compiling model (using torch.compile)...")
        model = torch.compile(model)

    # ===========================
    # Resume training
    ckpt_path = os.path.join(args.out_dir, "best_checkpoint.pt")

    if os.path.exists(ckpt_path):
        print("Loading checkpoint...")

        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        iter_num = checkpoint["iter_num"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming from iteration {iter_num}")

    else:
        iter_num = 0
        best_val_loss = 1e9

    # ------------------------------
    def get_lr(it):
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        if it > args.max_iters:
            return args.min_lr

        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # Training Loop
    t0 = time.time()
    X, Y = get_batch_train()
    X, Y = X.to(device), Y.to(device)

    print("\nStarting training run...")
    while iter_num <= args.max_iters:
        # Determine and set learning rate
        lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 1. Evaluation cycle
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, args.eval_iters, get_batch_train, get_batch_val, ctx)
            print(f"Step {iter_num}: Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e}")

            # Save best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'args': vars(args),
                    'dataset': args.dataset
                }
                torch.save(checkpoint, ckpt_path)
                print(f" -> Saved new best checkpoint to {ckpt_path} (Val Loss: {best_val_loss:.4f})")

            # Append to CSV training log
            log_path = os.path.join(args.out_dir, "train_log.csv")
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["iter", "train_loss", "val_loss", "lr"])
                writer.writerow([iter_num, f"{losses['train']:.6f}", f"{losses['val']:.6f}", f"{lr:.6e}"])

            # Autoregressive generation sample
            model.eval()
            with torch.no_grad():
                # Encode a seed prompt: "JULIET:" or similar
                seed_context = "JULIET:"
                seed_ids = data_loader.tokenizer.encode(seed_context)
                context_tensor = torch.tensor([seed_ids], dtype=torch.long, device=device)

                generated = model.generate(context_tensor, max_new_tokens=40, temperature=0.8, top_k=20)
                decoded_gen = data_loader.tokenizer.decode(generated[0].tolist())
                # Safe print to screen to avoid Unicode mapping errors in Windows terminal
                import sys
                safe_text = f"--- Generated Sample at iter {iter_num} ---\n{decoded_gen}\n-----------------------------------------"
                try:
                    print(safe_text)
                except UnicodeEncodeError:
                    encoding = sys.stdout.encoding or 'utf-8'
                    print(safe_text.encode(encoding, errors='replace').decode(encoding))
            model.train()

        # 2. Forward/Backward step
        t_step_start = time.time()

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(X, Y)

        # Retrieve next batch asynchronously while GPU/CPU computes backward pass
        X, Y = get_batch_train()
        X, Y = X.to(device), Y.to(device)

        loss.backward()

        # Gradient clipping
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        t_step_end = time.time()
        dt = t_step_end - t_step_start

        if iter_num % 10 == 0:
            print(f"Iter {iter_num}/{args.max_iters} | Loss: {loss.item():.4f} | Time: {dt*1000:.1f}ms")

        iter_num += 1

    print(f"\nTraining completed in {time.time() - t0:.2f} seconds.")
    print("Best validation loss achieved:", best_val_loss)

if __name__ == "__main__":
    main()