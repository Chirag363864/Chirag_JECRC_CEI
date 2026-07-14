import os
import sys
import time
import threading
import torch
from flask import Flask, render_template, jsonify, request
from model import GPTConfig, GPT
from tokenizer import SimpleBPETokenizer
from data_loader import DataLoader

# Initialize Flask App
app = Flask(__name__, template_folder="templates")

# Global variables to handle system state across Flask threads
state = {
    "is_training": False,
    "current_iter": 0,
    "max_iters": 300,
    "train_loss": [],
    "val_loss": [],
    "logs": [],
    "model_config": None,
    "device_str": "cpu",
    "stop_requested": False,
    "dataset": "shakespeare"
}

# Global references
model = None
tokenizer = None
data_loader = None
train_thread = None
state_lock = threading.Lock()

def initialize_default_components():
    global model, tokenizer, data_loader, state
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device_str"] = device
    
    # Default parameters
    dataset_name = "shakespeare"
    config = GPTConfig(
        vocab_size=2048,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        norm_type='layernorm',
        pos_emb_type='absolute',
        mlp_type='gelu'
    )
    
    # Check if a checkpoint exists, load it first to recover config and dataset name
    ckpt_path = os.path.join("out", "best_checkpoint.pt")
    checkpoint_loaded = False
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            config = checkpoint['config']
            dataset_name = checkpoint.get("dataset", "shakespeare")
            
            model = GPT(config)
            sd = checkpoint['model']
            for k, v in list(sd.items()):
                if k.startswith('_orig_mod.'):
                    sd[k[10:]] = sd.pop(k)
            model.load_state_dict(sd)
            checkpoint_loaded = True
            print(f"Loaded checkpoint from {ckpt_path} successfully (Dataset: {dataset_name}).")
        except Exception as e:
            print("Could not load checkpoint, starting fresh:", e)
            
    state["dataset"] = dataset_name
    state["model_config"] = config
    
    if not checkpoint_loaded:
        model = GPT(config)
        
    model.to(device)
    model.eval()
    
    # Load Tokenizer matching the dataset
    tokenizer_path = os.path.join("data", f"tokenizer_{dataset_name}.json")
    tokenizer = SimpleBPETokenizer()
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        # Fallback if not prepared yet
        dummy_text = "JULIET:\nRomeo, wherefore art thou Romeo?\nROMEO:\nLady, by yonder blessed moon I vow."
        tokenizer.train(dummy_text, vocab_size=512, max_text_len=1000)
        os.makedirs("data", exist_ok=True)
        tokenizer.save(tokenizer_path)

# --- Asynchronous Training Thread Worker ---

def training_worker(cfg, train_args):
    global model, data_loader, state
    
    device = state["device_str"]
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    try:
        # Re-initialize DataLoader with target parameters
        data_loader = DataLoader(
            data_dir="data",
            block_size=cfg.block_size,
            batch_size=train_args["batch_size"],
            vocab_size=cfg.vocab_size,
            dataset_name=train_args["dataset"]
        )
        data_loader.prepare_data(force=False)
        
        get_batch_train = data_loader.get_batch_generator('train')
        get_batch_val = data_loader.get_batch_generator('val')
        
        # Instantiate Model with selected architecture settings
        with state_lock:
            model = GPT(cfg)
            model.to(device)
            model.train()
            
        # Optimization
        from train import configure_optimizers, estimate_loss
        optimizer = configure_optimizers(
            model, 
            weight_decay=0.1, 
            learning_rate=train_args["lr"], 
            betas=(0.9, 0.95), 
            device_type=device_type
        )
        
        ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.amp.autocast(device_type='cpu', enabled=False)
        
        iter_num = 0
        max_iters = train_args["max_iters"]
        
        # Reset local metrics in state
        with state_lock:
            state["train_loss"] = []
            state["val_loss"] = []
            state["current_iter"] = 0
            state["max_iters"] = max_iters
            state["logs"] = ["Starting Training Loop..."]
            
        print("Background training launched.")
        
        X, Y = get_batch_train()
        X, Y = X.to(device), Y.to(device)
        
        while iter_num <= max_iters:
            # Check for early termination request
            with state_lock:
                if state["stop_requested"]:
                    state["logs"].append("Training halted by request.")
                    break
                    
            t_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            
            with ctx:
                _, loss = model(X, Y)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Fetch next batch
            X, Y = get_batch_train()
            X, Y = X.to(device), Y.to(device)
            
            t_end = time.time()
            dt = t_end - t_start
            
            # Evaluation and print cycle
            if iter_num % 10 == 0:
                with state_lock:
                    state["current_iter"] = iter_num
                    state["train_loss"].append({"iter": iter_num, "loss": loss.item()})
                    log_line = f"Iter {iter_num}/{max_iters} | Loss: {loss.item():.4f} | Step Time: {dt*1000:.1f}ms"
                    state["logs"].append(log_line)
                    
            if iter_num % 50 == 0:
                model.eval()
                with torch.no_grad():
                    losses = estimate_loss(model, 10, get_batch_train, get_batch_val, ctx)
                model.train()
                
                with state_lock:
                    state["val_loss"].append({"iter": iter_num, "loss": losses["val"]})
                    log_line = f"[Eval Cycle] Step {iter_num} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}"
                    state["logs"].append(log_line)
                    
                    os.makedirs("out", exist_ok=True)
                    ckpt_path = os.path.join("out", "best_checkpoint.pt")
                    checkpoint = {
                        'model': model.state_dict(),
                        'config': cfg,
                        'iter_num': iter_num,
                        'best_val_loss': losses["val"],
                        'dataset': train_args["dataset"]
                    }
                    torch.save(checkpoint, ckpt_path)
                    state["logs"].append(f"Saved checkpoint to {ckpt_path}")
                    
            iter_num += 1
            
        with state_lock:
            state["logs"].append("Training completed.")
            
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print("Training Thread Error:\n", err)
        with state_lock:
            state["logs"].append(f"ERROR: {str(e)}")
    finally:
        with state_lock:
            state["is_training"] = False
            state["stop_requested"] = False
            if model is not None:
                model.eval()

# --- HTTP Routes ---

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/status', methods=['GET'])
def get_status():
    global state
    with state_lock:
        return jsonify({
            "is_training": state["is_training"],
            "current_iter": state["current_iter"],
            "max_iters": state["max_iters"],
            "train_loss": state["train_loss"],
            "val_loss": state["val_loss"],
            "logs": state["logs"][-30:], # return last 30 logs
            "device": state["device_str"],
            "dataset": state["dataset"],
            "config": {
                "vocab_size": state["model_config"].vocab_size,
                "block_size": state["model_config"].block_size,
                "n_layer": state["model_config"].n_layer,
                "n_head": state["model_config"].n_head,
                "n_embd": state["model_config"].n_embd,
                "norm_type": state["model_config"].norm_type,
                "pos_emb_type": state["model_config"].pos_emb_type,
                "mlp_type": state["model_config"].mlp_type,
                "n_kv_head": state["model_config"].n_kv_head
            }
        })

@app.route('/api/train/start', methods=['POST'])
def start_training():
    global train_thread, state
    
    if state["is_training"]:
        return jsonify({"success": False, "error": "Already training!"})
        
    data = request.json or {}
    
    # GQA heads validation
    n_head = int(data.get("n_head", 2))
    n_kv_head_raw = data.get("n_kv_head")
    n_kv_head = int(n_kv_head_raw) if (n_kv_head_raw is not None and str(n_kv_head_raw).isdigit()) else None
    
    if n_kv_head is not None and n_kv_head > 0:
        if n_head % n_kv_head != 0:
            return jsonify({"success": False, "error": f"Number of attention heads ({n_head}) must be divisible by Key-Value heads ({n_kv_head}) for Grouped-Query Attention."})

    dataset = data.get("dataset", "shakespeare")
    
    # Parse Configurations
    cfg = GPTConfig(
        vocab_size=2048, # tiny-shakespeare/dante vocab fits BPE
        block_size=int(data.get("block_size", 128)),
        n_layer=int(data.get("n_layer", 2)),
        n_head=n_head,
        n_embd=int(data.get("n_embd", 64)),
        norm_type=data.get("norm_type", "layernorm"),
        pos_emb_type=data.get("pos_emb_type", "absolute"),
        mlp_type=data.get("mlp_type", "gelu"),
        n_kv_head=n_kv_head
    )
    
    train_args = {
        "batch_size": int(data.get("batch_size", 4)),
        "lr": float(data.get("learning_rate", 6e-4)),
        "max_iters": int(data.get("max_iters", 150)),
        "dataset": dataset
    }
    
    with state_lock:
        state["model_config"] = cfg
        state["dataset"] = dataset
        state["is_training"] = True
        state["stop_requested"] = False
        
    train_thread = threading.Thread(target=training_worker, args=(cfg, train_args))
    train_thread.daemon = True
    train_thread.start()
    
    return jsonify({"success": True})

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    global state
    if not state["is_training"]:
        return jsonify({"success": False, "error": "Not training."})
        
    with state_lock:
        state["stop_requested"] = True
        
    return jsonify({"success": True})

@app.route('/api/generate', methods=['POST'])
def generate_text():
    global model, tokenizer
    
    if model is None:
        return jsonify({"error": "Model not initialized."})
        
    data = request.json or {}
    prompt = data.get("prompt", "JULIET:")
    temp = float(data.get("temperature", 0.8))
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.9)
    max_tokens = int(data.get("max_tokens", 40))
    
    if top_k is not None:
        top_k = int(top_k)
    if top_p is not None:
        top_p = float(top_p)
        
    device = state["device_str"]
    
    # 1. Ensure correct tokenizer is loaded matching active dataset
    dataset_name = state.get("dataset", "shakespeare")
    tokenizer_path = os.path.join("data", f"tokenizer_{dataset_name}.json")
    tokenizer = SimpleBPETokenizer()
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        # Fallback if not prepared yet
        dummy_text = "JULIET:\nRomeo, wherefore art thou Romeo?\nROMEO:\nLady, by yonder blessed moon I vow."
        tokenizer.train(dummy_text, vocab_size=512, max_text_len=1000)
        os.makedirs("data", exist_ok=True)
        tokenizer.save(tokenizer_path)
        
    # 2. Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [256]
        
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # 2. Autoregressive generate
    with torch.no_grad():
        y = model.generate(x, max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
        full_seq_ids = y[0].tolist()
        
    # Decode individual tokens so the UI can represent them and map weights
    decoded_tokens = []
    # To decode tokens individually, we need to inspect BPE tokens.
    # Note that BPE encodes string combinations. We can print each token by decoding it.
    for tid in full_seq_ids:
        # Decode individual token
        tok_str = tokenizer.decode([tid])
        decoded_tokens.append(tok_str)
        
    generated_text = tokenizer.decode(full_seq_ids)
    
    # 3. Retrieve Self-Attention weights for visualizer
    # Recall shapes of weights: list of layers, each layer is list of heads, each head is (T, T) matrix.
    # We clip T to the last 20 tokens to guarantee standard visualization boxes
    T = min(len(full_seq_ids), 20)
    
    attn_data = [] # Shape: [layer][head][token_row][token_col]
    for layer_idx, block in enumerate(model.transformer.h):
        weights = block.attn.last_attn_weights # shape (nh, seq_len, seq_len)
        if weights is not None:
            # We slice the last T tokens along spatial dimensions
            # shape (nh, T, T)
            sliced = weights[:, -T:, -T:]
            # Convert to list
            attn_data.append(sliced.tolist())
            
    # Sliced decoded tokens as well
    sliced_tokens = decoded_tokens[-T:]
    
    # Escape spaces or special carriage returns for HTML/JSON formatting
    esc_sliced_tokens = []
    for t in sliced_tokens:
        clean = t.replace("\n", "↵ ").replace(" ", "␣ ")
        esc_sliced_tokens.append(clean)
        
    return jsonify({
        "generated_text": generated_text,
        "tokens": esc_sliced_tokens,
        "attention": attn_data, # [layer][head][row][col]
        "num_layers": len(attn_data),
        "num_heads": len(attn_data[0]) if attn_data else 0
    })

# --- Bootstrap ---

# Initialize everything immediately on launch
initialize_default_components()

if __name__ == "__main__":
    # Host on 127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
