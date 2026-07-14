# 🧠 Mini GPT-2 from Scratch

> A decoder-only transformer language model built entirely from scratch, inspired by Andrej Karpathy's educational series. Trains on real text, generates coherent language, and ships with a live interactive web dashboard.

---

## ✨ Features at a Glance

| Feature | Description |
|---|---|
| **Custom BPE Tokenizer** | From-scratch Byte-Pair Encoding tokenizer with save/load |
| **Full Transformer Stack** | Token embeddings, causal self-attention, MLP, layer norm, LM head |
| **Flexible Architecture** | Swap between LayerNorm / RMSNorm, GELU / SwiGLU, Absolute / RoPE |
| **Grouped-Query Attention** | Optional GQA/MQA support via `n_kv_head` |
| **Multilingual Training** | Ships with English (Shakespeare) **and** Italian (Dante's Inferno) datasets |
| **Web Dashboard** | Live loss chart, attention heatmap visualizer, and text generator |
| **CLI Training & Inference** | Full command-line pipeline with cosine LR decay and gradient clipping |
| **CSV Loss Logging** | Training losses saved to `out/train_log.csv` for offline plotting |

---

## 📁 Project Structure

```
mini_gpt2/
│
├── model.py          # GPT architecture: attention, MLP, transformer blocks
├── tokenizer.py      # Custom Byte-Pair Encoding (BPE) tokenizer
├── data_loader.py    # Dataset downloader, tokenizer trainer, binary encoder
├── train.py          # CLI training loop with checkpointing + CSV logging
├── generate.py       # CLI text generation from a saved checkpoint
├── plot_loss.py      # Plot training/val loss curve from CSV log
├── app.py            # Flask web dashboard (train + generate + visualize)
│
├── configs/
│   ├── tiny.yaml     # ~1.1M param config  (trains on CPU in ~2 min)
│   ├── mini.yaml     # ~11.5M param config (good GPU target)
│   └── gpt2_full.yaml# 124M param reference config (bonus)
│
├── templates/
│   └── index.html    # Single-page web dashboard UI
│
├── data/             # Downloaded text + tokenizer + binary token files
├── out/              # Checkpoints, train_log.csv, loss_curve.png
│
├── requirements.txt
└── test_app_api.py   # Automated Flask API test suite
```

---

## ⚙️ Installation

```bash
# Clone / navigate to project
cd mini_gpt2

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

**requirements.txt** installs:
- `torch >= 2.0.0`
- `flask >= 3.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0` (for `plot_loss.py`)

---

## 🏗️ Architecture

The model is a **decoder-only transformer** (same family as GPT-2):

```
Input tokens (B, T)
      │
  ┌───▼────────────────────────────────────┐
  │  Token Embedding  (vocab_size × n_embd) │
  │  + Position Embedding (block_size × n_embd)  │   ← Absolute or RoPE
  └───────────────────────────────────────┘
      │
  ┌───▼────────────────────────────────────┐  ×n_layer
  │  Pre-LayerNorm (or RMSNorm)            │
  │  → Causal Self-Attention               │  ← MHA or GQA
  │  → Residual Add                        │
  │  Pre-LayerNorm (or RMSNorm)            │
  │  → Feed-Forward MLP (GELU or SwiGLU)   │
  │  → Residual Add                        │
  └────────────────────────────────────────┘
      │
  Final LayerNorm → LM Head (tied weights)
      │
  Cross-Entropy Loss / Token Probabilities
```

### Parameter Count Comparison

| Config | Layers | Heads | d_model | Parameters |
|--------|--------|-------|---------|------------|
| **Tiny** (default CLI) | 4 | 4 | 128 | **~1.1M** |
| **Mini** (dashboard default) | 6 | 6 | 384 | **~11.5M** |
| **GPT-2 Full** (reference) | 12 | 12 | 768 | **~124M** |

### Architectural Variants (all implemented)

| Dimension | Options |
|---|---|
| Normalization | `LayerNorm` (GPT-2 style) · `RMSNorm` (LLaMA style) |
| Positional Embeddings | `Absolute` (learned) · `RoPE` (Rotary, LLaMA style) |
| MLP Activation | `GELU` (GPT-2 style) · `SwiGLU` (LLaMA style) |
| Attention | `MHA` (standard) · `GQA/MQA` (Grouped/Multi-Query via `n_kv_head`) |

---

## 🚀 Quick Start

### 1. Train the model (CLI)

```bash
# Tiny config — trains fast on CPU
python train.py --dataset shakespeare --n_layer 4 --n_head 4 --n_embd 128 --max_iters 600

# Mini config — better quality, needs a GPU
python train.py --dataset shakespeare --n_layer 6 --n_head 6 --n_embd 384 --max_iters 5000

# Italian (Dante's Inferno) — multilingual bonus!
python train.py --dataset dante --n_layer 4 --n_head 4 --n_embd 128 --max_iters 600

# Modern architecture (RoPE + RMSNorm + SwiGLU)
python train.py --pos_emb_type rope --norm_type rmsnorm --mlp_type swiglu

# Grouped-Query Attention (6 query heads, 2 KV heads)
python train.py --n_head 6 --n_kv_head 2 --n_embd 384
```

Training logs every 50 iterations and saves:
- `out/best_checkpoint.pt` — best model by validation loss
- `out/train_log.csv` — loss history for offline plotting

### 2. Generate text (CLI)

```bash
# Auto-resolves tokenizer from checkpoint metadata
python generate.py --prompt "ROMEO:" --num_samples 3 --max_new_tokens 200

# Italian generation
python generate.py --dataset dante --prompt "Nel mezzo" --temperature 0.9 --top_k 40

# Fine control of sampling
python generate.py --temperature 0.7 --top_k 50 --top_p 0.95 --max_new_tokens 300
```

### 3. Plot the loss curve

```bash
python plot_loss.py
# → saves out/loss_curve.png
```

### 4. Run the web dashboard

```bash
python app.py
# Open http://127.0.0.1:5000
```

The dashboard lets you:
- Configure and launch training live (with loss chart updated in real-time)
- Stop training at any time
- Generate text with a prompt
- Explore the **self-attention heatmap** — see exactly which tokens attend to which

---

## 🌍 Multilingual Training (Bonus)

The project ships with support for **Dante's Inferno** (Italian, 14th century), in addition to Shakespeare. This demonstrates the model's ability to learn non-English language patterns from scratch using the same BPE tokenizer pipeline.

```bash
# Train on Italian text
python train.py --dataset dante --n_layer 4 --n_head 4 --n_embd 128 --max_iters 600

# Generate Italian text
python generate.py --dataset dante --prompt "Nel mezzo del cammin" --temperature 0.85
```

**Sample output** (from a tiny config checkpoint trained on Dante's Inferno):

```
Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura,
ché la diritta via era smarrita.

Ahi quanto a dir qual era è cosa dura
esta selva selvaggia e aspra e forte
che nel pensier rinova la paura!

  Tantè è amara che poco è più morte;
  ma per trattar del ben ch'i' vi trovai,
  dirò de l'altre cose ch'i' v'ho scorte.
```

The model learns Italian meter, accent marks, and terza rima structure purely from data — no language-specific rules.

---

## 🧪 Running Tests

```bash
# First, launch the app in another terminal:
python app.py

# Then run the API test suite:
python test_app_api.py
```

Tests cover: root page render, `/api/status`, `/api/generate` (with attention weight export), `/api/train/start`, and `/api/train/stop`.

---

## 📊 Key Implementation Details

### BPE Tokenizer (`tokenizer.py`)
- Starts from raw UTF-8 bytes (256 base tokens)
- Iteratively merges the most frequent byte pair until `vocab_size` is reached
- Supports save/load to JSON; encode/decode roundtrip is lossless
- Word-level caching for fast repeated encoding

### Training Loop (`train.py`)
- **AdamW** optimizer with decoupled weight decay (2D params only)
- **Cosine annealing** with linear warmup
- **Gradient clipping** at 1.0
- **Memory-mapped** `.bin` files for efficient large dataset batching
- **Mixed-precision** autocast on CUDA

### Attention (`model.py`)
- Manual scaled dot-product with causal mask (tril)
- Attention weights saved per-layer for the web visualizer
- RoPE via precomputed `freqs_cis` applied to Q and K before matmul
- GQA via `repeat_kv()` expanding K/V heads to match Q head count

---

## 📖 References

- Radford et al. (2019) — [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Touvron et al. (2023) — [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Andrej Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT) (primary inspiration)
- Andrej Karpathy — [minBPE](https://github.com/karpathy/minbpe) (tokenizer reference)

---

## 👤 Author

Built from scratch as a hands-on exploration of transformer architectures, tokenization, and language modeling.
