import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256          # Context window length
    n_layer: int = 6               # Number of transformer blocks
    n_head: int = 6                # Number of attention heads
    n_embd: int = 384              # Embedding dimension
    dropout: float = 0.1
    norm_type: str = 'layernorm'    # 'layernorm' or 'rmsnorm'
    pos_emb_type: str = 'absolute'  # 'absolute' or 'rope' (Rotary Position Embeddings)
    mlp_type: str = 'gelu'          # 'gelu' or 'swiglu'
    n_kv_head: int = None          # Number of key/value heads for GQA/MQA. None means n_head (MHA)


# --- Custom Layers ---

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def get_norm(dim, norm_type):
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim)
    elif norm_type == 'rmsnorm':
        return RMSNorm(dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

# --- Helper functions for Rotary Position Embeddings (RoPE) ---

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=None):
    """Precompute cosine and sine frequencies for Rotary Embeddings (RoPE)."""
    # dim must be even
    assert dim % 2 == 0
    # freqs shape: (dim // 2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # t shape: (end)
    t = torch.arange(end, device=device, dtype=torch.float32)
    # freqs shape: (end, dim // 2)
    freqs = torch.outer(t, freqs)
    # Cosine and Sine
    freqs_cos = torch.cos(freqs) # (end, dim // 2)
    freqs_sin = torch.sin(freqs) # (end, dim // 2)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """Apply precomputed cosine/sine frequencies to Q and K."""
    # xq shape: (B, T, n_head, head_dim)
    # xk shape: (B, T, n_head, head_dim)
    # freqs_cos shape: (T, head_dim // 2) -> must become (1, T, 1, head_dim // 2)
    B, T, n_head, head_dim = xq.shape
    
    cos = freqs_cos[:T].view(1, T, 1, head_dim // 2)
    sin = freqs_sin[:T].view(1, T, 1, head_dim // 2)
    
    # Split query/key embeddings into 2 halves
    xq_r, xq_i = xq.chunk(2, dim=-1)
    xk_r, xk_i = xk.chunk(2, dim=-1)
    
    # Rotation transformation: 
    # [xq_r, xq_i] -> [xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos]
    xq_out = torch.cat([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1)
    xk_out = torch.cat([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1)
    
    return xq_out, xk_out

# --- Attention and Feed Forward ---

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key or value tensor along the heads dimension for Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return x
    B, T, n_kv_head, head_dim = x.shape
    return x[:, :, :, None, :].expand(B, T, n_kv_head, n_rep, head_dim).reshape(B, T, n_kv_head * n_rep, head_dim)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.pos_emb_type = config.pos_emb_type
        
        # If n_kv_head is not specified, default to n_head (Multi-Head Attention)
        self.n_kv_head = config.n_kv_head if (config.n_kv_head is not None and config.n_kv_head > 0) else config.n_head
        assert self.n_head % self.n_kv_head == 0, f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        self.num_queries_per_kv = self.n_head // self.n_kv_head
        
        # Combined projection for query, key, value
        # Query: n_embd. Key/Value: n_kv_head * head_dim each.
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.n_kv_head * self.head_dim, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask register (standard GPT causal attention masking)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        
        # Track attention weights for visualization in dashboard
        self.last_attn_weights = None

    def forward(self, x, rope_freqs=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Project and split query, key, values
        q_dim = self.n_embd
        kv_dim = self.n_kv_head * self.head_dim
        q, k, v = self.c_attn(x).split([q_dim, kv_dim, kv_dim], dim=2)
        
        # Shape queries, keys, values into multi-head layouts
        q = q.view(B, T, self.n_head, self.head_dim) # (B, T, nh, hs)
        k = k.view(B, T, self.n_kv_head, self.head_dim) # (B, T, n_kv_h, hs)
        v = v.view(B, T, self.n_kv_head, self.head_dim) # (B, T, n_kv_h, hs)
        
        # Apply Rotary Position Embeddings (RoPE) if requested (operates on K and Q heads before matching)
        if self.pos_emb_type == 'rope':
            assert rope_freqs is not None, "RoPE frequencies must be precomputed and passed"
            cos, sin = rope_freqs
            q, k = apply_rotary_emb(q, k, cos, sin)
            
        # Repeat Key & Value heads if using GQA or MQA to match query head count
        k = repeat_kv(k, self.num_queries_per_kv) # (B, T, nh, hs)
        v = repeat_kv(v, self.num_queries_per_kv) # (B, T, nh, hs)
        
        # Transpose to standard attention layout: (B, nh, T, hs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Manual scaled dot-product attention calculation (extract weights for visualizer)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Mask future steps
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Save attention weights of the first batch element: shape (n_head, T, T)
        self.last_attn_weights = att[0].detach().cpu()
        
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side-by-side
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class GELUMLP(nn.Module):
    """Standard GPT-2 Feed-Forward Network."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class SwiGLUMLP(nn.Module):
    """Modern SwiGLU Feed-Forward Network (LLaMA style)."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        # In LLaMA, the hidden dimension for SwiGLU is typically scaled to 2/3 of 4d, preserving parameters
        self.hidden_dim = int(2 * 4 * config.n_embd / 3)
        self.w1 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)  # Up projection
        self.w3 = nn.Linear(self.hidden_dim, config.n_embd, bias=False)  # Down projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Swish/SiLU active gate computation
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        out = gate * up
        out = self.w3(out)
        return self.dropout(out)

def get_mlp(config: GPTConfig):
    if config.mlp_type == 'gelu':
        return GELUMLP(config)
    elif config.mlp_type == 'swiglu':
        return SwiGLUMLP(config)
    else:
        raise ValueError(f"Unknown mlp_type: {config.mlp_type}")

# --- Transformer Block ---

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = get_norm(config.n_embd, config.norm_type)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = get_norm(config.n_embd, config.norm_type)
        self.mlp = get_mlp(config)

    def forward(self, x, rope_freqs=None):
        # Pre-normalization architecture
        x = x + self.attn(self.ln_1(x), rope_freqs)
        x = x + self.mlp(self.ln_2(x))
        return x

# --- Full GPT Model ---

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final Normalization Layer
            ln_f = get_norm(config.n_embd, config.norm_type),
        ))
        
        # Word token embeddings (wte) are tied to final language model output head mapping
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Tying weights
        
        # Optional absolute position embeddings (GPT-2 default)
        if config.pos_emb_type == 'absolute':
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
            
        # Initialize school weight configs
        self.apply(self._init_weights)
        
        # Apply special scaling parameter initialization to residual projection layers
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w3.weight'):
                # Rescale standard normal deviations: 0.02 / sqrt(2 * layers)
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Support RoPE positional cache on the fly
        if config.pos_emb_type == 'rope':
            # Precompute RoPE sine/cos coordinates up to block_size context length
            # Note we specify head_dim (n_embd // n_head) for RoPE rotations
            self.rope_dim = config.n_embd // config.n_head
        else:
            self.rope_dim = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size limit is {self.config.block_size}"
        
        # 1. Embed tokens
        x = self.transformer.wte(idx) # shape (b, t, n_embd)
        
        # 2. Add position embeddings
        if self.config.pos_emb_type == 'absolute':
            # Absolute position embeddings (GPT-2)
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # shape (t, n_embd)
            x = x + pos_emb
            rope_freqs = None
        else:
            # RoPE positional embedding frequencies (computed on-the-fly and matched to GPU/CPU device)
            cos, sin = precompute_freqs_cis(self.rope_dim, t, device=device)
            rope_freqs = (cos, sin)
            
        # 3. Apply transformer blocks
        for block in self.transformer.h:
            x = block(x, rope_freqs)
            
        # 4. Final normalization
        x = self.transformer.ln_f(x)
        
        # 5. Language model head scoring
        if targets is not None:
            # Loss calculation path (efficient cross entropy logits)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Generation path (only compute logits for final token step for efficiency)
            logits = self.lm_head(x[:, [-1], :]) # shape (b, 1, vocab_size)
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generates sequence autoregressively from seed prompt idx."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop index context window to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward index features
            logits, _ = self(idx_cond)
            
            # Pluck the final step logits and apply temperature scaling
            logits = logits[:, -1, :] / temperature
            
            # Optional Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Optional Top-P (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above cutoff threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep the first token that exceeds the threshold (shifting mask rights)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter indices to remove mask
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
                
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from categorical probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to active context
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


if __name__ == "__main__":
    # Perform unit checks on configurations
    configs_to_test = [
        GPTConfig(pos_emb_type='absolute', norm_type='layernorm', mlp_type='gelu'),
        GPTConfig(pos_emb_type='rope', norm_type='rmsnorm', mlp_type='swiglu'),
        GPTConfig(pos_emb_type='rope', norm_type='rmsnorm', mlp_type='swiglu', n_head=6, n_kv_head=2), # GQA test
    ]
    
    print("Running Model unit checks:")
    for i, cfg in enumerate(configs_to_test):
        print(f"\n--- Testing config {i+1} ---")
        print(cfg)
        model = GPT(cfg)
        
        # Dummy batch feed: shape (B=2, T=10)
        dummy_idx = torch.randint(0, cfg.vocab_size, (2, 10))
        dummy_targets = torch.randint(0, cfg.vocab_size, (2, 10))
        
        print(f"Embedding weights tied: {model.lm_head.weight is model.transformer.wte.weight}")
        
        logits, loss = model(dummy_idx, dummy_targets)
        print("Forward pass successful!")
        print("Logits shape:", logits.shape)
        print("Loss value:", loss.item())
        
        # Test backpropagation
        loss.backward()
        print("Backpropagation successful!")
        
        # Test generation
        gen_out = model.generate(dummy_idx, max_new_tokens=5, temperature=0.8, top_k=20)
        print("Generation successful! Out shape:", gen_out.shape)
        assert gen_out.shape == (2, 15)
        
    print("\nAll model checks passed successfully!")
