"""
Generate golden test data for v2 (nanochat) model components.

Uses the exact same math as nanochat/gpt.py to produce reference inputs/outputs.
All tensors are saved as .npy files for Elixir tests to load and compare.

Run: python scripts/golden_v2/gen_all.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test", "support", "golden_v2")
os.makedirs(OUT_DIR, exist_ok=True)

def save(name, tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().float().numpy()
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), tensor)
    print(f"  saved {name}.npy {tensor.shape} {tensor.dtype}")


# ============================================================================
# 1. RMSNorm
# ============================================================================
print("=== RMSNorm ===")
torch.manual_seed(42)
rmsnorm_input = torch.randn(2, 8, 64, dtype=torch.float32)
rmsnorm_output = F.rms_norm(rmsnorm_input, (rmsnorm_input.size(-1),))
save("rmsnorm_input", rmsnorm_input)
save("rmsnorm_output", rmsnorm_output)


# ============================================================================
# 2. RoPE (Rotary Position Embeddings)
# ============================================================================
print("\n=== RoPE ===")
torch.manual_seed(42)

ROPE_SEQ = 8
ROPE_HEAD_DIM = 16
ROPE_BATCH = 2
ROPE_NHEAD = 4
ROPE_BASE = 10000

half = ROPE_HEAD_DIM // 2
channel_range = torch.arange(0, ROPE_HEAD_DIM, 2, dtype=torch.float32)
inv_freq = 1.0 / (ROPE_BASE ** (channel_range / ROPE_HEAD_DIM))
t = torch.arange(ROPE_SEQ, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
rope_cos = freqs.cos()[None, :, None, :]  # (1, seq, 1, half)
rope_sin = freqs.sin()[None, :, None, :]

rope_input = torch.randn(ROPE_BATCH, ROPE_SEQ, ROPE_NHEAD, ROPE_HEAD_DIM)

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

rope_output = apply_rotary_emb(rope_input, rope_cos, rope_sin)

save("rope_cos", rope_cos)
save("rope_sin", rope_sin)
save("rope_input", rope_input)
save("rope_output", rope_output)


# ============================================================================
# 3. GQA Attention (with QK norm, sliding window)
# ============================================================================
print("\n=== GQA Attention ===")
torch.manual_seed(42)

ATT_BATCH = 2
ATT_SEQ = 8
ATT_NEMBD = 64
ATT_NHEAD = 4
ATT_NKVHEAD = 2
ATT_HEAD_DIM = ATT_NEMBD // ATT_NHEAD  # 16
ATT_WINDOW = 4

att_x = torch.randn(ATT_BATCH, ATT_SEQ, ATT_NEMBD)

# Weights stored in Elixir convention: (in_features, out_features)
# PyTorch Linear(in, out).weight is (out, in), so we transpose for saving
s = (3**0.5) * (ATT_NEMBD**-0.5)
att_c_q_w = torch.empty(ATT_NEMBD, ATT_NHEAD * ATT_HEAD_DIM).uniform_(-s, s)
att_c_k_w = torch.empty(ATT_NEMBD, ATT_NKVHEAD * ATT_HEAD_DIM).uniform_(-s, s)
att_c_v_w = torch.empty(ATT_NEMBD, ATT_NKVHEAD * ATT_HEAD_DIM).uniform_(-s, s)
att_c_proj_w = torch.zeros(ATT_NHEAD * ATT_HEAD_DIM, ATT_NEMBD)

# In Elixir we do: Nx.dot(x, [-1], weight, [0]) => x @ weight
# In PyTorch: F.linear(x, weight) => x @ weight.T where weight is (out, in)
# So our saved weights (in, out) match Nx.dot(x, [-1], w, [0]) directly
# For PyTorch computation here, we use: x @ w (since w is already (in, out))

q = (att_x @ att_c_q_w).view(ATT_BATCH, ATT_SEQ, ATT_NHEAD, ATT_HEAD_DIM)
k = (att_x @ att_c_k_w).view(ATT_BATCH, ATT_SEQ, ATT_NKVHEAD, ATT_HEAD_DIM)
v = (att_x @ att_c_v_w).view(ATT_BATCH, ATT_SEQ, ATT_NKVHEAD, ATT_HEAD_DIM)

# RoPE
half_a = ATT_HEAD_DIM // 2
ch_a = torch.arange(0, ATT_HEAD_DIM, 2, dtype=torch.float32)
inv_a = 1.0 / (10000 ** (ch_a / ATT_HEAD_DIM))
t_a = torch.arange(ATT_SEQ, dtype=torch.float32)
freqs_a = torch.outer(t_a, inv_a)
cos_a = freqs_a.cos()[None, :, None, :]
sin_a = freqs_a.sin()[None, :, None, :]

q = apply_rotary_emb(q, cos_a, sin_a)
k = apply_rotary_emb(k, cos_a, sin_a)

# QK norm
q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))

# GQA: expand K,V heads to match Q heads
repeat = ATT_NHEAD // ATT_NKVHEAD
k_exp = k[:, :, :, None, :].expand(ATT_BATCH, ATT_SEQ, ATT_NKVHEAD, repeat, ATT_HEAD_DIM)
k_exp = k_exp.reshape(ATT_BATCH, ATT_SEQ, ATT_NHEAD, ATT_HEAD_DIM)
v_exp = v[:, :, :, None, :].expand(ATT_BATCH, ATT_SEQ, ATT_NKVHEAD, repeat, ATT_HEAD_DIM)
v_exp = v_exp.reshape(ATT_BATCH, ATT_SEQ, ATT_NHEAD, ATT_HEAD_DIM)

# Standard attention: transpose to (B, H, T, D)
q_t = q.transpose(1, 2)
k_t = k_exp.transpose(1, 2)
v_t = v_exp.transpose(1, 2)

scores = q_t @ k_t.transpose(-2, -1)
# Note: nanochat does NOT divide by sqrt(d_k) because QK norm already normalizes
# Actually looking at the code again: flash_attn handles scaling internally
# But with QK norm, the scale factor is handled differently
# Let me check: nanochat does q, k = norm(q), norm(k) then flash_attn which does /sqrt(d)
# So we DO need the /sqrt(d_k) scaling even with QK norm
scores = scores / (ATT_HEAD_DIM ** 0.5)

# Causal + sliding window mask
i_idx = torch.arange(ATT_SEQ).unsqueeze(1)
j_idx = torch.arange(ATT_SEQ).unsqueeze(0)
causal = j_idx <= i_idx
window = j_idx > (i_idx - ATT_WINDOW)
mask = causal & window

scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
attn_weights = F.softmax(scores, dim=-1)
y = attn_weights @ v_t  # (B, H, T, D)

# Back to (B, T, C) and project
y = y.transpose(1, 2).contiguous().view(ATT_BATCH, ATT_SEQ, -1)
att_output = y @ att_c_proj_w  # (B, T, n_embd)

save("att_input", att_x)
save("att_c_q_w", att_c_q_w)
save("att_c_k_w", att_c_k_w)
save("att_c_v_w", att_c_v_w)
save("att_c_proj_w", att_c_proj_w)
save("att_output", att_output)


# ============================================================================
# 4. MLP (ReLU squared)
# ============================================================================
print("\n=== MLP ===")
torch.manual_seed(42)

MLP_NEMBD = 64
MLP_BATCH = 2
MLP_SEQ = 8

mlp_x = torch.randn(MLP_BATCH, MLP_SEQ, MLP_NEMBD)
mlp_c_fc_w = torch.empty(MLP_NEMBD, 4 * MLP_NEMBD).uniform_(-s, s)
mlp_c_proj_w = torch.zeros(4 * MLP_NEMBD, MLP_NEMBD)

h = mlp_x @ mlp_c_fc_w
h = F.relu(h).square()
mlp_output = h @ mlp_c_proj_w

save("mlp_input", mlp_x)
save("mlp_c_fc_w", mlp_c_fc_w)
save("mlp_c_proj_w", mlp_c_proj_w)
save("mlp_output", mlp_output)


# ============================================================================
# 5. Full v2 forward pass (tiny config)
# ============================================================================
print("\n=== Full Forward ===")
torch.manual_seed(42)

# Tiny config matching our Elixir tiny_config
VOCAB = 256
SEQ_LEN = 16
N_LAYER = 2
N_HEAD = 4
N_KV_HEAD = 2
N_EMBD = 64
HEAD_DIM = N_EMBD // N_HEAD  # 16
KV_DIM = N_KV_HEAD * HEAD_DIM
SOFTCAP = 20.0
WINDOW_PATTERN = "SL"  # short, long alternating
VE_GATE_CHANNELS = 32

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

# Compute window sizes
windows = []
for i in range(N_LAYER):
    c = WINDOW_PATTERN[i % len(WINDOW_PATTERN)]
    if c == 'L':
        windows.append(SEQ_LEN)
    else:
        windows.append(SEQ_LEN // 2)
windows[-1] = SEQ_LEN  # final layer always full

# Init weights (matching nanochat's init_weights exactly)
wte = torch.randn(VOCAB, N_EMBD)  # std=1.0
lm_head_w = torch.randn(VOCAB, N_EMBD) * 0.001  # std=0.001

resid_lambdas = torch.ones(N_LAYER)
x0_lambdas = torch.full((N_LAYER,), 0.1)

s_init = (3**0.5) * (N_EMBD**-0.5)

blocks = []
for layer_i in range(N_LAYER):
    c_q = torch.empty(N_EMBD, N_HEAD * HEAD_DIM).uniform_(-s_init, s_init)
    c_k = torch.empty(N_EMBD, KV_DIM).uniform_(-s_init, s_init)
    c_v = torch.empty(N_EMBD, KV_DIM).uniform_(-s_init, s_init)
    c_proj = torch.zeros(N_HEAD * HEAD_DIM, N_EMBD)
    ve_gate = torch.zeros(VE_GATE_CHANNELS, N_KV_HEAD)
    c_fc = torch.empty(N_EMBD, 4 * N_EMBD).uniform_(-s_init, s_init)
    c_proj_mlp = torch.zeros(4 * N_EMBD, N_EMBD)
    blocks.append({
        'c_q': c_q, 'c_k': c_k, 'c_v': c_v, 'c_proj': c_proj,
        've_gate': ve_gate, 'c_fc': c_fc, 'c_proj_mlp': c_proj_mlp,
    })

value_embeds = {}
for i in range(N_LAYER):
    if has_ve(i, N_LAYER):
        value_embeds[i] = torch.empty(VOCAB, KV_DIM).uniform_(-s_init, s_init)

# RoPE precompute
half_f = HEAD_DIM // 2
ch_f = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
inv_f = 1.0 / (10000 ** (ch_f / HEAD_DIM))
t_f = torch.arange(SEQ_LEN, dtype=torch.float32)
freqs_f = torch.outer(t_f, inv_f)
cos_f = freqs_f.cos()[None, :, None, :]
sin_f = freqs_f.sin()[None, :, None, :]

# Input tokens
idx = torch.randint(0, VOCAB, (2, SEQ_LEN))

# === Forward pass ===
x = wte[idx]  # (B, T, n_embd) -- embedding lookup
x = F.rms_norm(x, (x.size(-1),))  # norm after embed
x0 = x.clone()

for i in range(N_LAYER):
    # Per-layer scaling
    x = resid_lambdas[i] * x + x0_lambdas[i] * x0

    block = blocks[i]
    window_size = windows[i]

    # --- Attention ---
    x_norm = F.rms_norm(x, (x.size(-1),))

    q_f = (x_norm @ block['c_q']).view(2, SEQ_LEN, N_HEAD, HEAD_DIM)
    k_f = (x_norm @ block['c_k']).view(2, SEQ_LEN, N_KV_HEAD, HEAD_DIM)
    v_f = (x_norm @ block['c_v']).view(2, SEQ_LEN, N_KV_HEAD, HEAD_DIM)

    # Value embedding
    if i in value_embeds:
        ve_f = value_embeds[i][idx].view(2, SEQ_LEN, N_KV_HEAD, HEAD_DIM)
        gate_f = 2 * torch.sigmoid(x_norm[..., :VE_GATE_CHANNELS] @ block['ve_gate'])
        v_f = v_f + gate_f.unsqueeze(-1) * ve_f

    # RoPE
    q_f = apply_rotary_emb(q_f, cos_f, sin_f)
    k_f = apply_rotary_emb(k_f, cos_f, sin_f)

    # QK norm
    q_f = F.rms_norm(q_f, (q_f.size(-1),))
    k_f = F.rms_norm(k_f, (k_f.size(-1),))

    # GQA expand
    rep = N_HEAD // N_KV_HEAD
    k_exp_f = k_f[:, :, :, None, :].expand(2, SEQ_LEN, N_KV_HEAD, rep, HEAD_DIM).reshape(2, SEQ_LEN, N_HEAD, HEAD_DIM)
    v_exp_f = v_f[:, :, :, None, :].expand(2, SEQ_LEN, N_KV_HEAD, rep, HEAD_DIM).reshape(2, SEQ_LEN, N_HEAD, HEAD_DIM)

    # Standard attention
    q_t_f = q_f.transpose(1, 2)
    k_t_f = k_exp_f.transpose(1, 2)
    v_t_f = v_exp_f.transpose(1, 2)

    scores_f = (q_t_f @ k_t_f.transpose(-2, -1)) / (HEAD_DIM ** 0.5)

    # Causal + window mask
    i_f = torch.arange(SEQ_LEN).unsqueeze(1)
    j_f = torch.arange(SEQ_LEN).unsqueeze(0)
    mask_f = (j_f <= i_f) & (j_f > (i_f - window_size))
    scores_f = scores_f.masked_fill(~mask_f.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_f = F.softmax(scores_f, dim=-1)
    y_f = (attn_f @ v_t_f).transpose(1, 2).contiguous().view(2, SEQ_LEN, -1)
    attn_out = y_f @ block['c_proj']

    x = x + attn_out

    # --- MLP ---
    x_norm2 = F.rms_norm(x, (x.size(-1),))
    h_f = x_norm2 @ block['c_fc']
    h_f = F.relu(h_f).square()
    mlp_out = h_f @ block['c_proj_mlp']

    x = x + mlp_out

# Final norm + lm_head + softcap
x = F.rms_norm(x, (x.size(-1),))
logits = x @ lm_head_w.T  # lm_head: (vocab, n_embd), so x @ lm_head.T = (B,T,vocab)
logits = SOFTCAP * torch.tanh(logits / SOFTCAP)

save("fwd_idx", idx)
save("fwd_wte", wte)
save("fwd_lm_head_w", lm_head_w)
save("fwd_resid_lambdas", resid_lambdas)
save("fwd_x0_lambdas", x0_lambdas)
save("fwd_logits", logits)

# Save all block weights
for i, block in enumerate(blocks):
    for k, v in block.items():
        save(f"fwd_block{i}_{k}", v)

# Save value embeddings
for i, ve in value_embeds.items():
    save(f"fwd_ve_{i}", ve)

# Save window sizes for reference
np.save(os.path.join(OUT_DIR, "fwd_windows.npy"), np.array(windows, dtype=np.int64))

print(f"\nAll golden data saved to {OUT_DIR}")
