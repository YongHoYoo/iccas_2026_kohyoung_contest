"""
V2 FOV Inspection Order Planner — Attention Model with Beam Search

Architecture Overview (Encoder-Decoder Attention Model for Combinatorial Optimization):

    ┌─────────────────────────────────────────────────────────────────────┐
    │  ENCODER (Edge-Aware Graph Attention Network)                      │
    │                                                                     │
    │  Input: points_5d (N,5) = [norm_x, norm_y, img_t, recon_t, comp_t]│
    │         t_move    (N,N) = pairwise movement time (edge features)   │
    │                                                                     │
    │  Layer 0: Linear Embedding (5 → 128)                               │
    │  Layer 1: Edge-Aware Multi-Head Attention (8 heads, d_k=16)        │
    │           - Q,K,V projections on node embeddings                   │
    │           - E projection on edge features (movement time)          │
    │           - attn_out = Σ(attn_weight * V) + Σ(attn_weight * E)    │
    │           → Skip Connection + LayerNorm + FFN + Skip + LayerNorm   │
    │  Layer 2: Standard MHA (8 heads) + FFN + LayerNorm                 │
    │  Layer 3: Standard MHA (8 heads) + FFN + LayerNorm                 │
    │  Output:  h = final_layer_out + layer1_out  (residual, (N, 128))   │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  DECODER (Autoregressive Pointer Network with Thread-Aware PE)     │
    │                                                                     │
    │  At each decoding step t:                                          │
    │    Context: h_current  = h[prev_node] + PE[thread_assignment]      │
    │             h_average  = mean of unvisited node embeddings + PE    │
    │             h_thread   = mean PE of inspection threads             │
    │             (h_last    = h[depot] for closed-loop model)           │
    │                                                                     │
    │    MHA:     q = Linear([h_current; h_average; h_thread])           │
    │             k,v = Linear(h_context + PE[thread])                   │
    │             attn → weighted sum → project + h_average → LayerNorm  │
    │                                                                     │
    │    Pointer: logit = (q3 · k2) / √d  with visited-node masking     │
    │             prob  = softmax(logit)                                  │
    │                                                                     │
    │  Output: probability distribution over next FOV to visit           │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  BEAM SEARCH with Thread Pipeline Simulation                       │
    │                                                                     │
    │  - Maintains top-k (48) candidate sequences                        │
    │  - Each candidate tracks full pipeline state:                      │
    │      ref[1]           : Thread 0 timeline (camera move + capture)  │
    │      recon[2]         : Thread 1-2 timeline (3D reconstruction)    │
    │      comp[n_thread]   : Thread 3+ timeline (component inspection)  │
    │  - Thread assignment: greedy — assign to earliest-available thread │
    │  - Step group mask (z_table): enforces same-step-height grouping   │
    │  - Final ranking by simulated CT = max(all thread completion)      │
    └─────────────────────────────────────────────────────────────────────┘

Training (REINFORCE with Greedy Rollout Baseline):
    - Policy: π_θ(a_t | s_t) = decoder probability at step t
    - Reward: R = -CT (negative cycle time, minimize)
    - Baseline: greedy rollout (argmax at each step) for variance reduction
    - Loss: L = E[(R - R_baseline) * Σ log π_θ(a_t | s_t)]
    - Data: randomly generated FOV instances via generator.py
    - Curriculum: increasing sequence length during training

Positional Encoding (Thread-Aware):
    PE encodes both FOV-type (recon thread 0/1) and inspection-thread index.
    Shape: (n_positions*2, embed_dim) — interleaved [fov_type, thread_pos] pairs.
    This lets the model learn thread-assignment-dependent orderings.
"""

import numpy as np
import math

try:
    from ky_solver._cy_math import cy_layer_norm as _fast_layer_norm_impl
except ImportError:
    _fast_layer_norm_impl = None

import zlib as _zlib
import io as _io


_V2_WEIGHTS = {}


def _load_weights(model_type):
    if model_type not in _V2_WEIGHTS:
        from ky_solver._weights import get_weights
        _V2_WEIGHTS[model_type] = get_weights(model_type)
    return _V2_WEIGHTS[model_type]


def _layer_norm(x, weight, bias, eps=1e-5):
    """Standard Layer Normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _calculate_movement_time(points_2d, v_x, v_y, a_x, a_y):
    """Pairwise FOV-to-FOV movement time matrix using trapezoidal velocity profile.
    Each axis moves independently; total time = max(T_x, T_y).
    Returns: (N, N) float32 matrix.
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    # X-axis: trapezoidal profile  (threshold = v²/a)
    x_dist = np.abs(x[:, None] - x[None, :])
    x_th = v_x * v_x / a_x
    x_over = 2 * v_x / a_x + (x_dist - x_th) / v_x      # long distance (trapezoid)
    x_under = 2 * np.sqrt(x_dist / a_x)                    # short distance (triangle)
    x_time = np.where(x_dist >= x_th, x_over, x_under)

    # Y-axis: same profile
    y_dist = np.abs(y[:, None] - y[None, :])
    y_th = v_y * v_y / a_y
    y_over = 2 * v_y / a_y + (y_dist - y_th) / v_y
    y_under = 2 * np.sqrt(y_dist / a_y)
    y_time = np.where(y_dist >= y_th, y_over, y_under)

    return np.maximum(x_time, y_time).astype(np.float32)


def _compute_pe(n_positions, embed_dim):
    """Thread-aware positional encoding.
    Encodes two components:
      1. FOV-type PE    — which reconstruction thread (0 or 1) this FOV is assigned to
      2. Position PE    — which inspection thread index (sinusoidal)
    Interleaved: [fov_type_0, pos_0, fov_type_1, pos_1, ...] → (n_positions*2, embed_dim)
    """
    half = embed_dim // 2

    def sin_pe(n, dim):
        pe = np.zeros((n, half), dtype=np.float32)
        log_10000 = math.log(10000.0)
        for pos in range(n):
            for i in range(0, half, 2):
                div = math.exp(-log_10000 / half * i)
                angle = pos * div
                pe[pos, i] = math.sin(angle)
                pe[pos, i + 1] = math.cos(angle)
        return pe

    # FOV-type encoding: binary (recon thread 0 vs 1)
    pe_fov = np.zeros((2, half), dtype=np.float32)
    pe_fov[0] = sin_pe(2, embed_dim)[0]
    pe_fov[1] = 1.0 - pe_fov[0]

    # Position encoding: sinusoidal for inspection thread index
    pe_pos = sin_pe(n_positions, embed_dim)

    # Interleave: [fov_type, thread_pos] pairs
    pe_comp = np.zeros((n_positions * 2, half), dtype=np.float32)
    pe_fov_exp = np.zeros((n_positions * 2, half), dtype=np.float32)
    for i in range(n_positions):
        pe_comp[i * 2] = pe_pos[i]
        pe_comp[i * 2 + 1] = pe_pos[i]
        pe_fov_exp[i * 2] = pe_fov[0]
        pe_fov_exp[i * 2 + 1] = pe_fov[1]

    return np.concatenate([pe_fov_exp, pe_comp], axis=1)


def _mha_layer(h, Q, K, V, W_flat, norm_w, norm_b, ff0_w, ff0_b, ff2_w, ff2_b, norm1_w, norm1_b):
    """Standard Transformer block: MultiHeadAttention + FFN, both with skip connections and LayerNorm.
    Pre-norm variant: Skip → MHA → Norm → Skip → FFN → Norm
    """
    n_head = Q.shape[0]
    seq, dim = h.shape
    k_dim = Q.shape[2]
    norm_factor = 1.0 / math.sqrt(k_dim)

    # Multi-Head Attention
    q = np.einsum('sd,hdk->hsk', h, Q)   # (n_head, seq, k_dim)
    k = np.einsum('sd,hdk->hsk', h, K)
    v = np.einsum('sd,hdk->hsk', h, V)

    qk = norm_factor * np.einsum('hik,hjk->hij', q, k)   # scaled dot-product
    attn = _softmax(qk, axis=-1)
    head = np.einsum('hij,hjk->hik', attn, v)

    head_flat = head.transpose(1, 0, 2).reshape(seq, -1)  # concat heads
    out = head_flat @ W_flat                                # output projection

    # Skip + Norm
    h1 = _layer_norm(h + out, norm_w, norm_b)

    # Feed-Forward Network (ReLU activation)
    ff_out = np.maximum(0, h1 @ ff0_w + ff0_b) @ ff2_w + ff2_b

    # Skip + Norm
    h2 = _layer_norm(h1 + ff_out, norm1_w, norm1_b)
    return h2


def v2_encode(points_5d, t_move, w):
    """Encoder: Edge-Aware Graph Attention Network.
    Produces node embeddings that capture both node features and pairwise movement costs.

    Args:
        points_5d: (seq, 5) — [norm_x, norm_y, imaging_time, recon_time, comp_time]
        t_move:    (seq, seq) — pairwise movement time matrix (edge features)
        w:         weight dict from .npz

    Returns:
        h: (seq, 128) — node embeddings with residual connection from layer 1
    """
    seq = points_5d.shape[0]
    n_head = 8
    k_dim = 16
    norm_factor = 1.0 / math.sqrt(k_dim)

    # Initial linear embedding: (seq, 5) → (seq, 128)
    embed = points_5d @ w['enc_init_embed_w']

    # === Edge-Aware Attention Layer (Layer 0) ===
    # Edge projection: movement time → per-head edge features
    edge_flat = t_move.reshape(-1, 1).astype(np.float32)
    e = np.einsum('bi,hik->bhk', edge_flat, w['enc_E'])                 # (seq*seq, 8, 16)
    e = e.reshape(seq, seq, n_head, k_dim).transpose(2, 0, 1, 3)        # (8, seq, seq, 16)

    # Standard Q,K,V from node embeddings
    q = np.einsum('sd,hdk->hsk', embed, w['enc_Q'])
    k = np.einsum('sd,hdk->hsk', embed, w['enc_K'])
    v = np.einsum('sd,hdk->hsk', embed, w['enc_V'])

    # Attention weights (node-to-node)
    qk = norm_factor * np.einsum('hik,hjk->hij', q, k)
    attn = _softmax(qk, axis=-1)

    # Aggregation: value + edge features weighted by attention
    head_v = np.einsum('hij,hjk->hik', attn, v)            # standard value aggregation
    head_e = (attn[:, :, :, None] * e).sum(axis=2)          # edge-aware aggregation

    head = head_v + head_e
    head_flat = head.transpose(1, 0, 2).reshape(seq, -1)
    out = head_flat @ w['enc_W']                             # output projection

    # FFN + Residual + LayerNorm (non-standard order for this layer)
    out1 = _layer_norm(out, w['enc_norm1_w'], w['enc_norm1_b'])
    out2 = np.maximum(0, out1 @ w['enc_ff0_w'] + w['enc_ff0_b']) @ w['enc_ff2_w'] + w['enc_ff2_b']
    out2 = out2 + out
    h_ = _layer_norm(out2, w['enc_norm2_w'], w['enc_norm2_b'])

    # === Standard MHA Layers (Layer 1, 2) ===
    h = _mha_layer(h_,
                   w['enc_l0_Q'], w['enc_l0_K'], w['enc_l0_V'], w['enc_l0_W'],
                   w['enc_l0_norm0_w'], w['enc_l0_norm0_b'],
                   w['enc_l0_ff0_w'], w['enc_l0_ff0_b'], w['enc_l0_ff2_w'], w['enc_l0_ff2_b'],
                   w['enc_l0_norm1_w'], w['enc_l0_norm1_b'])

    h = _mha_layer(h,
                   w['enc_l1_Q'], w['enc_l1_K'], w['enc_l1_V'], w['enc_l1_W'],
                   w['enc_l1_norm0_w'], w['enc_l1_norm0_b'],
                   w['enc_l1_ff0_w'], w['enc_l1_ff0_b'], w['enc_l1_ff2_w'], w['enc_l1_ff2_b'],
                   w['enc_l1_norm1_w'], w['enc_l1_norm1_b'])

    # Long residual connection: final output + edge-aware layer output
    return (h + h_).astype(np.float32)


def v2_decode_step(h_current, h_context, h_thread, mask, w, h_last=None):
    """Decoder: single autoregressive step (pointer network).
    Computes probability distribution over next FOV to visit.

    Args:
        h_current: (batch, 128) — embedding of last visited FOV + thread PE
        h_context: (batch, seq, 128) — all FOV embeddings
        h_thread:  (batch, 128) — mean PE of inspection threads (pipeline state)
        mask:      (batch, seq) — 0=unvisited, 1=visited (masked out)
        h_last:    (batch, 128) — depot embedding for closed-loop model, None for open

    Returns:
        prob: (batch, seq) — next-node selection probability
    """
    batch, seq, dim = h_context.shape
    n_head = 8
    k_dim = 16
    norm_factor = 1.0 / math.sqrt(k_dim)

    # Context K,V projections (shared across steps in beam search)
    h_flat = h_context.reshape(-1, dim)
    k = np.einsum('sd,hdk->hsk', h_flat, w['dec_K']).reshape(n_head, batch, seq, k_dim)
    v = np.einsum('sd,hdk->hsk', h_flat, w['dec_V']).reshape(n_head, batch, seq, k_dim)
    k2 = (h_flat @ w['dec_K2']).reshape(batch, seq, dim)   # second-stage key for pointer

    # Graph embedding: mean of unvisited node embeddings
    mask_expand = (1 - mask)[:, :, None]
    h_enable = h_context * mask_expand
    n_visible = np.maximum((1 - mask).sum(axis=1, keepdims=True), 1.0)
    h_average = h_enable.sum(axis=1) / n_visible

    mask_inf = np.where(mask == 1, -np.inf, 0.0).astype(np.float32)

    # Query construction: concatenate current state features
    if h_last is not None:
        h_concat = np.concatenate([h_current, h_average, h_thread, h_last], axis=1)
    else:
        h_concat = np.concatenate([h_current, h_average, h_thread], axis=1)

    # === Stage 1: Multi-Head Attention over context ===
    q = np.einsum('bd,hdk->hbk', h_concat, w['dec_Q'])

    qk = norm_factor * np.einsum('hbk,hbsk->hbs', q, k)
    qk = qk + mask_inf[None, :, :]
    attn = _softmax(qk.reshape(-1, batch, seq), axis=-1).reshape(n_head, batch, 1, seq)

    q2 = np.einsum('hbis,hbsk->hbik', attn, v)
    q2_flat = q2.transpose(1, 2, 0, 3).reshape(batch, -1)

    # Project + residual (graph embedding) + LayerNorm
    q3 = q2_flat @ w['dec_project_w'] + h_average
    q3 = _layer_norm(q3, w['dec_norm_w'], w['dec_norm_b'])
    q3 = q3[:, None, :]

    # === Stage 2: Pointer attention (single-head, produces selection logits) ===
    dim_norm = 1.0 / math.sqrt(dim)
    qk2 = dim_norm * np.einsum('bid,bjd->bij', q3, k2)
    qk2 = qk2.squeeze(1) + mask_inf
    prob = _softmax(qk2, axis=-1)

    return prob.astype(np.float32)


def _build_single_priority_table(group):
    """Step-height group constraint table.
    Forces beam search to visit FOVs of the same step-height consecutively,
    minimizing Z-axis transition penalties (+5s each).
    Returns: (N, N) adjacency mask — table[i,j]=1 if FOV j has the same step as FOV i.
    """
    n = len(group)
    table = np.zeros((n, n), dtype=np.float32)
    groups = {}
    for i in range(n):
        g = int(group[i])
        if g >= 0:
            if g not in groups:
                groups[g] = []
            groups[g].append(i)
    for i in range(n):
        g = int(group[i])
        if g >= 0:
            for j in groups[g]:
                table[i, j] = 1.0
    return table


def _fast_layer_norm(x, w, b):
    """Cython-accelerated LayerNorm if available, else pure numpy fallback."""
    if _fast_layer_norm_impl is not None:
        return _fast_layer_norm_impl(np.ascontiguousarray(x, dtype=np.float32), w, b)
    return _layer_norm(x, w, b)


def v2_beam_search(fov_centers, fov_imaging, fov_recon, fov_comp, fov_steps,
                   v_x, v_y, a_x, a_y, model_type='open', topk=48):
    """Top-level beam search: encode FOVs, then decode with thread-aware beam search.

    Args:
        fov_centers:  (N, 2) — FOV center coordinates (mm)
        fov_imaging:  (N,) — per-FOV imaging time (capture + side_capture if applicable)
        fov_recon:    (N,) — per-FOV 3D reconstruction time
        fov_comp:     (N,) — per-FOV max component inspection time
        fov_steps:    (N,) int — step-height group index per FOV
        v_x, v_y, a_x, a_y: camera motion parameters
        model_type:   'open' (no return) or 'closed' (return to depot)
        topk:         beam width

    Returns:
        list of candidate orderings, ranked by estimated CT (best first)
    """
    w = _load_weights(model_type)
    is_closed = (model_type == 'closed')
    seqlen = fov_centers.shape[0]

    if seqlen <= 2:
        return [np.arange(seqlen)]

    n_a_thread = min(16, max(8, seqlen // 8))
    embed_dim = 128

    # Normalize FOV coordinates to [0, 1] range for encoder input
    max_vals = fov_centers.max(axis=0)
    min_vals = fov_centers.min(axis=0)
    center = (max_vals + min_vals) / 2.0
    length = max(max_vals - min_vals)
    if length < 1e-6:
        length = 1.0
    fov_norm = (fov_centers - center) / length + 0.5

    # Encoder input: [norm_x, norm_y, imaging_time, recon_time, comp_time]
    points_5d = np.column_stack([fov_norm, fov_imaging, fov_recon, fov_comp]).astype(np.float32)
    t_move = _calculate_movement_time(fov_centers.astype(np.float32), v_x, v_y, a_x, a_y)
    h_base = v2_encode(points_5d, t_move, w)
    pe = _compute_pe(32, embed_dim)
    z_table = _build_single_priority_table(fov_steps)
    h_last_base = h_base[seqlen - 1:seqlen] if is_closed else None

    result, _ = _beam_search_core(
        h_base, points_5d, t_move, pe, z_table, w,
        is_closed, seqlen, n_a_thread, topk, h_last_base)
    return result


def _beam_search_core(h_base, points_5d, t_move, pe, z_table, w,
                      is_closed, seqlen, n_a_thread, topk, h_last_base):
    """Core beam search loop with inline thread pipeline simulation.

    Pipeline model (simulated per beam candidate):
        Thread 0          : camera move + capture (sequential)
        Thread 1-2        : 3D reconstruction (2 recon threads, round-robin)
        Thread 3..n_thread: component inspection (greedy earliest-available)

    The thread assignment for each FOV is encoded as:
        tc = recon_thread_idx + inspection_thread_idx * 2
    This composite index is used to look up thread-aware PE during decoding.
    """
    embed_dim = 128
    nf = 1.0 / math.sqrt(16)       # MHA scale factor (d_k=16)
    dn = 1.0 / math.sqrt(embed_dim) # pointer attention scale
    B = 1                            # initial beam width (grows to topk)

    logits = np.zeros((B, seqlen), dtype=np.int32)     # selected node indices per step
    threads = np.zeros((B, seqlen), dtype=np.int32)     # thread assignment per step

    # Visitation mask: 1=unvisited, 0=visited
    seq_mask = np.ones((B, seqlen), dtype=np.float32)
    seq_mask[:, 0] = 0.0
    if is_closed:
        seq_mask[:, seqlen - 1] = 0.0

    # Step-height group mask: restricts next-node selection to same-step FOVs
    z_mask = z_table[0:1].copy()

    # === Thread pipeline state ===
    ref = np.zeros((B, 1), dtype=np.float32)               # T0: camera timeline
    recon = np.zeros((B, 2), dtype=np.float32)              # T1,T2: reconstruction timelines
    comp = np.zeros((B, n_a_thread), dtype=np.float32)      # T3+: inspection timelines

    # Average thread PE (represents pipeline state embedding)
    pe_thr = pe[:(n_a_thread - 3) * 2].mean(axis=0) if n_a_thread > 3 else pe[:2].mean(axis=0)
    t_img = points_5d[:, 2]    # per-FOV imaging time
    t_rec = points_5d[:, 3]    # per-FOV reconstruction time
    t_cmp = points_5d[:, 4]    # per-FOV inspection time

    # Pre-compute decoder K,V projections for base embeddings and PE
    kb = np.einsum('sd,hdk->hsk', h_base, w['dec_K'])
    vb = np.einsum('sd,hdk->hsk', h_base, w['dec_V'])
    pk = np.einsum('pd,hdk->hpk', pe, w['dec_K'])
    pv = np.einsum('pd,hdk->hpk', pe, w['dec_V'])
    kb_T = np.ascontiguousarray(kb.transpose(0, 2, 1))
    pk_T = np.ascontiguousarray(pk.transpose(0, 2, 1))
    k2b_T = np.ascontiguousarray((h_base @ w['dec_K2']).T)
    pk2_T = np.ascontiguousarray((pe @ w['dec_K2']).T)

    dQ = w['dec_Q']
    dP = w['dec_project_w']
    dNw = w['dec_norm_w']
    dNb = w['dec_norm_b']

    for step in range(seqlen - 1):
        # Apply step-height group constraint (prefer same-step FOVs)
        z_m = z_mask * seq_mask
        zero = z_m.sum(axis=1) < 1e-6
        if zero.any():
            z_m[zero] = 1.0    # fallback: allow all if no same-step candidates

        # Last step handling: exclude depot for open model
        if step == seqlen - 2:
            sm = seq_mask.copy()
            sm[:, seqlen - 1] = 1.0
        else:
            sm = seq_mask

        enabled = sm * z_m
        dm = 1.0 - enabled     # decoder mask: 1=blocked, 0=available

        # === Thread pipeline simulation (vectorized over all candidates) ===
        if step == 0:
            # Initialize pipeline with first FOV
            n0 = logits[:, 0]
            bi0 = np.arange(B)
            ref[:, 0] += t_img[n0]                                  # T0: capture first FOV
            recon = np.maximum(recon, ref)                           # T1/T2 wait for capture
            ri0 = threads[:, 0] % 2                                 # recon thread (0 or 1)
            recon[bi0, ri0] += t_rec[n0]                             # add recon time
            comp = np.maximum(comp, recon[bi0, ri0][:, None])        # inspection waits for recon
            ci0 = threads[:, 0] // 2                                 # inspection thread index
            comp[bi0, ci0] += t_cmp[n0]                              # add inspection time

        prevs = logits[:, step]
        bi = np.arange(B)[:, None]
        ci = np.arange(seqlen)[None, :]

        # Simulate pipeline for ALL possible next nodes (vectorized)
        ref_a = ref + t_move[prevs] + t_img[None, :]                # T0: move + capture
        recon_a = np.maximum(recon[:, None, :], ref_a[:, :, None])   # T1/T2: wait for T0
        mr = np.argmin(recon_a, axis=2)                              # greedy: earliest recon thread
        recon_a[bi, ci, mr] += t_rec[None, :]
        rv = recon_a[bi, ci, mr]
        comp_a = np.maximum(comp[:, None, :], rv[:, :, None])        # T3+: wait for recon
        mc = np.argmin(comp_a, axis=2)                               # greedy: earliest insp thread
        comp_a[bi, ci, mc] += t_cmp[None, :]
        tc = mr + mc * 2                                             # composite thread index for PE

        # Handle closed model: unmask depot at last step
        if step == seqlen - 2 and is_closed:
            dm_u = dm.copy()
            dm_u[:, seqlen - 1] = 0.0
        else:
            dm_u = dm

        # === Decoder forward pass (inline, fused with PE) ===
        # Graph embedding: mean of unvisited (node embedding + thread PE)
        vis = 1.0 - dm_u
        n_vis = np.maximum(vis.sum(axis=1, keepdims=True), 1.0)
        pe_tc = pe[tc]
        h_avg = (vis @ h_base + np.matmul(vis[:, None, :], pe_tc).squeeze(1)) / n_vis

        mi = np.where(dm_u == 1, -np.inf, 0.0).astype(np.float32)   # attention mask

        # Current node embedding + thread PE
        h_cur = h_base[prevs] + pe[threads[:, step]]
        h_thr = np.tile(pe_thr, (B, 1)).astype(np.float32)
        if is_closed:
            h_cat = np.concatenate([h_cur, h_avg, h_thr, np.tile(h_last_base, (B, 1))], axis=1)
        else:
            h_cat = np.concatenate([h_cur, h_avg, h_thr], axis=1)

        # Stage 1: Multi-Head Attention (fused base + PE keys)
        q = np.matmul(h_cat[None], dQ)
        ba = np.arange(B)
        q_dot_pk = np.matmul(q, pk_T)
        qk = nf * (np.matmul(q, kb_T) + q_dot_pk[:, ba[:, None], tc]) + mi[None]
        attn = _softmax(qk, axis=-1)

        # Attention-weighted values (base + PE)
        q2_base = np.matmul(attn, vb)
        pv_tc = pv[:, tc]
        q2_pe = np.matmul(attn[:, :, None, :], pv_tc).squeeze(2)
        q2 = (q2_base + q2_pe).transpose(1, 0, 2).reshape(B, -1)

        # Project + graph embedding residual + LayerNorm
        q3 = _fast_layer_norm(q2 @ dP + h_avg, dNw, dNb)

        # Stage 2: Pointer logits (fused base + PE keys)
        qk2 = dn * (q3 @ k2b_T + (q3 @ pk2_T)[ba[:, None], tc]) + mi
        prob = _softmax(qk2, axis=-1).astype(np.float32)

        # === Beam expansion: select top-k candidates ===
        pf = prob.reshape(-1)
        tk = min(topk, max(1, int(enabled.sum())))
        if tk >= pf.size:
            ix = np.argsort(-pf)[:tk]
        else:
            ix = np.argpartition(-pf, tk)[:tk]
            ix = ix[np.argsort(-pf[ix])]

        p = ix // seqlen     # parent beam index
        n = ix % seqlen      # selected node index
        tr = np.arange(tk)

        # Update beam state
        logits = logits[p].copy()
        logits[tr, step + 1] = n
        threads = threads[p].copy()
        threads[tr, step + 1] = tc[p, n]

        # Update pipeline state for selected beams
        ref = ref_a[p, n][:, None]
        recon = recon_a[p, n].copy()
        comp = comp_a[p, n].copy()

        # Update masks
        seq_mask = seq_mask[p].copy()
        seq_mask[tr, n] = 0.0
        z_mask = z_table[n]
        B = tk

    # Rank beams by estimated CT (= max thread completion time)
    ct = comp.max(axis=1)
    rank = np.argsort(ct)
    return [logits[rank[i]] for i in range(len(rank))], float(ct[rank[0]])
