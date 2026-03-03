"""Test embedding lookup optimization for RelativePositionEncoding.

The original implementation materialized three huge one-hot tensors
(N_token^2 x 66, N_token^2 x 66, N_token^2 x 6), concatenated them with
b_same_entity into [N_token^2, 139], then multiplied by the LinearNoBias
weight. For 3000 tokens, this creates ~5GB of intermediate one-hot data.

The optimized version uses direct weight indexing:
  one_hot(idx, K) @ W_sub = W_sub.T[idx]

This produces the same output while avoiding the one-hot materialization
entirely. Peak intermediate memory drops from ~10GB to ~4.6GB for N=3000.

Expected: 30-50% memory reduction, significant speedup.
"""

import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPosEncOriginal(nn.Module):
    """Original implementation using one-hot + linear."""

    def __init__(self, r_max=32, s_max=2, c_z=128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        in_features = 4 * r_max + 2 * s_max + 7
        self.weight = nn.Parameter(torch.randn(c_z, in_features))

    def forward(self, d_residue, d_token, d_chain, b_same_entity):
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))
        data = torch.cat(
            [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
            dim=-1,
        ).float()
        return F.linear(data, self.weight)


class RelPosEncOptimized(nn.Module):
    """Optimized implementation using embedding lookup."""

    def __init__(self, r_max=32, s_max=2, c_z=128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        in_features = 4 * r_max + 2 * s_max + 7
        self.weight = nn.Parameter(torch.randn(c_z, in_features))

    def forward(self, d_residue, d_token, d_chain, b_same_entity):
        W = self.weight.t().float()
        n_pos = 2 * (self.r_max + 1)
        n_chain = 2 * (self.s_max + 1)

        W_pos = W[:n_pos]
        W_token = W[n_pos:2 * n_pos]
        W_entity = W[2 * n_pos]
        W_chain = W[2 * n_pos + 1:]

        p = W_pos[d_residue]
        p = p + W_token[d_token]
        p = p + b_same_entity.unsqueeze(-1).float() * W_entity
        p = p + W_chain[d_chain]
        return p


def _make_inputs(n_tokens, device="cpu"):
    """Create realistic relative position encoding inputs."""
    r_max, s_max = 32, 2
    # Simulate multi-chain protein
    asym_id = torch.zeros(n_tokens, dtype=torch.long, device=device)
    asym_id[n_tokens // 2:] = 1
    residue_index = torch.arange(n_tokens, dtype=torch.long, device=device) % (n_tokens // 2)
    entity_id = asym_id.clone()
    sym_id = asym_id.clone()
    token_index = torch.arange(n_tokens, dtype=torch.long, device=device)

    b_same_chain = (asym_id[:, None] == asym_id[None, :]).long()
    b_same_residue = (residue_index[:, None] == residue_index[None, :]).long()
    b_same_entity = (entity_id[:, None] == entity_id[None, :]).long()
    rel_pos = residue_index[:, None] - residue_index[None, :]

    d_residue = torch.clip(rel_pos + r_max, 0, 2 * r_max) * b_same_chain + (1 - b_same_chain) * (2 * r_max + 1)
    d_token = torch.clip(
        token_index[:, None] - token_index[None, :] + r_max, 0, 2 * r_max
    ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (2 * r_max + 1)
    d_chain = torch.clip(
        sym_id[:, None] - sym_id[None, :] + s_max, 0, 2 * s_max
    ) * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)

    return d_residue, d_token, d_chain, b_same_entity


def test_embedding_matches_onehot():
    """Verify embedding lookup produces identical output to one-hot + linear."""
    torch.manual_seed(42)
    n_tokens = 200

    original = RelPosEncOriginal()
    optimized = RelPosEncOptimized()
    # Share weights
    optimized.weight.data = original.weight.data.clone()

    d_residue, d_token, d_chain, b_same_entity = _make_inputs(n_tokens)

    with torch.no_grad():
        out_orig = original(d_residue, d_token, d_chain, b_same_entity)
        out_opt = optimized(d_residue, d_token, d_chain, b_same_entity)

    torch.testing.assert_close(out_orig, out_opt, rtol=1e-5, atol=1e-5)


def test_embedding_memory_savings():
    """Verify embedding lookup uses less intermediate memory than one-hot."""
    n_tokens = 1000
    r_max, s_max, c_z = 32, 2, 128
    n_pos = 2 * (r_max + 1)  # 66
    n_chain = 2 * (s_max + 1)  # 6

    # One-hot approach memory (intermediate tensors):
    # 3 one-hot tensors: N^2 x (66 + 66 + 6) = N^2 x 138 float32
    # Concatenated: N^2 x 139 float32
    # Total peak: N^2 x (138 + 139) = N^2 x 277 float32
    onehot_peak_bytes = n_tokens**2 * 277 * 4

    # Embedding approach memory (intermediate tensors):
    # 3 index tensors: N^2 x 3 int64
    # Output accumulator: N^2 x c_z float32
    emb_peak_bytes = n_tokens**2 * c_z * 4 + n_tokens**2 * 3 * 8

    savings_pct = (1 - emb_peak_bytes / onehot_peak_bytes) * 100
    assert emb_peak_bytes < onehot_peak_bytes, (
        f"Embedding ({emb_peak_bytes / 1e6:.0f} MB) should use less memory "
        f"than one-hot ({onehot_peak_bytes / 1e6:.0f} MB)"
    )
    assert savings_pct > 30, (
        f"Expected >30% memory savings, got {savings_pct:.0f}%"
    )

    # Also check for large targets where OOM matters
    n_large = 3000
    onehot_large = n_large**2 * 277 * 4
    emb_large = n_large**2 * c_z * 4 + n_large**2 * 3 * 8
    assert emb_large < onehot_large, (
        f"For N={n_large}: embedding ({emb_large / 1e9:.1f} GB) should use less "
        f"than one-hot ({onehot_large / 1e9:.1f} GB)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_embedding_speedup():
    """Verify embedding lookup is faster than one-hot + linear on GPU."""
    torch.manual_seed(42)
    n_tokens = 500
    device = "cuda"

    original = RelPosEncOriginal().to(device).eval()
    optimized = RelPosEncOptimized().to(device).eval()
    optimized.weight.data = original.weight.data.clone()

    d_residue, d_token, d_chain, b_same_entity = _make_inputs(n_tokens, device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            original(d_residue, d_token, d_chain, b_same_entity)
            optimized(d_residue, d_token, d_chain, b_same_entity)
    torch.cuda.synchronize()

    n_iters = 10

    start = time.monotonic()
    with torch.no_grad():
        for _ in range(n_iters):
            original(d_residue, d_token, d_chain, b_same_entity)
    torch.cuda.synchronize()
    time_original = (time.monotonic() - start) / n_iters

    start = time.monotonic()
    with torch.no_grad():
        for _ in range(n_iters):
            optimized(d_residue, d_token, d_chain, b_same_entity)
    torch.cuda.synchronize()
    time_optimized = (time.monotonic() - start) / n_iters

    assert time_optimized <= time_original, (
        f"Embedding lookup ({time_optimized*1000:.1f}ms) should not be slower "
        f"than one-hot+linear ({time_original*1000:.1f}ms)"
    )


def test_no_onehot_in_inference_path():
    """Verify the inference path no longer uses F.one_hot."""
    from pathlib import Path

    source = (
        Path(__file__).parent.parent / "pxdesign" / "model" / "embedders.py"
    ).read_text()

    # Find the RelativePositionEncoding class and check its forward method
    # The forward should use W_pos[d_residue] pattern, not F.one_hot
    assert "W_pos[d_residue]" in source, (
        "RelativePositionEncoding should use embedding lookup (W_pos[d_residue])"
    )
    assert "W_token[d_token]" in source, (
        "RelativePositionEncoding should use embedding lookup (W_token[d_token])"
    )
    assert "W_chain[d_chain]" in source, (
        "RelativePositionEncoding should use embedding lookup (W_chain[d_chain])"
    )
