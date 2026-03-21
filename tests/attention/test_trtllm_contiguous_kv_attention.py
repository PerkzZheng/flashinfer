"""Tests for trtllm-gen ContiguousKv prefill/decode attention.

Scenarios (all with batch_size=1, BNHD layout):
  Scenario 1 – causal=True
    Decode:  q=[1,1,32,128],  k=v=[1,1,4,128],   cache=[1,28600,4,128]
    Prefill: q=[1,72,32,128], k=v=[1,72,4,128],  cache=[1,28600,4,128]

  Scenario 2 – causal=False
    Prefill: q=k=v=[1,764,4,128], cache=[1,315860,4,128]
"""

import math

import pytest
import torch

from flashinfer.utils import get_compute_capability

DEVICE = "cuda:0"
WORKSPACE_SIZE = 256 * 1024 * 1024

# Skip entire module on unsupported GPUs (needs SM 100 / Blackwell for trtllm-gen ContiguousKv).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device(DEVICE))[0] < 10,
    reason="trtllm-gen ContiguousKv kernels require SM >= 100 (Blackwell)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_workspace(device=DEVICE) -> torch.Tensor:
    """Zero-initialised workspace required by trtllm-gen kernels."""
    return torch.zeros(WORKSPACE_SIZE, dtype=torch.uint8, device=device)


def sm_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def reference_attention(
    q: torch.Tensor,  # [B, S_q, H_q, D]
    k: torch.Tensor,  # [B, S_kv, H_kv, D]
    v: torch.Tensor,  # [B, S_kv, H_kv, D]
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Naive PyTorch reference for multi-head attention (supports GQA).

    The causal mask is derived from the shapes: past_kv_len = S_kv - S_q.
    Query token i is at global position (past_kv_len + i) and may attend to
    KV tokens 0 .. past_kv_len + i.  For a fresh prefill S_kv == S_q so
    past_kv_len == 0 (standard causal).  For decode S_q == 1 and
    past_kv_len == S_kv - 1, allowing the query to see all cached tokens.
    """
    B, S_q, H_q, D = q.shape
    S_kv = k.shape[1]
    H_kv = k.shape[2]
    groups = H_q // H_kv
    past_kv_len = S_kv - S_q

    q_f = q.float()  # [B, S_q, H_q, D]
    k_f = k.float().repeat_interleave(groups, dim=2)  # [B, S_kv, H_q, D]
    v_f = v.float().repeat_interleave(groups, dim=2)

    # [B, H_q, S_q, S_kv]
    scores = torch.einsum("bqhd,bkhd->bhqk", q_f, k_f) * scale

    if is_causal:
        # Query token i is at global position (past_kv_len + i).
        # Causal rule: mask if k_pos > past_kv_len + q_i.
        q_positions = torch.arange(S_q, device=q.device) + past_kv_len  # [S_q]
        k_positions = torch.arange(S_kv, device=q.device)  # [S_kv]
        causal_mask = k_positions[None, :] > q_positions[:, None]  # [S_q, S_kv]
        scores.masked_fill_(causal_mask[None, None], float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", weights, v_f)  # [B, S_q, H_q, D]
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Scenario 1 – Decode (causal, q_len=1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_contiguous_kv_decode_scenario1(dtype):
    """Scenario 1 decode: q=[1,1,32,128], cache=[1,28600,4,128], causal=True."""
    B, S_q, H_q, H_kv, D = 1, 1, 32, 4, 128
    cache_len = 100  # cached tokens already present
    max_cache = 315860

    torch.manual_seed(0)
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    # Seed the first cache_len tokens.
    k_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )
    v_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )

    seq_lens = torch.tensor([cache_len], dtype=torch.int32, device=DEVICE)
    workspace = make_workspace()
    scale = sm_scale(D)

    from flashinfer.prefill import trtllm_contiguous_kv_attention_decode

    # Flat query [B*S_q, H_q, D]
    q_flat = q.flatten(0, 1)
    out = trtllm_contiguous_kv_attention_decode(
        query=q_flat,
        key_cache=k_cache,
        value_cache=v_cache,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=S_q,
        max_kv_len=cache_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=B,
    )
    assert out.shape == q_flat.shape, f"Expected {q_flat.shape}, got {out.shape}"
    assert not out.isnan().any(), "Output contains NaN"
    assert not out.isinf().any(), "Output contains Inf"

    # Reference: S_kv=cache_len, S_q=1 → past_kv_len=cache_len-1; query sees all cached tokens.
    ref = reference_attention(
        q,
        k_cache[:, :cache_len],
        v_cache[:, :cache_len],
        scale=scale,
        is_causal=True,
    )  # [B, S_q, H_q, D]
    ref_flat = ref.flatten(0, 1)

    torch.testing.assert_close(out.float(), ref_flat.float(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Scenario 1 – Prefill (causal, q_len=72)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_contiguous_kv_context_scenario1(dtype):
    """Scenario 1 context: q=[1,72,32,128], cache=[1,28600,4,128], causal=True."""
    B, S_q, H_q, H_kv, D = 1, 72, 32, 4, 128
    cache_len = S_q  # fresh prefill: kv length equals query length
    max_cache = 28600

    torch.manual_seed(1)
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    k_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )
    v_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )

    seq_lens = torch.tensor([cache_len], dtype=torch.int32, device=DEVICE)
    cum_seq_lens_q = torch.tensor([0, S_q], dtype=torch.int32, device=DEVICE)
    cum_seq_lens_kv = torch.tensor([0, cache_len], dtype=torch.int32, device=DEVICE)
    workspace = make_workspace()
    scale = sm_scale(D)

    from flashinfer.prefill import trtllm_contiguous_kv_attention_context

    q_flat = q.flatten(0, 1)  # [B*S_q, H_q, D]
    out = trtllm_contiguous_kv_attention_context(
        query=q_flat,
        key_cache=k_cache,
        value_cache=v_cache,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=S_q,
        max_kv_len=cache_len,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=B,
        is_causal=True,
    )
    assert out.shape == q_flat.shape, f"Expected {q_flat.shape}, got {out.shape}"
    assert not out.isnan().any(), "Output contains NaN"
    assert not out.isinf().any(), "Output contains Inf"

    ref = reference_attention(
        q,
        k_cache[:, :cache_len],
        v_cache[:, :cache_len],
        scale=scale,
        is_causal=True,
    )
    ref_flat = ref.flatten(0, 1)
    torch.testing.assert_close(out.float(), ref_flat.float(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Scenario 2 – Prefill non-causal (causal=False, q_len=764)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_contiguous_kv_context_scenario2_noncausal(dtype):
    """Scenario 2 context: q=k=v=[1,764,4,128], cache=[1,315860,4,128], causal=False."""
    B, S_q, H_q, H_kv, D = 1, 764, 4, 4, 128
    cache_len = S_q
    max_cache = 315860

    torch.manual_seed(2)
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    k_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )
    v_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )

    seq_lens = torch.tensor([cache_len], dtype=torch.int32, device=DEVICE)
    cum_seq_lens_q = torch.tensor([0, S_q], dtype=torch.int32, device=DEVICE)
    cum_seq_lens_kv = torch.tensor([0, cache_len], dtype=torch.int32, device=DEVICE)
    workspace = make_workspace()
    scale = sm_scale(D)

    from flashinfer.prefill import trtllm_contiguous_kv_attention_context

    q_flat = q.flatten(0, 1)
    out = trtllm_contiguous_kv_attention_context(
        query=q_flat,
        key_cache=k_cache,
        value_cache=v_cache,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=S_q,
        max_kv_len=cache_len,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=B,
        is_causal=False,
    )
    assert out.shape == q_flat.shape, f"Expected {q_flat.shape}, got {out.shape}"
    assert not out.isnan().any(), "Output contains NaN"
    assert not out.isinf().any(), "Output contains Inf"

    ref = reference_attention(
        q,
        k_cache[:, :cache_len],
        v_cache[:, :cache_len],
        scale=scale,
        is_causal=False,
    )
    ref_flat = ref.flatten(0, 1)
    torch.testing.assert_close(out.float(), ref_flat.float(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Scenario 1 – Decode with existing cache (cache_len > S_q)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_contiguous_kv_decode_with_history(dtype):
    """Decode where the cache already holds tokens from a prior prefill."""
    B, S_q, H_q, H_kv, D = 1, 1, 32, 4, 128
    history_len = 72  # tokens from previous prefill
    max_cache = 28600

    torch.manual_seed(3)
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=DEVICE)
    k_cache = torch.zeros(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    v_cache = torch.zeros(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    k_cache[:, :history_len] = torch.randn(
        B, history_len, H_kv, D, dtype=dtype, device=DEVICE
    )
    v_cache[:, :history_len] = torch.randn(
        B, history_len, H_kv, D, dtype=dtype, device=DEVICE
    )

    seq_lens = torch.tensor([history_len], dtype=torch.int32, device=DEVICE)
    workspace = make_workspace()
    scale = sm_scale(D)

    from flashinfer.prefill import trtllm_contiguous_kv_attention_decode

    q_flat = q.flatten(0, 1)
    out = trtllm_contiguous_kv_attention_decode(
        query=q_flat,
        key_cache=k_cache,
        value_cache=v_cache,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=S_q,
        max_kv_len=history_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=B,
    )
    assert out.shape == q_flat.shape
    assert not out.isnan().any()

    # S_kv=history_len, S_q=1 → past_kv_len=history_len-1; query sees all history tokens.
    ref = reference_attention(
        q,
        k_cache[:, :history_len],
        v_cache[:, :history_len],
        scale=scale,
        is_causal=True,
    )
    ref_flat = ref.flatten(0, 1)
    torch.testing.assert_close(out.float(), ref_flat.float(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# MultiCtasKv context: long seqLenKv triggers GmemReductionWithSeparateKernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_contiguous_kv_context_multi_ctas_kv(dtype):
    """Context with large seqLenKv to exercise multiCtasKv (GmemReductionWithSeparateKernel)."""
    B, S_q, H_q, H_kv, D = 1, 764, 4, 1, 128
    cache_len = S_q
    max_cache = 20000

    torch.manual_seed(4)
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    k_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )
    v_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )

    seq_lens = torch.tensor([cache_len], dtype=torch.int32, device=DEVICE)
    cum_seq_lens_q = torch.tensor([0, S_q], dtype=torch.int32, device=DEVICE)
    cum_seq_lens_kv = torch.tensor([0, cache_len], dtype=torch.int32, device=DEVICE)
    workspace = make_workspace()
    scale = sm_scale(D)

    from flashinfer.prefill import trtllm_contiguous_kv_attention_context

    q_flat = q.flatten(0, 1)
    out = trtllm_contiguous_kv_attention_context(
        query=q_flat,
        key_cache=k_cache,
        value_cache=v_cache,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=S_q,
        max_kv_len=cache_len,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=B,
        is_causal=True,
    )
    assert out.shape == q_flat.shape
    assert not out.isnan().any()

    ref = reference_attention(
        q, k_cache[:, :cache_len], v_cache[:, :cache_len], scale=scale, is_causal=True
    )
    torch.testing.assert_close(
        out.float(), ref.flatten(0, 1).float(), rtol=1e-2, atol=1e-2
    )


# ---------------------------------------------------------------------------
# HeadDim=32 decode — ContiguousKv KeepsMmaAbForGeneration kernels support D=32
# (Context + ContiguousKv only compiled for headDim=128)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "ContiguousKv headDim=32 cubins are not yet in the flashinfer cubin registry. "
        "The kernels exist in trtllm-gen (KeepsMmaAbForGeneration) but must be published "
        "to the cubin distribution before this test can pass."
    ),
    strict=True,
)
def test_contiguous_kv_decode_headdim32():
    """Decode with headDim=32; KeepsMmaAbForGeneration ContiguousKv kernels support D=32."""
    B, S_q, H_q, H_kv, D = 1, 1, 4, 4, 32
    cache_len = 64
    max_cache = 4096

    torch.manual_seed(5)
    dtype = torch.float16
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(B, max_cache, H_kv, D, dtype=dtype, device=DEVICE)
    k_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )
    v_cache[:, :cache_len] = torch.randn(
        B, cache_len, H_kv, D, dtype=dtype, device=DEVICE
    )

    seq_lens = torch.tensor([cache_len], dtype=torch.int32, device=DEVICE)
    workspace = make_workspace()
    scale = sm_scale(D)

    from flashinfer.prefill import trtllm_contiguous_kv_attention_decode

    q_flat = q.flatten(0, 1)
    out = trtllm_contiguous_kv_attention_decode(
        query=q_flat,
        key_cache=k_cache,
        value_cache=v_cache,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=S_q,
        max_kv_len=cache_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=B,
    )
    assert out.shape == q_flat.shape
    assert not out.isnan().any()
    assert not out.isinf().any()

    ref = reference_attention(
        q, k_cache[:, :cache_len], v_cache[:, :cache_len], scale=scale, is_causal=True
    )
    torch.testing.assert_close(
        out.float(), ref.flatten(0, 1).float(), rtol=1e-2, atol=1e-2
    )
