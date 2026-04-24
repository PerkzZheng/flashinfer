# TRTLLM-Gen HeadDim 256 Cubins

This branch (`perkzz/head-dim-256-nvfp4-separate-qkv`) expects a local TRTLLM-Gen FMHA cubin package that contains the headDim 256 cubins exported from `trtllm-gen`, including the NVFP4 KV separate-QKV prefill/decode kernels, the per-sequence padded V-SF layout support, and the CGA-reduction decode variants.

## Package Contents

The package layout is:

```text
flashinfer-headDim256-trtllm-gen-fmha-cubins-c12e684c/
  README.md
  SHA256SUMS
  flashinfer_cubin.py
  cubins/
    c12e684c0027f8803d237a1291a87332008c272faf77bed034f33df32b40fcc3/
      fmha/trtllm-gen/
        checksums.txt
        include/flashInferMetaInfo.h
        *.cubin
```

The `fmha/trtllm-gen` directory contains `2383` cubins. The package-level `SHA256SUMS` file verifies the README, helper module, metadata, checksums, and every cubin in the package. The package tarball is:

```text
/workspace/debug2/flashinfer-headDim256-trtllm-gen-fmha-cubins-c12e684c.tar.gz
sha256: f191932f573eb1ba139630df1006032ef0678803683b1e3e21f8ec8896be6e89
```

For separate-QKV NVFP4 headDim 256, K scaling factors remain logical `[sum_seq_kv, num_kv_heads, head_dim / 16]`. V scaling factors use a separate physical prefix domain: each request is padded independently to a multiple of 4 tokens in V-SF storage, while logical `seq_lens` stay unchanged. The FlashInfer Python wrapper accepts logical linear V scales `[sum_seq_kv, num_kv_heads, head_dim / 16]`, padded linear V scales `[sum_seq_sfs_v, num_kv_heads, head_dim / 16]`, or interleaved V scales `[sum_seq_sfs_v / 4, num_kv_heads, (head_dim / 16) * 4]` and computes the matching `cum_seq_lens_sfs_v` tensor internally.

Paged NVFP4 decode via `trtllm_batch_decode_with_kv_cache` is different: it uses page-local V-scale swizzling from `nvfp4_quantize_paged_kv_cache` and does not use `cum_seq_lens_sfs_v`.

The same `trtllm_batch_decode_with_kv_cache` API now also accepts separate-QKV decode input as `kv_cache=(k, v)` with `k` and `v` shaped `[sum_seq_kv, num_kv_heads, head_dim]`. That route forwards into the separate-QKV decode kernels, ignores `block_tables`, derives `cum_seq_lens_kv` from `seq_lens`, and does use `cum_seq_lens_sfs_v` for headDim 256 NVFP4 V scales.

## Use With This Branch

Clone and check out the FlashInfer branch:

```bash
git clone https://github.com/PerkzZheng/flashinfer.git
cd flashinfer
git switch perkzz/head-dim-256-nvfp4-separate-qkv
git submodule update --init --recursive
```

Extract the cubin package and point FlashInfer at it:

```bash
tar -xzf /path/to/flashinfer-headDim256-trtllm-gen-fmha-cubins-c12e684c.tar.gz -C /path/to/workdir
export CUBIN_PKG=/path/to/workdir/flashinfer-headDim256-trtllm-gen-fmha-cubins-c12e684c

export PYTHONPATH="$CUBIN_PKG:$PWD:$PYTHONPATH"
export FLASHINFER_CUBIN_DIR="$CUBIN_PKG/cubins"
export FLASHINFER_NO_DOWNLOAD=1
```

`flashinfer_cubin.py` is included so this package wins over an installed `flashinfer-cubin` wheel when `CUBIN_PKG` is first in `PYTHONPATH`. `FLASHINFER_CUBIN_DIR` is also set for environments without an installed `flashinfer-cubin` module. `FLASHINFER_NO_DOWNLOAD=1` makes missing or mismatched artifacts fail locally instead of falling back to the public cubin repository.

Verify that FlashInfer sees the packaged cubins:

```bash
python - <<'PY'
import flashinfer_cubin
from flashinfer.jit import env

print("flashinfer_cubin:", flashinfer_cubin.get_cubin_dir())
print("jit cubin dir:", env.FLASHINFER_CUBIN_DIR)
PY
```

The two paths should point to `$CUBIN_PKG/cubins`.

## Verify The Package

From the extracted package directory:

```bash
sha256sum -c SHA256SUMS
find cubins/c12e684c0027f8803d237a1291a87332008c272faf77bed034f33df32b40fcc3/fmha/trtllm-gen -name '*.cubin' | wc -l
```

The cubin count should be `2383`.

## Focused Test

With the environment variables above set from the FlashInfer repo root:

```bash
pytest -q tests/attention/test_trtllm_gen_attention.py::test_trtllm_separate_qkv_nvfp4_head_dim_256 -s
python -m pytest -q tests/attention/test_trtllm_gen_attention.py -k "test_trtllm_batch_decode_with_kv_cache_separate_kv_nvfp4_head_dim_256" -s
python -m pytest -q tests/attention/test_trtllm_gen_attention.py -k "test_trtllm_batch_decode_with_kv_cache_separate_qkv_nvfp4_head_dim_256" -s
python -m pytest -q tests/attention/test_trtllm_gen_attention.py -k "test_trtllm_batch_decode_with_kv_cache_separate_qkv_prepares_cum_seq_lens_sfs_v" -s
python -m pytest -q tests/attention/test_trtllm_gen_attention.py -k "test_trtllm_separate_qkv_nvfp4_decode_prepares_cum_seq_lens_sfs_v" -s
```

The paged decode test covers separate-K/V NVFP4 headDim 256 paged decode with logical per-request `seq_lens` that are not multiples of 4. The separate-QKV decode tests cover both direct `trtllm_ragged_attention_deepseek` usage and the new `trtllm_batch_decode_with_kv_cache` separate-QKV route. They verify logical, padded-linear, and interleaved V-scale inputs and assert that `cum_seq_lens_sfs_v` is prepared and passed to the separate-QKV decode kernels.

Broader headDim 256 coverage used while preparing this branch:

```bash
pytest -q tests/attention/test_trtllm_gen_attention.py::test_trtllm_batch_decode_head_dim_256 -x --tb=short -s
pytest -q tests/attention/test_trtllm_gen_attention.py::test_trtllm_batch_prefill_bs1 -k '256' -x --tb=short -s
```
