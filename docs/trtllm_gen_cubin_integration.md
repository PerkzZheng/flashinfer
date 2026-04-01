# Integrating trtllm-gen Cubins into FlashInfer Locally

This guide covers how to build cubins from trtllm-gen and wire them into a local
FlashInfer checkout for development and testing, bypassing the published artifact
registry.

## Prerequisites

| Requirement | Notes |
|---|---|
| trtllm-gen repo | `perkzz/contiguous-kv` branch (or whichever has your kernel changes) |
| FlashInfer repo | editable install (`pip install --no-build-isolation -e .`) |
| CUDA 12.9+ | Required for SM100/SM100f compilation |
| Python 3.10+ | For ExportCubin.py |

---

## Step 1 — Build trtllm-gen

```bash
cd /path/to/trtllm-gen
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

---

## Step 2 — Generate Cubins with ExportCubin.py

`ExportCubin.py` compiles kernels and writes `.cubin` files plus a
`flashInferMetaInfo.h` header into an output directory.

```bash
cd /path/to/trtllm-gen

# Choose an output directory (can be anything; we'll copy from here next).
EXPORT_DIR=/tmp/trtllm-gen-export

python kernels/Fmha/tools/ExportCubin.py \
    --flashInfer \
    --smVer sm100f \
    $EXPORT_DIR
```

**Key flags:**

| Flag | Description |
|---|---|
| `--flashInfer` | Enables FlashInfer-specific export mode (generates `flashInferMetaInfo.h`) |
| `--smVer` | Target SM version: `sm100f` (Blackwell B100/H100NVL), `sm103a` (GB200), `sm107a` (Rubin) |
| `--filter REGEX` | Limit to a subset of kernels, e.g. `--filter ContiguousKv` |
| `output_dir` | Where to write the generated files (default: current directory) |

> **Note:** ExportCubin.py calls `scripts/run_on_amodel.sh` to download CUDA
> artifacts needed for compilation. This requires VPN / network access to NVIDIA
> internal infra. If it fails, the script will print a warning and skip that SM
> version (the try/except added in `perkzz/contiguous-kv`).

After the run you should see files like:

```
$EXPORT_DIR/
  FmhaSm100fKernel_QkvFp16OFp16H128ContiguousKv...Context.cubin
  FmhaSm100fKernel_QkvFp16OFp16H128ContiguousKv...KeepsAbForGen.cubin
  ...
  include/
    flashInferMetaInfo.h
```

---

## Step 3 — Set Up the Local Cubin Directory

FlashInfer loads cubins from `FLASHINFER_CUBIN_DIR`, which defaults to a
system path but can be overridden with the environment variable.

```bash
# Pick a local directory to act as the cubin store.
CUBIN_DIR=/tmp/flashinfer-cubins

# This must match ArtifactPath.TRTLLM_GEN_FMHA in flashinfer/artifacts.py.
ARTIFACT_PATH="a049237e8fa659619d0362bac025177d2f6f9c3e/fmha/trtllm-gen"

DEST=$CUBIN_DIR/$ARTIFACT_PATH
mkdir -p $DEST/include
```

### 3a — Copy and rename cubins

The kernel loader looks up cubins by `mFuncName` from `flashInferMetaInfo.h`,
which starts with a **lowercase** `f` (`fmhaSm100f...`). ExportCubin.py emits
files with an **uppercase** `F` (`FmhaSm100f...`). Rename on copy:

```bash
for f in $EXPORT_DIR/*.cubin; do
    base=$(basename "$f")
    newname=$(echo "$base" | sed 's/^F/f/')   # uppercase F -> lowercase f
    cp "$f" "$DEST/$newname"
done
```

### 3b — Copy the meta-info header

```bash
cp $EXPORT_DIR/include/flashInferMetaInfo.h $DEST/include/flashInferMetaInfo.h
```

### 3c — Generate checksums.txt

```bash
(
  cd $DEST
  # Hash every cubin and the header; format: "<sha256>  <filename>"
  sha256sum *.cubin include/flashInferMetaInfo.h > checksums.txt
)
```

---

## Step 4 — Update `flashinfer/artifacts.py`

FlashInfer verifies `checksums.txt` against a hardcoded hash in
`CheckSumHash.TRTLLM_GEN_FMHA`. After regenerating cubins you must update it.

```bash
sha256sum $DEST/checksums.txt
# Example output:
# 18f7399b4d06d75155fef63aad1c0a21ab80f8ec2d8ddce5b5fb2919c53dabe6  checksums.txt
```

Open `flashinfer/artifacts.py` and update the value:

```python
class CheckSumHash:
    TRTLLM_GEN_FMHA: str = (
        "18f7399b4d06d75155fef63aad1c0a21ab80f8ec2d8ddce5b5fb2919c53dabe6"  # <-- new hash
    )
```

> If `ArtifactPath.TRTLLM_GEN_FMHA` is also changing (e.g. a new artifact
> publish path), update that string too and make sure your `$ARTIFACT_PATH`
> above matches it.

---

## Step 5 — Install FlashInfer (editable)

If you haven't already:

```bash
cd /path/to/flashinfer
pip install --no-build-isolation -e . -v
```

---

## Step 6 — Run Tests

Always pass `FLASHINFER_CUBIN_DIR` so FlashInfer uses your local cubins instead
of trying to download from the artifact registry:

```bash
FLASHINFER_CUBIN_DIR=/tmp/flashinfer-cubins \
    python -m pytest tests/attention/test_trtllm_contiguous_kv_attention.py -v
```

Expected output (30 tests, headDim × dtype × scenario):

```
tests/attention/test_trtllm_contiguous_kv_attention.py::test_contiguous_kv_decode_scenario1[dtype0-32] PASSED
...
30 passed in 0.91s
```

---

## Troubleshooting

### Checksum mismatch

```
RuntimeError: Failed to download cubins: checksum mismatch
```

`CheckSumHash.TRTLLM_GEN_FMHA` does not match the `checksums.txt` you generated.
Re-run `sha256sum $DEST/checksums.txt` and update `artifacts.py`.

### Missing kernel error

```
RuntimeError: Missing TRTLLM-GEN kernel ContiguousKv (context): ...
```

The requested kernel combination (dtype, headDim, maskType, etc.) was not
compiled. Check that `ExportCubin.py`'s `skipKernelGen` function does not filter
out that combination, then re-export and redo steps 3–4.

### Cubin not found / SHA256 mismatch at load time

The cubin filename must match `mFuncName` in `flashInferMetaInfo.h`. Verify the
rename in step 3a actually produced a lowercase-`f` filename:

```bash
ls $DEST/*.cubin | head -3
# should print: fmhaSm100f...cubin  (not FmhaSm100f...)
```

### `FLASHINFER_CUBIN_CHECKSUM_DISABLED`

For quick iteration you can skip SHA256 verification at load time:

```bash
FLASHINFER_CUBIN_CHECKSUM_DISABLED=1 \
FLASHINFER_CUBIN_DIR=/tmp/flashinfer-cubins \
    python -m pytest tests/attention/test_trtllm_contiguous_kv_attention.py -v
```

Do **not** rely on this in CI or production.

---

## Quick-Reference Script

```bash
#!/usr/bin/env bash
# integrate_cubins.sh — run from the trtllm-gen root after building

set -euo pipefail

TRTLLM_GEN_ROOT=$(pwd)
FLASHINFER_ROOT=/path/to/flashinfer
EXPORT_DIR=/tmp/trtllm-gen-export
CUBIN_DIR=/tmp/flashinfer-cubins
ARTIFACT_PATH="a049237e8fa659619d0362bac025177d2f6f9c3e/fmha/trtllm-gen"

# 1. Generate cubins
python kernels/Fmha/tools/ExportCubin.py --flashInfer --smVer sm100f "$EXPORT_DIR"

# 2. Set up destination
DEST="$CUBIN_DIR/$ARTIFACT_PATH"
mkdir -p "$DEST/include"

# 3. Copy + rename cubins (uppercase F -> lowercase f)
for f in "$EXPORT_DIR"/*.cubin; do
    base=$(basename "$f")
    cp "$f" "$DEST/$(echo "$base" | sed 's/^F/f/')"
done

# 4. Copy header
cp "$EXPORT_DIR/include/flashInferMetaInfo.h" "$DEST/include/"

# 5. Generate checksums
(cd "$DEST" && sha256sum *.cubin include/flashInferMetaInfo.h > checksums.txt)

# 6. Print the hash to paste into artifacts.py
echo ""
echo "==> Paste this into CheckSumHash.TRTLLM_GEN_FMHA in flashinfer/artifacts.py:"
sha256sum "$DEST/checksums.txt" | awk '{print $1}'
```
