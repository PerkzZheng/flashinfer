# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Storage geometry for sparse MLA caches, independent of the FP8 MMA type."""


def cache_record_size(cache_format: str, rope_dim: int) -> int:
    """Elements per token; quantized formats use uint8 elements."""
    return {
        "native": 512 + rope_dim,
        "nvfp4": 288,
        "nvfp4.fp8rope": 352,
        "mxfp8": 528,
    }[cache_format]


class ScaledKVConfig:
    """Common format predicates for both attention families."""

    @property
    def use_scaled_kv(self) -> bool:
        return self.kv_cache_format != "native"

    @property
    def has_mxfp8(self) -> bool:
        return "mxfp8" in (self.kv_cache_format, self.extra_kv_cache_format)

    @property
    def max_cache_record_size(self) -> int:
        return max(
            cache_record_size(self.kv_cache_format, self.rope_dim),
            cache_record_size(self.extra_kv_cache_format, self.rope_dim),
        )

    @property
    def raw_record_stride(self) -> int:
        # gather4 requires every four-row destination to be 128-byte aligned.
        return (self.max_cache_record_size + 31) // 32 * 32
