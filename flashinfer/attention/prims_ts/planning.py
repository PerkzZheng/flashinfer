# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared attention LSE/sink bindings and scale/layout checks."""

import math
import numbers
import struct
from typing import Literal, cast

import torch

from flashinfer.utils import (
    _check_workspace_buffer_alignment,
    check_shape_dtype_device,
)


def _validate_scale(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive Python scalar")
    try:
        value_as_float = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive Python scalar") from error
    if not math.isfinite(value_as_float) or value_as_float <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        value_as_float32 = struct.unpack("=f", struct.pack("=f", value_as_float))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            f"{name} must be representable as a positive float32"
        ) from error
    if not math.isfinite(value_as_float32) or value_as_float32 <= 0.0:
        raise ValueError(f"{name} must be representable as a positive float32")
    return value_as_float32


def _validate_kv_layout(kv_layout: str) -> None:
    if not isinstance(kv_layout, str):
        raise TypeError("kv_layout must be a string")
    if kv_layout not in ("HND", "NHD"):
        raise ValueError(f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}")


def validate_lse_mode(mode: str) -> Literal["none", "base2"]:
    if mode not in ("none", "base2"):
        raise ValueError("lse_mode must be 'none' or 'base2'")
    return cast(Literal["none", "base2"], mode)


def prepare_lse(
    lse: torch.Tensor | None,
    *,
    q: torch.Tensor,
    mode: str,
    return_lse: bool,
    validate: bool,
) -> torch.Tensor | None:
    """Bind an independent caller buffer, or allocate the enabled output."""
    if mode == "none":
        if return_lse or lse is not None:
            raise ValueError("plan() must enable lse_mode='base2'")
        return None
    if lse is None:
        return torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    if validate:
        check_shape_dtype_device(lse, q.shape[:-1], torch.float32, q.device, "lse")
        if not lse.is_contiguous():
            raise ValueError("lse must be contiguous")
        _check_workspace_buffer_alignment(lse, "lse", alignment=4)
    return lse


def validate_attention_sinks(sinks, *, q, enabled, validate):
    """Bind sink logits; callers guarantee finite values or negative infinity."""
    if enabled != (sinks is not None):
        raise ValueError(
            "attention_sinks must be supplied exactly when plan(use_attention_sinks=True)"
        )
    if sinks is not None and validate:
        check_shape_dtype_device(
            sinks, (q.shape[-2],), torch.float32, q.device, "attention_sinks"
        )
        if not sinks.is_contiguous():
            raise ValueError("attention_sinks must be contiguous")
        _check_workspace_buffer_alignment(sinks, "attention_sinks", alignment=4)
    return sinks
