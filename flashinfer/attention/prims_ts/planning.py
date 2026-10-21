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

"""Shared attention plan defaults, workspace bounds and output bindings."""

import functools
import inspect
import math
import numbers
import struct
from typing import Any, Callable, Literal, cast

import torch

from flashinfer.utils import (
    _check_workspace_buffer_alignment,
    check_shape_dtype_device,
    get_device_sm_count,
    round_up,
)
from ._hardware import get_max_active_clusters


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


def set_constructor_plan_options(wrapper: Any, options: dict[str, Any]) -> None:
    """Store optional model defaults without allocating execution resources."""
    if options:
        inspect.signature(wrapper.plan).bind_partial(**options)
    wrapper._constructor_plan_options = options


def get_constructor_plan_options(wrapper: Any) -> dict[str, Any]:
    """Resolve model defaults for an optional pre-plan workspace query."""
    bound = inspect.signature(wrapper.plan).bind_partial(
        **wrapper._constructor_plan_options
    )
    bound.apply_defaults()
    return bound.arguments


def constructor_plan_defaults(plan: Callable) -> Callable:
    """Merge constructor defaults with explicitly supplied plan arguments.

    Bind only explicitly supplied plan arguments before merging. This keeps
    explicit False/None/default-valued overrides meaningful, and leaves the
    original signature, API logging and tracing metadata available to tooling.
    """
    signature = inspect.signature(plan)

    @functools.wraps(plan)
    def with_defaults(self, *args, **kwargs):
        options = self._constructor_plan_options
        if not options:
            return plan(self, *args, **kwargs)
        explicit = signature.bind_partial(self, *args, **kwargs)
        explicit.arguments.pop("self")
        return plan(self, **{**options, **explicit.arguments})

    return with_defaults


@functools.cache
def _device_service_slots(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    with torch.cuda.device(index):
        return max(
            get_device_sm_count(torch.device("cuda", index)),
            get_max_active_clusters(1),
            2 * get_max_active_clusters(2),
        )


def decode_control_workspace_bytes(device: torch.device) -> int:
    """Reserve a fixed control tail disjoint from every partial-result view."""
    # 4 = up to two policy service waves times two for reducer-capacity rounding.
    # 128 = maximum query/head rows per work tile (a conservative counter bound).
    # Each counter is int32 (4 bytes). The final 64 bytes hold two 32-byte-aligned
    # dummy metadata slots. All arena sections use 256-byte alignment.
    return round_up(_device_service_slots(device) * 4 * 128 * 4 + 64, 256)


def reduction_workspace_bound(
    device: torch.device, head_dim: int, *, mla: bool = False
) -> int:
    """Bound automatic split scratch by resident physical query/head work.

    Ordinary fanout fits one service wave. Jointly scored FMHA recipes may
    cross its boundary once; rounding reducer capacities adds at most another
    factor of two. Each physical work tile stores at most 128 query/head rows.
    Output-sized statistics and forced-split formats are checked separately.
    """
    service_slots = _device_service_slots(device)
    # MLA 1CTA uses M<=64 and at most 2x capacity rounding. Its 2CTA path
    # uses M128 with half as many resident clusters and the same rounding.
    work_tiles = service_slots * (1 if mla else 4)
    partial_rows = work_tiles * 128
    stats_bytes = 4 if mla else 8
    size = partial_rows * (head_dim * 2 + stats_bytes) + work_tiles * 4 + 256
    if not mla:
        size += decode_control_workspace_bytes(device)
    return round_up(size, 256)


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
