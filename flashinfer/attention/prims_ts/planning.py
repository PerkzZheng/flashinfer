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

"""Shared attention planning, runtime bindings and workspace ownership."""

from dataclasses import dataclass, field
from collections import OrderedDict
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
from .policy_cachec import SharedPolicy, model_key, shared_policy_cache


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
    bound = inspect.signature(wrapper.plan).bind(**wrapper._constructor_plan_options)
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


@dataclass
class RuntimePlan:
    """Per-wrapper scratch, warmed bindings and captured resource owners.

    The global policy cache shares only CPU dispatch records. This local cache
    keeps each layer's tensor bindings and captured addresses independent.

    Warmup may grow owned storage for auxiliary metadata or forced-split
    formats. The first capture freezes its address permanently. Native views
    are rebuilt after growth, and cache misses cannot allocate during capture.
    """

    options: dict[str, Any]
    workspace: torch.Tensor | None = None
    owns_workspace: bool = False
    prepared: OrderedDict[tuple, Any] = field(default_factory=OrderedDict)
    captured_keys: set[tuple] = field(default_factory=set)
    control_bytes: int = 0
    initialized_control_pointer: int | None = None
    control_reset_required: bool = False
    family: str = ""
    layout: str | None = None
    empty_i32: torch.Tensor | None = None
    model_key: frozenset = field(init=False)

    def __post_init__(self):
        self.model_key = model_key(self.options)

    def problem(
        self,
        query,
        *,
        max_seq_len_q,
        max_kv_len,
        validate,
        seq_lens=None,
        qo_indptr=None,
        kv_indptr=None,
    ) -> tuple[int, int, int]:
        """Resolve host execution bounds; trusted calls never read CUDA values."""
        packed = self.options.get(
            "packed", self.options.get("packed_query", self.family == "paged_context")
        )
        if packed and qo_indptr is None:
            raise ValueError("packed execution requires qo_indptr")
        batch = (
            int(seq_lens.numel())
            if seq_lens is not None
            else (int(qo_indptr.numel()) - 1 if packed else int(query.shape[0]))
        )
        sq = (
            runtime_query_length_bound(max_seq_len_q)
            if packed
            else (
                1
                if self.family == "decode" and query.ndim == 3
                else int(query.shape[1])
            )
        )
        sk = runtime_length_bound(
            max_kv_len,
            kv_indptr if kv_indptr is not None else seq_lens,
            validate=validate,
            name="max_kv_len",
            cumulative=kv_indptr is not None,
        )
        return batch, sq, sk

    def prepare(
        self,
        key: tuple,
        factory: Callable[[], SharedPolicy],
        bind: Callable[[SharedPolicy, torch.Tensor | None], Any],
        required_bytes: int | Callable[[SharedPolicy], int] = 0,
    ) -> Any:
        device = self.options["device"]
        with torch.cuda.device(device):
            capturing = torch.cuda.is_current_stream_capturing()
        prepared = self.prepared.get(key)
        if prepared is not None:
            if capturing:
                self.captured_keys.add(key)
            self.prepared.move_to_end(key)
            return prepared
        policy_key = (self.family, self.layout, key, self.model_key)
        if capturing:
            # Standalone calls create a temporary wrapper. A warmed shared
            # decode policy can bind caller scratch using only tensor views;
            # it needs no local cache entry or device metadata allocation.
            policy = shared_policy_cache.peek(policy_key)
            if policy is None or self.family not in ("decode", "mla_decode"):
                raise RuntimeError(
                    "warm this attention execution shape with run() before CUDA graph capture"
                )
        else:
            policy = shared_policy_cache.get_or_create(policy_key, factory)
        if callable(required_bytes):
            required_bytes = required_bytes(policy)
        available = 0 if self.workspace is None else self.workspace.numel()
        if required_bytes > available:
            if capturing or self.captured_keys or not self.owns_workspace:
                raise ValueError(
                    f"runtime attention needs {required_bytes} workspace bytes; "
                    f"the plan has {available}. Supply sufficient workspace in "
                    "plan() and warm the execution shapes before graph capture."
                )
            if self.workspace is not None:
                self.workspace.record_stream(torch.cuda.current_stream(device))
            self.workspace = torch.empty(
                max(required_bytes, available * 2),
                dtype=torch.int8,
                device=device,
            )
            self.prepared.clear()
        prepared = bind(policy, self.workspace)
        self.prepared[key] = prepared
        if capturing:
            self.captured_keys.add(key)
        # Captured resource owners remain pinned. Bound eager metadata
        # caching so a growing KV length does not retain one child per token.
        while len(self.prepared) > len(self.captured_keys) + 32:
            oldest = next(k for k in self.prepared if k not in self.captured_keys)
            retired = self.prepared.pop(oldest)
            state = retired
            for name in (
                "empty_i32",
                "variable_window_padded_starts",
                "variable_window_cta_starts",
            ):
                tensor = getattr(state, name, None)
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(torch.cuda.current_stream(device))
        return prepared


def runtime_query_length_bound(value: int | None) -> int:
    """Require a caller-provided packed-Q bound without reading CUDA offsets."""
    if value is None:
        raise ValueError(
            "packed Q requires max_seq_len_q; supply it in run() or plan()"
        )
    return runtime_length_bound(value, None, validate=False, name="max_seq_len_q")


def runtime_length_bound(
    value: int | None,
    metadata: torch.Tensor | None,
    *,
    validate: bool,
    name: str,
    cumulative: bool = False,
) -> int:
    """Resolve a host length bound, reading live metadata only during validation."""
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value
    if not validate:
        raise ValueError(f"validate=False requires runtime {name}")
    if metadata is None:
        raise ValueError(f"execution requires {name} or length metadata")
    lengths = (
        metadata[1:].to(torch.int64) - metadata[:-1].to(torch.int64)
        if cumulative
        else metadata
    )
    return max(int(lengths.max().item()), 1)


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
