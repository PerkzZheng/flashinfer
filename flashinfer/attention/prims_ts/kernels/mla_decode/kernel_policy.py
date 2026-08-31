# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit selector for MLA TS kernel families.

The runner asks this module to choose between the throughput 2CTA M128 schedule
and the throughput-latency 1CTA schedule. Selection is explicit:
``requested_policy`` is honored first, and there is no implicit fallback between
families. The returned ``MlaKernelDecision`` tells the caller whether the
requested policy is implemented for the shape; callers must reject or report
``implementation_ready=False`` before constructing a kernel.

The 2CTA predicate is a pure shape/feature eligibility check. The 1CTA path is
additionally gated by profile enumeration, and the returned
``config``/``profile_name`` are populated only when a matching 1CTA profile
exists. Invalid policy names raise ``ValueError`` in
``normalize_mla_kernel_policy``; invalid explicit profile names are propagated
from ``make_throughput_latency_mla_config``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from .helpers.constants import SUPPORTED_MLA_PAGE_SIZES
from .throughput_latency_1cta.config import (
    MlaConfig,
    SUPPORTED_TILE_SIZE_Q,
    enumerate_throughput_latency_mla_profiles,
    make_throughput_latency_mla_config,
    tile_size_q_from_profile_name,
)


MlaKernelPolicy = Literal["throughput_2cta", "throughput_latency_1cta"]
"""User-visible TS MLA policy names accepted by the runner."""

MlaKernelName = MlaKernelPolicy
"""Concrete TS MLA kernel family selected for a launch."""


@dataclass(frozen=True)
class MlaAutomaticWork:
    """Candidate topology facts consumed by automatic family selection."""

    one_cta_work: int | None = None
    one_cta_capacity: int | None = None
    two_cta_cluster_work: int | None = None
    two_cta_cluster_capacity: int | None = None
    allow_equal_wave_crossover: bool = False
    one_cta_tile_size_q: int | None = None
    one_cta_split_kv: int | None = None
    two_cta_split_kv: int | None = None
    seq_len_k: int | None = None
    qkv_dtype: str | None = None
    one_cta_kernel_variant: str | None = None


_SMALL_FLAT_2CTA_WAVE_FILL_NUMERATOR = 5
_SMALL_FLAT_2CTA_WAVE_FILL_DENOMINATOR = 6
_SMALL_FLAT_2CTA_MAX_REFERENCE_SPLIT = 4
# Bound each 2CTA producer's K128 mainloop independently of the 1CTA
# split-rounding policy; the two decisions model different task graphs.
_SMALL_FLAT_DIRECT_BF16_MAX_LOCAL_K_TILES = 16
_SMALL_FLAT_DIRECT_FP8_MAX_LOCAL_K_TILES = 18
_SMALL_FLAT_COMPACT_BF16_MIN_LOCAL_K_TILES = 5
_SMALL_FLAT_COMPACT_BF16_MAX_LOCAL_K_TILES = 10
_SMALL_FLAT_COMPACT_FP8_MIN_LOCAL_K_TILES = 9
_SMALL_FLAT_COMPACT_FP8_MAX_LOCAL_K_TILES = 13


def _retains_normalized_producer_work(
    *,
    candidate_work: int,
    candidate_capacity: int,
    reference_work: int,
    reference_capacity: int,
) -> bool:
    """Return whether candidate occupancy retains five sixths of reference."""

    return (
        candidate_work * reference_capacity * _SMALL_FLAT_2CTA_WAVE_FILL_DENOMINATOR
        >= reference_work * candidate_capacity * _SMALL_FLAT_2CTA_WAVE_FILL_NUMERATOR
    )


def _prefer_small_flat_2cta(
    num_heads: int,
    seq_len_q: int,
    work: MlaAutomaticWork,
) -> bool:
    """Return whether a compact 2CTA topology qualifies for automatic use."""

    total_q_rows = num_heads * seq_len_q
    if (
        not 1 <= total_q_rows <= 64
        or work.one_cta_tile_size_q not in SUPPORTED_TILE_SIZE_Q
        or total_q_rows > work.one_cta_tile_size_q
    ):
        return False
    inputs = (
        work.one_cta_split_kv,
        work.two_cta_split_kv,
        work.seq_len_k,
        work.one_cta_work,
        work.one_cta_capacity,
        work.two_cta_cluster_work,
        work.two_cta_cluster_capacity,
    )
    if any(value is None for value in inputs):
        return False
    if any(int(value) <= 0 for value in inputs):
        raise ValueError("automatic MLA topology inputs must be positive")

    one_cta_split_kv = work.one_cta_split_kv
    two_cta_split_kv = work.two_cta_split_kv
    seq_len_k = work.seq_len_k
    one_cta_work = work.one_cta_work
    one_cta_capacity = work.one_cta_capacity
    two_cta_cluster_work = work.two_cta_cluster_work
    two_cta_cluster_capacity = work.two_cta_cluster_capacity
    assert one_cta_split_kv is not None
    assert two_cta_split_kv is not None
    assert seq_len_k is not None
    assert one_cta_work is not None
    assert one_cta_capacity is not None
    assert two_cta_cluster_work is not None
    assert two_cta_cluster_capacity is not None
    if one_cta_split_kv <= 1:
        return False
    # Limit this crossover to direct output and the bounded reference-reducer
    # loop. Equal producer-wave work does not model the increasing reduction
    # cost of deeper split domains.
    if (
        two_cta_split_kv >= one_cta_split_kv
        or two_cta_split_kv > _SMALL_FLAT_2CTA_MAX_REFERENCE_SPLIT
    ):
        return False
    if (
        two_cta_cluster_work * _SMALL_FLAT_2CTA_WAVE_FILL_DENOMINATOR
        < two_cta_cluster_capacity * _SMALL_FLAT_2CTA_WAVE_FILL_NUMERATOR
        or not _retains_normalized_producer_work(
            candidate_work=two_cta_cluster_work,
            candidate_capacity=two_cta_cluster_capacity,
            reference_work=one_cta_work,
            reference_capacity=one_cta_capacity,
        )
    ):
        return False

    total_k_tiles = (seq_len_k + MlaConfig.tile_size_kv - 1) // MlaConfig.tile_size_kv
    # A 2CTA cluster divides each split rank across two producer CTAs. Measure
    # the candidate's actual per-CTA mainloop instead of inferring it from the
    # independently selected 1CTA split count.
    candidate_ranks = 2 * two_cta_split_kv
    local_k_tiles = (total_k_tiles + candidate_ranks - 1) // candidate_ranks
    if two_cta_split_kv == 1:
        # Direct output has no reducer.  Its crossover is mainloop depth within
        # the physical task graph: BF16 remains favorable through 16 local K128
        # tiles, while the FP8 Keeps producer crosses earlier.  FP8 Swaps does
        # not have enough per-CTA row work to offset 2CTA coordination.
        if work.qkv_dtype == "bf16":
            return local_k_tiles <= _SMALL_FLAT_DIRECT_BF16_MAX_LOCAL_K_TILES
        return (
            work.qkv_dtype == "e4m3"
            and work.one_cta_kernel_variant == "keeps_mma_ab"
            and local_k_tiles <= _SMALL_FLAT_DIRECT_FP8_MAX_LOCAL_K_TILES
        )

    # The compact reducer is eligible only while each 2CTA producer owns a
    # bounded local mainloop. BF16 reaches that window sooner; FP8 participates
    # only for the Keeps task graph, where its parallel reducer has more work.
    two_cta_is_power_of_two = two_cta_split_kv & (two_cta_split_kv - 1) == 0
    if work.qkv_dtype == "bf16":
        min_local_k_tiles = _SMALL_FLAT_COMPACT_BF16_MIN_LOCAL_K_TILES
        max_local_k_tiles = _SMALL_FLAT_COMPACT_BF16_MAX_LOCAL_K_TILES
    elif work.qkv_dtype == "e4m3" and work.one_cta_kernel_variant == "keeps_mma_ab":
        min_local_k_tiles = _SMALL_FLAT_COMPACT_FP8_MIN_LOCAL_K_TILES
        max_local_k_tiles = _SMALL_FLAT_COMPACT_FP8_MAX_LOCAL_K_TILES
    else:
        return False
    return (
        min_local_k_tiles <= local_k_tiles <= max_local_k_tiles
        and two_cta_is_power_of_two
    )


@dataclass(frozen=True)
class MlaKernelDecision:
    """Python-side result of explicit MLA TS kernel selection.

    ``requested_policy`` and ``selected_kernel`` are equal because selection does
    not fall back. ``throughput_latency_candidate`` means the requested shape has at
    least one throughput-latency 1CTA profile. ``implementation_ready`` is the
    final launchability bit for the requested policy. ``reason`` is a
    user-facing explanation for ready or rejected decisions. ``config`` is
    ``None`` for throughput 2CTA decisions and for unsupported 1CTA requests.
    ``available_profiles`` lists the profile names visible for explicit 1CTA
    selection.
    """

    requested_policy: MlaKernelPolicy
    selected_kernel: MlaKernelName
    throughput_latency_candidate: bool
    implementation_ready: bool
    reason: str
    config: MlaConfig | None
    profile_name: str | None = None
    available_profiles: tuple[str, ...] = ()


def normalize_mla_kernel_policy(policy: str) -> MlaKernelPolicy:
    """Return a typed MLA TS policy or raise ``ValueError`` for bad input."""

    if policy not in ("throughput_2cta", "throughput_latency_1cta"):
        raise ValueError(
            "TS MLA kernel policy must be 'throughput_2cta', "
            "or 'throughput_latency_1cta'"
        )
    return cast(MlaKernelPolicy, policy)


def select_default_mla_kernel_policy(
    num_heads: int,
    seq_len_q: int,
    automatic_work: MlaAutomaticWork | None = None,
) -> MlaKernelPolicy:
    """Choose the default TS MLA family from work and resident capacity.

    Logical Q rows up to 64 normally retain the throughput-latency family. A
    one-tile flat launch uses 2CTA when its direct-output or smaller reference
    reduction retains the 1CTA candidate's normalized producer-wave work.
    Larger shapes normally retain 2CTA, but an automatic caller may provide
    projected work for both candidates. The underfilled-grid probe requires
    the complete 2CTA launch to occupy at most one quarter of its resident cluster
    wave and the 1CTA candidate to improve normalized occupancy. A candidate
    with a separately proven one-wave local-work bound may use the whole 2CTA
    wave and accept equal normalized occupancy. No device-name whitelist
    participates in any decision.
    """

    work = automatic_work or MlaAutomaticWork()

    # When one physical 1CTA tile needs a short-K standalone reducer, a compact
    # 2CTA wave can remove or shrink that reducer without losing wave fill.
    if num_heads * seq_len_q <= 64:
        if _prefer_small_flat_2cta(
            num_heads,
            seq_len_q,
            work,
        ):
            return "throughput_2cta"
        return "throughput_latency_1cta"

    occupancy_inputs = (
        work.one_cta_work,
        work.one_cta_capacity,
        work.two_cta_cluster_work,
        work.two_cta_cluster_capacity,
    )
    if any(value is None for value in occupancy_inputs):
        return "throughput_2cta"
    if any(int(value) <= 0 for value in occupancy_inputs):
        raise ValueError("automatic MLA family work and capacities must be positive")

    one_cta_work = work.one_cta_work
    one_cta_capacity = work.one_cta_capacity
    two_cta_cluster_work = work.two_cta_cluster_work
    two_cta_cluster_capacity = work.two_cta_cluster_capacity
    assert one_cta_work is not None
    assert one_cta_capacity is not None
    assert two_cta_cluster_work is not None
    assert two_cta_cluster_capacity is not None
    if work.allow_equal_wave_crossover:
        two_cta_underfilled = two_cta_cluster_work <= two_cta_cluster_capacity
        one_cta_has_more_occupancy = (
            one_cta_work * two_cta_cluster_capacity
            >= two_cta_cluster_work * one_cta_capacity
        )
    else:
        two_cta_underfilled = two_cta_cluster_work * 4 <= two_cta_cluster_capacity
        one_cta_has_more_occupancy = (
            one_cta_work * two_cta_cluster_capacity
            > two_cta_cluster_work * one_cta_capacity
        )
    if two_cta_underfilled and one_cta_has_more_occupancy:
        return "throughput_latency_1cta"
    return "throughput_2cta"


def resolve_mla_kernel_policy(
    policy: str | None,
    num_heads: int,
    seq_len_q: int,
    automatic_work: MlaAutomaticWork | None = None,
) -> tuple[MlaKernelPolicy, str]:
    """Resolve an explicit or automatic TS MLA kernel policy."""

    if policy in (None, "", "auto"):
        return (
            select_default_mla_kernel_policy(
                num_heads,
                seq_len_q,
                automatic_work,
            ),
            "auto",
        )
    return normalize_mla_kernel_policy(policy), "explicit"


def is_throughput_2cta_mla_supported_shape(
    *,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    dtype: str,
    out_dtype: str = "bf16",
) -> bool:
    """Return whether the 2CTA M128 TS MLA path is eligible.

    The predicate is intentionally side-effect free: it only checks the dense
    MLA shape constraints that can be known before kernel construction.
    """

    del batch_size
    return (
        dtype in ("bf16", "e4m3")
        and out_dtype in ("bf16", "e4m3")
        and latent_dim == 512
        and rope_dim == 64
        and page_size in SUPPORTED_MLA_PAGE_SIZES
        and 1 <= num_heads <= 128
        and seq_len_q >= 1
        and seq_len_k >= 1
    )


def _make_config_for_profile(
    *,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    qkv_dtype: str,
    o_dtype: str,
    profile_name: str | None,
    tile_size_q: int | None,
    max_active_clusters: int,
    explicit_split_kv: int | None,
    explicit_persistent: bool | None,
) -> MlaConfig:
    """Build the 1CTA config for one candidate policy profile."""

    return make_throughput_latency_mla_config(
        batch_size=batch_size,
        num_heads_q=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        num_tokens_per_page=page_size,
        qkv_dtype=qkv_dtype,
        o_dtype=o_dtype,
        profile=profile_name,
        tile_size_q=tile_size_q,
        max_active_clusters=max_active_clusters,
        explicit_split_kv=explicit_split_kv,
        explicit_persistent=explicit_persistent,
    )


def select_mla_ts_kernel(
    *,
    requested_policy: MlaKernelPolicy,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    dtype: str = "bf16",
    out_dtype: str = "bf16",
    throughput_latency_profile: str | None = None,
    throughput_latency_tile_size_q: int | None = None,
    max_active_clusters: int,
    throughput_latency_split_kv: int | None = None,
    throughput_latency_persistent: bool | None = None,
) -> MlaKernelDecision:
    """Resolve one of the two explicit TS MLA kernel families.

    ``requested_policy`` is never rewritten to another family. For
    throughput 2CTA requests, ``implementation_ready`` mirrors the static 2CTA
    eligibility predicate. For throughput-latency 1CTA requests, it mirrors
    profile availability and includes the selected profile config. A bad policy
    string raises ``ValueError`` through ``normalize_mla_kernel_policy``; a bad
    explicit profile raises from the config factory.
    """

    requested_policy = normalize_mla_kernel_policy(requested_policy)
    throughput_2cta_candidate = is_throughput_2cta_mla_supported_shape(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        page_size=page_size,
        dtype=dtype,
        out_dtype=out_dtype,
    )
    profile_tile_size_q = throughput_latency_tile_size_q
    if profile_tile_size_q is None:
        profile_tile_size_q = tile_size_q_from_profile_name(throughput_latency_profile)

    profiles = enumerate_throughput_latency_mla_profiles(
        batch_size=batch_size,
        num_heads_q=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        num_tokens_per_page=page_size,
        tile_size_q=profile_tile_size_q,
        max_active_clusters=max_active_clusters,
        qkv_dtype=dtype,
        explicit_split_kv=throughput_latency_split_kv,
        explicit_persistent=throughput_latency_persistent,
    )
    profile_names = tuple(profile.name for profile in profiles)
    supported_dtype_pair = dtype in ("bf16", "e4m3") and out_dtype in (
        "bf16",
        "e4m3",
    )
    throughput_latency_candidate = supported_dtype_pair and bool(profiles)
    default_profile = None
    if throughput_latency_candidate:
        default_profile = throughput_latency_profile
        if default_profile is None and profile_names:
            default_profile = profile_names[0]

    if requested_policy == "throughput_2cta":
        reason = (
            "forced throughput 2CTA M128 TS MLA path"
            if throughput_2cta_candidate
            else (
                "throughput 2CTA M128 TS MLA path requires BF16 or E4M3 input "
                "with BF16 or E4M3 output"
                if dtype not in ("bf16", "e4m3") or out_dtype not in ("bf16", "e4m3")
                else "shape/features outside the throughput 2CTA M128 TS MLA path"
            )
        )
        return MlaKernelDecision(
            requested_policy=requested_policy,
            selected_kernel="throughput_2cta",
            throughput_latency_candidate=throughput_latency_candidate,
            implementation_ready=throughput_2cta_candidate,
            reason=reason,
            config=None,
            profile_name=None,
            available_profiles=profile_names,
        )

    cfg = (
        _make_config_for_profile(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            latent_dim=latent_dim,
            rope_dim=rope_dim,
            page_size=page_size,
            qkv_dtype=dtype,
            o_dtype=out_dtype,
            profile_name=default_profile,
            tile_size_q=throughput_latency_tile_size_q,
            max_active_clusters=max_active_clusters,
            explicit_split_kv=throughput_latency_split_kv,
            explicit_persistent=throughput_latency_persistent,
        )
        if throughput_latency_candidate
        else None
    )
    reason = (
        "forced throughput-latency 1CTA MLA TS path"
        if throughput_latency_candidate
        else (
            "throughput-latency 1CTA MLA TS path requires BF16 or E4M3 input with BF16 or E4M3 output"
            if not supported_dtype_pair
            else "shape/features outside the throughput-latency 1CTA MLA TS path"
        )
    )
    return MlaKernelDecision(
        requested_policy=requested_policy,
        selected_kernel="throughput_latency_1cta",
        throughput_latency_candidate=throughput_latency_candidate,
        implementation_ready=throughput_latency_candidate,
        reason=reason,
        config=cfg,
        profile_name=default_profile,
        available_profiles=profile_names,
    )
