# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Shared dispatch records; all tensor storage stays with its owning wrapper."""

from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Any

import torch

PolicyCacheInfo = namedtuple("PolicyCacheInfo", "hits misses maxsize currsize")
# Resource bindings and caller validation do not select a native policy.
_BINDING_OPTIONS = frozenset(
    (
        "initialize_workspace",
        "validate",
        "sm_scale",
        "scale_softmax_log2",
        "output_scale",
    )
)


def model_key(options: Mapping) -> frozenset:
    """Freeze typed model fields once per plan; frozenset caches its hash."""
    values = []
    for name, value in options.items():
        if name in _BINDING_OPTIONS:
            continue
        if isinstance(value, torch.Tensor):
            raise TypeError("shared attention policy keys cannot retain tensors")
        values.append((name, type(value), value))
    return frozenset(values)


def _copy_mutable_record(value: Any) -> Any:
    params = getattr(type(value), "__dataclass_params__", None)
    return copy.copy(value) if params is not None and not params.frozen else value


@dataclass(frozen=True)
class SharedPolicy:
    """Host plan state; callers exclude every per-layer tensor binding."""

    state_type: type
    values: Mapping

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def bind(self, **bindings) -> Any:
        values = {
            name: _copy_mutable_record(value) for name, value in self.values.items()
        }
        return self.state_type(**values, **bindings)


class PolicyCache:
    """Process-local CPU cache with one builder per concurrent key."""

    def __init__(self):
        self._entries: dict[tuple, SharedPolicy] = {}
        self._pending: dict[tuple, threading.Event] = {}
        self._lock = threading.Lock()
        self._hits = self._misses = 0

    def peek(self, key: tuple) -> SharedPolicy | None:
        with self._lock:
            return self._entries.get(key)

    def get_or_create(
        self, key: tuple, factory: Callable[[], SharedPolicy]
    ) -> SharedPolicy:
        while True:
            with self._lock:
                policy = self._entries.get(key)
                if policy is not None:
                    self._hits += 1
                    return policy
                event = self._pending.get(key)
                if event is None:
                    event = threading.Event()
                    self._pending[key] = event
                    self._misses += 1
                    break
            event.wait()
        try:
            policy = factory()
        except BaseException:
            with self._lock:
                self._pending.pop(key).set()
            raise
        with self._lock:
            self._entries[key] = policy
            self._pending.pop(key).set()
        return policy

    def cache_info(self):
        with self._lock:
            return PolicyCacheInfo(self._hits, self._misses, None, len(self._entries))

    def cache_clear(self):
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot clear a policy cache with active builders")
            self._entries.clear()
            self._hits = self._misses = 0


shared_policy_cache = PolicyCache()
