# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static regression checks for the PrimTS FMHA decode PDL contract.

These tests deliberately parse the kernel sources instead of racing two GPU
launches.  A missing PDL wait or an early producer trigger is timing-dependent,
so a runtime test can pass despite violating the ordering contract.
"""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DECODE_KERNEL = (
    _REPO_ROOT
    / "flashinfer/attention/prims_ts/kernels/fmha_decode/fmha_decode_kernel.py"
)
_DECODE_REDUCTION = (
    _REPO_ROOT / "flashinfer/attention/prims_ts/kernels/fmha_decode/reduction.py"
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one definition of {name}, found {len(matches)}"
    return matches[0]


def _launched_kernel(call: ast.Call) -> str | None:
    """Return ``kernel`` for ``kernel(...).launch(...)`` AST nodes."""
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "launch":
        return None
    construction = call.func.value
    if not isinstance(construction, ast.Call):
        return None
    target = construction.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _launches(node: ast.AST, kernel: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _launched_kernel(child) == kernel
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    assert len(matches) <= 1, f"duplicate {name}= on line {call.lineno}"
    return matches[0] if matches else None


def _is_cfg_property(node: ast.expr | None, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "cfg"
    )


def _griddep_actions(node: ast.AST) -> list[tuple[str, int]]:
    actions: list[tuple[str, int]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr != "griddepcontrol":
            continue
        kind = _keyword(child, "kind")
        assert isinstance(kind, ast.Attribute), (
            f"griddepcontrol on line {child.lineno} must name a GridDepAction"
        )
        actions.append((kind.attr, child.lineno))
    return sorted(actions, key=lambda action: action[1])


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (
            (isinstance(child.func, ast.Name) and child.func.id == name)
            or (isinstance(child.func, ast.Attribute) and child.func.attr == name)
        )
    ]


def _executable_body(function: ast.FunctionDef) -> list[ast.stmt]:
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def test_only_parallel_reducer_uses_a_pdl_launch() -> None:
    """The metadata-reading producer must retain ordinary stream ordering."""
    producer_tree = _parse(_DECODE_KERNEL)
    reduction_tree = _parse(_DECODE_REDUCTION)

    producer_launches = _launches(producer_tree, "decode_gen_kernel")
    assert producer_launches, "no PrimTS FMHA decode producer launch found"
    assert all(
        isinstance(_keyword(launch, "use_pdl"), ast.Constant)
        and _keyword(launch, "use_pdl").value is False
        for launch in producer_launches
    ), "decode_gen_kernel reads runtime metadata and must launch with use_pdl=False"

    reducer_launches = _launches(
        reduction_tree, "decode_gen_parallel_separate_reduction_kernel"
    )
    assert len(reducer_launches) == 1
    assert _is_cfg_property(
        _keyword(reducer_launches[0], "use_pdl"),
        "use_parallel_separate_reduction_pdl",
    )

    pdl_launches: list[str] = []
    for tree in (producer_tree, reduction_tree):
        for child in ast.walk(tree):
            if not isinstance(child, ast.Call):
                continue
            kernel = _launched_kernel(child)
            if kernel is None:
                continue
            use_pdl = _keyword(child, "use_pdl")
            if use_pdl is not None and not (
                isinstance(use_pdl, ast.Constant) and use_pdl.value is False
            ):
                pdl_launches.append(kernel)
    assert pdl_launches == ["decode_gen_parallel_separate_reduction_kernel"]


def test_parallel_reducer_initializes_local_state_before_pdl_wait() -> None:
    """Both reducer schedules acquire after setup and before partial reads."""
    tree = _parse(_DECODE_REDUCTION)
    compact = _function(tree, "_reduce_exact_splits_body")
    reducer = _function(tree, "decode_gen_parallel_separate_reduction_kernel")

    compact_actions = _griddep_actions(compact)
    assert len(compact_actions) == 1 and compact_actions[0][0] == "WAIT"
    compact_wait_line = compact_actions[0][1]
    assert "wait_for_pdl_producer" in ast.unparse(compact)
    assert max(call.lineno for call in _calls(compact, "Array")) < compact_wait_line
    assert compact_wait_line < min(call.lineno for call in _calls(compact, "inttoptr"))

    reducer_actions = _griddep_actions(reducer)
    assert len(reducer_actions) == 1 and reducer_actions[0][0] == "WAIT"
    reducer_wait_line = reducer_actions[0][1]
    smem_arrays = [
        call
        for call in _calls(reducer, "Array")
        if "AddressSpace.smem" in ast.unparse(call)
    ]
    assert len(smem_arrays) == 2
    assert max(call.lineno for call in smem_arrays) < reducer_wait_line
    assert reducer_wait_line < min(call.lineno for call in _calls(reducer, "inttoptr"))


def test_attention_producer_releases_only_after_tmem_teardown() -> None:
    """Active producer CTAs trigger the reducer at their converged true tail."""
    producer = _function(_parse(_DECODE_KERNEL), "_run_decode_gen_active")
    body = _executable_body(producer)
    assert body and isinstance(body[-1], ast.If)
    release_guard = body[-1]
    assert "use_parallel_separate_reduction_pdl" in ast.unparse(release_guard.test)

    actions = _griddep_actions(producer)
    assert len(actions) == 1 and actions[0][0] == "LAUNCH_DEPENDENTS"
    deallocs = _calls(producer, "tcgen05_dealloc")
    assert deallocs, "producer no longer has a TMEM teardown to order before release"
    assert max(call.lineno for call in deallocs) < actions[0][1]

    # The tail must first converge the CTA, then let exactly one thread issue
    # the grid-wide scheduling trigger.
    assert _calls(release_guard, "barrier_cta_sync")
    assert actions[0][1] == max(
        child.lineno for child in ast.walk(release_guard) if hasattr(child, "lineno")
    )


def test_zero_work_producers_use_the_padded_tail_signal() -> None:
    """Inactive split and packed-Q padding paths still release the reducer."""
    tree = _parse(_DECODE_KERNEL)
    helper = _function(tree, "_signal_padded_pdl_producer")
    helper_body = _executable_body(helper)
    assert len(helper_body) == 1 and isinstance(helper_body[0], ast.If)
    assert "use_parallel_separate_reduction_pdl" in ast.unparse(helper_body[0].test)
    assert [action for action, _ in _griddep_actions(helper)] == ["LAUNCH_DEPENDENTS"]

    # One call retires an inactive split; the other retires a padded packed-Q
    # tile. Keeping both explicit protects CUDA-graph envelope launches.
    runtime_prefix = _function(tree, "_run_decode_gen_runtime_prefix")
    decode_kernel = _function(tree, "decode_gen_kernel")
    assert len(_calls(runtime_prefix, "_signal_padded_pdl_producer")) == 1
    assert len(_calls(decode_kernel, "_signal_padded_pdl_producer")) == 1
