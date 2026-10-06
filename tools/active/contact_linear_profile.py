#!/usr/bin/env python3
"""Read-only GPU-symbol accounting for selected contact-frame Nsight captures.

This deliberately does not attribute replayed kernels through captured CostScope
labels. Every activity has exactly one symbol/kind group; unknowns remain visible.
Intervals are clipped to completed ipc.physical_frame ranges. This is elapsed
activity accounting, not a critical-path analysis or a predicted speedup.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

from analyze_ipc_light_cost import clip, file_hash, merge, row_timing, timing


# Exact function identifiers, permitting namespaces and template arguments.
# Generic CUB reduction is classified by operation but its operand is unresolved:
# neither an NVTX label nor a graphNodeId identifies rho versus p^T A p.
RULES = [
    ("mas.numeric_scatter", ("prepare_hessian_bcoo_kernel",)),
    ("mas.numeric_reduce", ("prepare_hessian_bcoo_sum_kernel",)),
    ("mas.inverse_prepare", ("__inverse6_P96x96", "__inverse6_P96x96_typed")),
    ("mas.factor_prepare", ("symmetric_cholesky",)),
    ("mas.restrict", ("__buildMultiLevelR_optimized_new", "__buildMultiLevelR_optimized_new_wide",
                      "deterministic_restrict", "warp_deterministic_restrict")),
    ("mas.local_action", ("_schwarzLocalXSym6", "_schwarzLocalXSym6_wide",
                          "cholesky_action", "factor_inverse_action")),
    ("mas.prolong", ("__collectFinalZ_new", "__collectFinalZ_new_wide")),
    ("mas.prolong_fused_dot", ("mas_collect_final_z_dot",)),
    ("spmv", ("warp_reduce_sym_spmv_kernel",)),
    ("pcg.dot_partial", ("PCG_vdv_Reduction",)),
    ("pcg.vector_update", ("graph_dx_r",)),
    ("pcg.vector_and_condition", ("graph_p_continue",)),
    ("pcg.control_and_validation", ("graph_alpha", "graph_beta", "graph_init", "graph_check_zero_rho")),
]
COMPILED = [(group, [re.compile(r"(?<![A-Za-z0-9_])" + re.escape(fn) + r"(?=[(<])")
                      for fn in names]) for group, names in RULES]


def classify(name: str) -> str:
    hits = [group for group, patterns in COMPILED if any(p.search(name) for p in patterns)]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous classification: {name}: {hits}")
    if hits:
        return hits[0]
    if "cub::" in name and re.search(r"(?<![A-Za-z0-9_])DeviceReduce(?:SingleTile|Kernel)", name):
        return "reduce.cub_unspecified_operand"
    return "unknown_kernel"


def intersection(a, b):
    a, b = merge(a), merge(b)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            result.append((lo, hi))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def summarize(rows, windows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    all_timing = row_timing(rows, windows)
    categories = {}
    for category, members in sorted(grouped.items()):
        graph = [r for r in members if r.get("graphNodeId")]
        categories[category] = {
            **row_timing(members, windows),
            "graph_node_activity": row_timing(graph, windows),
            "non_graph_activity": row_timing([r for r in members if not r.get("graphNodeId")], windows),
        }
    # Every row has one category. Across-category time overlap is preserved.
    assert abs(sum(x["sum_ms"] for x in categories.values()) - all_timing["sum_ms"]) < 1e-7
    overlap = sum(x["union_ms"] for x in categories.values()) - all_timing["union_ms"]
    return {"all_gpu_activity": all_timing, "categories": categories,
            "cross_category_union_overlap_ms": max(0, overlap),
            "sum_minus_union_ms": max(0, all_timing["sum_ms"] - all_timing["union_ms"])}


def analyze(path: Path, cost_path: Path | None = None):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def load(table):
            return [dict(r) for r in connection.execute(f'SELECT * FROM "{table}"')] if table in tables else []
        strings = {r["id"]: r["value"] for r in load("StringIds")}
        def name(row, fields):
            for field in fields:
                value = row.get(field)
                if value is not None:
                    return strings.get(value, str(value)) if isinstance(value, int) else value
            return "<unnamed>"
        ranges = []
        for row in load("NVTX_EVENTS"):
            if row.get("end") is not None and row["end"] > row["start"]:
                row["name"] = name(row, ("text", "textId"))
                ranges.append(row)
        frames = sorted((r for r in ranges if r["name"] == "ipc.physical_frame"), key=lambda r: r["start"])
        if not frames:
            raise ValueError("No completed ipc.physical_frame range")
        windows = merge((r["start"], r["end"]) for r in frames)
        activities = []
        missing_tables = []
        for kind, table in [("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
                            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
                            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET")]:
            if table not in tables:
                missing_tables.append(table)
            for row in load(table):
                if row.get("end") is None or row["end"] <= row["start"] or not clip(row["start"], row["end"], windows):
                    continue
                row["name"] = name(row, ("demangledName", "shortName", "name", "nameId")) if kind == "kernel" else kind
                row["kind"] = kind
                row["category"] = classify(row["name"]) if kind == "kernel" else "memory." + kind
                activities.append(row)
    finally:
        connection.close()

    frame_ids = []
    if cost_path is not None and cost_path.exists():
        for line in cost_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("stage") == "ipc.physical_frame":
                frame_ids.append(row["frame"])
    frame_ids = sorted(set(frame_ids))
    frame_identity_exact = len(frame_ids) == len(frames)
    frame_results = []
    for index, frame in enumerate(frames):
        w = [(frame["start"], frame["end"])]
        members = [r for r in activities if clip(r["start"], r["end"], w)]
        frame_results.append({"capture_index": index,
            "simulation_frame": frame_ids[index] if frame_identity_exact else None,
            "cpu_physical_frame_wall_ms": (frame["end"] - frame["start"]) / 1e6,
            **summarize(members, w)})

    symbols = defaultdict(list)
    for row in activities:
        if row["kind"] == "kernel":
            symbols[row["name"]].append(row)
    symbol_results = []
    for symbol, members in sorted(symbols.items()):
        graph = [r for r in members if r.get("graphNodeId")]
        symbol_results.append({"symbol": symbol, "category": members[0]["category"],
            **row_timing(members, windows), "graph_node_activity": row_timing(graph, windows),
            "graph_node_ids_count": len({r["graphNodeId"] for r in graph}),
            "registers_per_thread_values": sorted({r["registersPerThread"] for r in members if r.get("registersPerThread") is not None}),
            "block_thread_counts": sorted({r.get("blockX", 1)*r.get("blockY", 1)*r.get("blockZ", 1) for r in members})})

    graph_entries = []
    all_intervals = [(r["start"], r["end"]) for r in activities]
    for scope in sorted((r for r in ranges if r["name"] == "graph.entry"), key=lambda r: r["start"]):
        w = clip(scope["start"], scope["end"], windows)
        if not w:
            continue
        graph = [r for r in activities if r.get("graphNodeId") and clip(r["start"], r["end"], w)]
        gt = row_timing(graph, w)
        span = [(gt["first_start_ns"], gt["last_end_ns"])] if graph else []
        busy = timing(intersection(all_intervals, span))["union_ms"]
        graph_entries.append({"scope_start_ns": scope["start"], "scope_end_ns": scope["end"],
            "graph_node_activity": gt,
            "all_captured_gpu_busy_in_first_last_graph_node_span_ms": busy,
            "graph_span_not_covered_by_any_captured_gpu_ms": max(0, gt["first_to_last_span_ms"]-busy),
            "attribution": "Temporal containment of graph-node activity; no graph/solve identity claim. Gaps include device scheduling and profiler effects, not proven removable host overhead."})

    return {"schema": "contact_linear_profile.v1", "read_only": True,
        "source_sqlite": str(path.resolve()), "source_sqlite_sha256": file_hash(path),
        "analyzer_sha256": file_hash(Path(__file__)),
        "cost_jsonl": str(cost_path.resolve()) if cost_path is not None and cost_path.exists() else None,
        "frame_identity_basis": "Sorted physical-frame cost records matched by equal count; inspect capture configuration as well." if frame_identity_exact else "Unknown: missing or unequal physical-frame cost records.",
        "missing_activity_tables": missing_tables,
        "method": {"classification": "Exact kernel function allowlist; unmatched symbols remain unknown. CUB reductions retain unresolved operands. Memcpy/memset remain unassigned to operators.",
                   "graph": "graphNodeId partitions graph and non-graph activity. No captured CostScope label or CUPTI runtime correlation is required for symbol accounting.",
                   "intervals": "Activities clipped to completed physical-frame windows; per-category sum and union plus cross-category overlap are explicit.",
                   "limits": "GPU sums, CPU ranges, waits and gaps cannot be added into a critical path or used as speedup predictions. Frame-level symbol cost is not full-scene cost. Registers/block size are descriptive, not occupancy or bottleneck measurements."},
        "classification_rules": {k: list(v) for k, v in RULES},
        "frames": frame_results, "aggregate": summarize(activities, windows),
        "symbols_alphabetical": symbol_results, "graph_entries_temporal": graph_entries,
        "performance_certified": False}


def self_test():
    assert classify("gipc::<unnamed>::warp_reduce_sym_spmv_kernel(double)") == "spmv"
    assert classify("void _schwarzLocalXSym6_wide<A>(int)") == "mas.local_action"
    assert classify("prepare_hessian_bcoo_sum_kernel(int)") == "mas.numeric_reduce"
    assert classify("unrelated_spmv_kernel(int)") == "unknown_kernel"
    assert classify("cub::X::DeviceReduceSingleTileKernel<T>(int)") == "reduce.cub_unspecified_operand"
    rows = [{"start": 0, "end": 10, "category": "a", "graphNodeId": 1},
            {"start": 5, "end": 15, "category": "b", "graphNodeId": 0},
            {"start": 20, "end": 25, "category": "unknown_kernel", "graphNodeId": 0}]
    result = summarize(rows, [(2, 22)])
    assert abs(result["all_gpu_activity"]["sum_ms"]-20e-6) < 1e-12
    assert abs(result["all_gpu_activity"]["union_ms"]-15e-6) < 1e-12
    assert abs(result["cross_category_union_overlap_ms"]-5e-6) < 1e-12
    assert intersection([(0, 10), (8, 15)], [(4, 6), (12, 20)]) == [(4, 6), (12, 15)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--cost-jsonl", type=Path)
    parser.add_argument("--output", type=Path, help="Exclusive creation; never overwrite an existing report")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"self_test": "passed"}))
        return
    if args.sqlite is None:
        parser.error("--sqlite is required unless --self-test")
    data = analyze(args.sqlite, args.cost_jsonl or args.sqlite.parent / "cost.jsonl")
    output = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output)
        print(json.dumps({"output": str(args.output.resolve()), "frames": len(data["frames"]),
                          "gpu_union_ms": data["aggregate"]["all_gpu_activity"]["union_ms"]}))
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
