#!/usr/bin/env python3
"""Read-only Nsight SQLite accounting; timings are not speedup predictions.

Example: python analyze_ipc_light_cost.py --sqlite run/nsight.sqlite --output new.json
Only the requested output is written (exclusive creation, never overwritten).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


APP_PREFIXES = (
    "ipc.", "collision.", "linear.", "mas.", "pcg.", "graph.", "abd.",
    "preconditioner.", "global_preconditioner.", "local_preconditioner.",
    "diagnostic.", "transfer.",
)
TABLES = {
    "kernel": "CUPTI_ACTIVITY_KIND_KERNEL",
    "memcpy": "CUPTI_ACTIVITY_KIND_MEMCPY",
    "memset": "CUPTI_ACTIVITY_KIND_MEMSET",
}


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def clip(start, end, windows):
    return [(max(start, a), min(end, b)) for a, b in windows
            if start < b and end > a]


def timing(intervals, count=None):
    intervals = list(intervals)
    union = merge(intervals)
    summed = sum(b - a for a, b in intervals)
    covered = sum(b - a for a, b in union)
    span = union[-1][1] - union[0][0] if union else 0
    return {
        "count": len(intervals) if count is None else count,
        "sum_ms": summed / 1e6,
        "union_ms": covered / 1e6,
        "first_start_ns": union[0][0] if union else None,
        "last_end_ns": union[-1][1] if union else None,
        "first_to_last_span_ms": span / 1e6,
        "gaps_between_activities_ms": (span - covered) / 1e6,
    }


def selected(rows, windows):
    return [r for r in rows if clip(r["start"], r["end"], windows)]


def row_timing(rows, windows=None):
    return timing((part for r in rows for part in
                   (clip(r["start"], r["end"], windows) if windows is not None
                    else [(r["start"], r["end"])])), len(rows))


def group_rows(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(path, top):
    # No schema migration, index, journal, or temporary view is created.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        existing = {r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        schemas = {table: [r[1] for r in connection.execute(
            f'PRAGMA table_info("{table}")')] for table in
            ["StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", *TABLES.values()]
            if table in existing}

        def load(table):
            if table not in existing:
                return []
            return [dict(r) for r in connection.execute(f'SELECT * FROM "{table}"')]

        strings = {r["id"]: r["value"] for r in load("StringIds")}

        def name(row, fields):
            for field in fields:
                value = row.get(field)
                if value is not None:
                    return strings.get(value, str(value)) if isinstance(value, int) else value
            return "<unnamed>"

        ranges = []
        for i, row in enumerate(load("NVTX_EVENTS")):
            if row.get("end") is None or row["end"] <= row["start"]:
                continue
            row.update(id=i, name=name(row, ("text", "textId")))
            ranges.append(row)
        frames = sorted((r for r in ranges if r["name"] == "ipc.physical_frame"),
                        key=lambda r: (r["start"], r["end"]))
        if not frames:
            raise ValueError("No completed ipc.physical_frame NVTX range; capture may be truncated.")
        windows = merge((r["start"], r["end"]) for r in frames)
        app_ranges = [r for r in ranges if r["name"].startswith(APP_PREFIXES)]
        runtime = load("CUPTI_ACTIVITY_KIND_RUNTIME")
        runtime = [r for r in runtime if r.get("end") is not None and r["end"] >= r["start"]]
        correlation = defaultdict(list)
        ranges_by_thread = group_rows(app_ranges, lambda r: r.get("globalTid"))
        for i, row in enumerate(runtime):
            row.update(id=i, name=name(row, ("name", "nameId")))
            owners = [s for s in ranges_by_thread.get(row.get("globalTid"), [])
                      if s["start"] <= row["start"] and row["end"] <= s["end"]]
            row["owner"] = min(owners, key=lambda s: s["end"] - s["start"]) if owners else None
            if row.get("correlationId") is not None:
                correlation[row["correlationId"]].append(row)

        activities = []
        for kind, table in TABLES.items():
            for i, row in enumerate(load(table)):
                if row.get("end") is None or row["end"] <= row["start"]:
                    continue
                row.update(id=f"{kind}:{i}", kind=kind,
                           name=name(row, ("demangledName", "shortName", "name", "nameId")))
                matches = correlation.get(row.get("correlationId"), [])
                # A collision across processes or repeated IDs stays unassigned.
                row["runtime"] = matches[0] if len(matches) == 1 else None
                row["correlation_match"] = "unique" if len(matches) == 1 else (
                    "ambiguous" if matches else "missing")
                activities.append(row)
        labels = {r["id"]: r.get("label", r.get("name", str(r["id"])))
                  for r in load("ENUM_CUDA_MEMCPY_OPER")}
    finally:
        connection.close()

    def api_summary(rows, active_windows):
        groups = group_rows(rows, lambda r: r["name"])
        by_name = {n: row_timing(g, active_windows) for n, g in sorted(groups.items())}
        event_groups = {
            "create": [r for r in rows if r["name"].startswith("cudaEventCreate")],
            "record": [r for r in rows if r["name"].startswith("cudaEventRecord")],
            "synchronize": [r for r in rows if r["name"].startswith("cudaEventSynchronize")],
            "destroy": [r for r in rows if r["name"].startswith("cudaEventDestroy")],
        }
        no_events = not any(event_groups[k] for k in ("create", "record", "synchronize"))
        event_names_available = "CUPTI_ACTIVITY_KIND_RUNTIME" in schemas and all(
            not r["name"].isdigit() and r["name"] != "<unnamed>" for r in rows)
        memcpy = [r for r in rows if r["name"].startswith("cudaMemcpy")]
        synchronization = [r for r in rows if any(r["name"].startswith(p) for p in
                            ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"))]
        return {
            "all": row_timing(rows, active_windows),
            "by_name": by_name,
            "memcpy_including_waits": row_timing(memcpy, active_windows),
            "explicit_synchronize_including_waits": row_timing(synchronization, active_windows),
            "cuda_events": {k: row_timing(v, active_windows) for k, v in event_groups.items()},
            "cost_event_audit": {
                "observed_create_record_sync_all_zero": no_events if event_names_available else None,
                "additional_cost_event_apis_in_selected_windows": 0 if no_events and event_names_available else None,
                "zero_additional_cost_event_apis_observed_in_selected_windows": no_events and event_names_available,
                "basis": "Runtime table or API names are unavailable; event counts are unknown."
                    if not event_names_available else
                    "No create/record/sync API observed in the selected windows. This does not cover "
                    "instrumentation before/after those windows or dropped capture records." if no_events else
                    "SQLite API rows cannot distinguish production events from CostScope events; "
                    "configuration/source evidence or a matched baseline is required.",
            },
        }

    def activity_summary(rows, active_windows):
        directions = group_rows([r for r in rows if r["kind"] == "memcpy"],
                                lambda r: labels.get(r.get("copyKind"), str(r.get("copyKind"))))
        coverage = group_rows(rows, lambda r: r["correlation_match"])
        by_kind = {kind: row_timing([r for r in rows if r["kind"] == kind], active_windows)
                   for kind in TABLES}
        return {
            "all": row_timing(rows, active_windows),
            "by_kind": by_kind,
            "by_device": {str(k): row_timing(v, active_windows) for k, v in sorted(
                group_rows(rows, lambda r: str(r.get("deviceId", "unknown"))).items())},
            "memcpy_directions": {key: {**row_timing(group, active_windows),
                "bytes_full_records": sum(r.get("bytes", 0) for r in group),
                "max_bytes": max((r.get("bytes", 0) for r in group), default=0),
                "boundary_crossing_records": sum(len(clip(r["start"], r["end"], active_windows)) != 1
                    or clip(r["start"], r["end"], active_windows)[0] != (r["start"], r["end"])
                    for r in group) if active_windows is not None else 0}
                for key, group in sorted(directions.items())},
            "correlation_coverage": {key: row_timing(group, active_windows)
                                     for key, group in sorted(coverage.items())},
            "graph_node_records": sum(bool(r.get("graphNodeId")) for r in rows),
            "unmatched_graph_node_records": sum(bool(r.get("graphNodeId")) and r["runtime"] is None
                                                for r in rows),
        }

    def owner_name(row):
        api = row["runtime"]
        if api is not None and api["owner"] is not None:
            return api["owner"]["name"]
        return "<unattributed_graph_node>" if row.get("graphNodeId") else "<unattributed>"

    frame_reports = []
    for i, frame in enumerate(frames):
        window = [(frame["start"], frame["end"])]
        gpu = selected(activities, window)
        apis = selected(runtime, window)
        # Correlated work may complete after the CPU range, unlike timeline clipping.
        related = [r for r in activities if r["runtime"] is not None and
                   r["runtime"].get("globalTid") == frame.get("globalTid") and
                   frame["start"] <= r["runtime"]["start"] < frame["end"]]
        frame_reports.append({
            "capture_frame_index": i + 1,
            "simulation_frame_number": None,
            "global_tid": frame.get("globalTid"),
            "cpu_nvtx_wall": row_timing([frame]),
            "gpu_timeline_clipped": activity_summary(gpu, window),
            "gpu_related_runtime_full_intervals": activity_summary(related, None),
            "runtime_api": api_summary(apis, window),
        })

    gpu = selected(activities, windows)
    apis = selected(runtime, windows)
    kernel_groups = group_rows([r for r in gpu if r["kind"] == "kernel"], lambda r: r["name"])
    hottest = sorted(kernel_groups, key=lambda n: row_timing(kernel_groups[n], windows)["sum_ms"],
                     reverse=True)[:top] if top else list(kernel_groups)
    kernels = []
    for kernel_name in sorted(hottest):
        rows = kernel_groups[kernel_name]
        kernels.append({"name": kernel_name, **row_timing(rows, windows),
                        "exclusive_runtime_correlation_stage": {
                            key: row_timing(group, windows) for key, group in sorted(
                                group_rows(rows, owner_name).items())}})
    stage_groups = group_rows(selected(app_ranges, windows), lambda r: r["name"])
    exclusive_groups = group_rows(gpu, owner_name)
    stages = []
    for stage_name, rows in sorted(stage_groups.items()):
        stage_windows = merge(part for r in rows for part in clip(r["start"], r["end"], windows))
        stages.append({
            "name": stage_name,
            "cpu_nvtx_inclusive": row_timing(rows, windows),
            "gpu_temporal_overlap_inclusive_not_causal": row_timing(selected(gpu, stage_windows), stage_windows),
            "gpu_exclusive_runtime_correlation": row_timing(exclusive_groups.get(stage_name, []), windows),
        })
    return {
        "schema": 1,
        "source_sqlite": str(path.resolve()),
        "source_sha256": file_hash(path),
        "analyzer_sha256": file_hash(Path(__file__)),
        "read_only": True,
        "captured_physical_frames": len(frames),
        "available_table_columns": schemas,
        "missing_activity_tables": sorted(set(TABLES.values()) - set(schemas)),
        "whole_capture_runtime_api": api_summary(runtime, None),
        "method": {
            "time_unit": "milliseconds; original timestamps are nanoseconds",
            "timeline": "GPU intervals are clipped to the union of CPU physical-frame windows. "
                        "Bytes count full records, with boundary crossings reported.",
            "sum_vs_union": "Sum counts overlapping activities separately; union measures covered wall time. "
                            "Neither is a critical-path saving. Aggregate inter-frame gaps include uncaptured time.",
            "scope_accounting": "NVTX CPU and GPU temporal-overlap stages are inclusive and must not be added. "
                                "Exclusive runtime-correlation stages partition attributed GPU records only.",
            "correlation": "A unique correlationId maps a GPU record to its runtime API; the smallest "
                           "enclosing same-thread application NVTX range owns it. Ambiguous IDs stay unassigned. "
                           "Graph capture/replay nodes often lack runtime linkage. Temporal overlap does not "
                           "repair that causal gap, and a graph.replay owner does not identify an internal operator.",
            "cpu_api": "CUDA API duration includes GPU waits and can overlap GPU activity; never add it to GPU time.",
            "gaps": "GPU inactivity does not imply removable CPU overhead: dependencies, driver scheduling, "
                    "uncaptured activity, and asynchronous boundaries also cause gaps.",
            "events": "Only API counts are observable; attributing nonzero events to CostScope requires "
                      "configuration/source evidence or a matched baseline. Missing tables mean unknown, not zero.",
            "devices": "The combined union is wall-time coverage across all observed devices/processes; "
                       "per-device results are also provided. This is not GPU occupancy or utilization.",
        },
        "frames": frame_reports,
        "aggregate": {
            "cpu_physical_frame_wall": row_timing(frames),
            "gpu_timeline_clipped": activity_summary(gpu, windows),
            "runtime_api": api_summary(apis, windows),
            "exclusive_runtime_correlation_stages": {
                n: row_timing(g, windows) for n, g in sorted(exclusive_groups.items())},
        },
        "kernel_hotspots": {"selection": "top by clipped sum_ms, displayed alphabetically",
                            "top_limit": top, "distinct_kernels": len(kernel_groups), "rows": kernels},
        "stages_alphabetical": stages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New JSON only; existing files are refused")
    parser.add_argument("--top", type=int, default=20, help="Kernel hotspot count, 0 for all (default: 20)")
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be nonnegative")
    try:
        result = analyze(args.sqlite, args.top)
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
            print(json.dumps({"output": str(args.output.resolve()),
                              "captured_physical_frames": result["captured_physical_frames"],
                              "gpu": result["aggregate"]["gpu_timeline_clipped"]["all"]}))
        else:
            print(encoded)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
