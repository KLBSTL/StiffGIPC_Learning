"""Read saved local measurements; do not run or modify any simulator."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
WORKSPACE = TASK.parent
OLD = WORKSPACE / "stiffGIPC/benchmarks/stiff4-v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_run(root: Path, scene: str, arm: str, repeat: int):
    run = root / scene / arm / f"r{repeat:02d}"
    rows = read_csv(run / "frames.csv")
    status = read_json(run / "runner_status.json")
    assert status["state"] == "completed_diagnostic", run
    assert [int(r["frame"]) for r in rows] == list(range(1, len(rows) + 1)), run
    field = "newton_iterations" if arm == "robust2026" else "newton_direction_solves"
    data = {
        "path": str(run.relative_to(WORKSPACE)), "frames": len(rows),
        "step_seconds": sum(float(r["step_wall_ms"]) for r in rows) / 1000,
        "newton_seconds": sum(float(r["newton_wall_ms"]) for r in rows) / 1000,
        "directions": sum(int(r[field]) for r in rows),
        "direction_count_histogram": dict(sorted(Counter(int(r[field]) for r in rows).items())),
        "max_directions_per_frame": max(int(r[field]) for r in rows),
        "environment": read_json(run / "environment.json"),
        "frame_csv_sha256": digest(run / "frames.csv"),
    }
    data["windows"] = {}
    for name, lo, hi in [("first21", 1, 21), ("frames22_30", 22, 30), ("first30", 1, 30)]:
        if hi <= len(rows):
            subset = rows[lo - 1:hi]
            data["windows"][name] = {
                "step_seconds": sum(float(r["step_wall_ms"]) for r in subset) / 1000,
                "newton_seconds": sum(float(r["newton_wall_ms"]) for r in subset) / 1000,
                "directions": sum(int(r[field]) for r in subset),
            }
    if arm == "robust2026":
        log = (run / "process.log").read_text(encoding="utf-8", errors="replace")
        iterations = [int(v) for v in re.findall(r"Iterative linear solver iteration count: (\d+)", log)]
        assert len(iterations) == data["directions"], (run, len(iterations), data["directions"])
        data["pcg_iterations_reported"] = sum(iterations)
        data["pcg_log_records"] = len(iterations)
        data["effective_config"] = read_json(run / "robust_requested_config.json")["config"]
        data["newton_cap_log_hits"] = log.count("Newton Iteration Exits with Max Iteration")
        data["linear_cap_log_hits"] = log.count("Linear Solver Exits with Max Iteration")
    else:
        data["effective_config"] = read_json(run / "effective_scene.json")
        data["newton_cap_hits"] = sum(int(r.get("newton_cap_hits", 0)) for r in rows)
    return data


def current_run(arm: str, repeat: int):
    run = TASK / "runs/local" / f"local_small_v29_default_cloth_sphere7_l_{arm}_measure_r{repeat:02d}"
    stats = read_json(run / "output/stats.json")["frames"]
    times = read_csv(run / "trace/frames.csv")
    result = read_json(run / "result.json")
    assert result["status"] == "completed" and len(stats) == len(times) == 100, run
    windows = {}
    for name, lo, hi in [("first21", 1, 21), ("frames22_30", 22, 30), ("first30", 1, 30),
                         ("frames31_100", 31, 100), ("all100", 1, 100)]:
        frames = stats[lo - 1:hi]
        directions = [n for f in frames for n in f["newton"] if "pcg" in n]
        earlier_directions = [n for f in stats[:lo - 1] for n in f["newton"] if "pcg" in n]
        outers = [o for f in frames for o in f.get("toi", [])]
        contacts = [o.get("active_self", 0) + o.get("active_ground", 0) for o in outers]
        alpha = [o["alpha"] for o in outers]
        window = {
            "frame_range": [lo, hi],
            "solver_seconds": sum(float(r["solver_ms"]) for r in times[lo - 1:hi]) / 1000,
            "directions": len(directions),
            "pcg_iterations": sum(n["pcg"]["iterations"] for n in directions),
            "pcg_limit_hits": sum(n["pcg"].get("iteration_limit", False) for n in directions),
            "toi_outer": len(outers),
            "inner_extra_beyond_one_per_outer": len(directions) - len(outers) if outers else 0,
            "outer_per_frame_histogram": dict(sorted(Counter(len(f.get("toi", [])) for f in frames).items())),
            "exit_reason_histogram": dict(Counter(f.get("toi_exit", "ipc") for f in frames)),
            "line_search_backtrack_directions": sum(n.get("line_search_backtracks", 0) > 0 for n in directions),
            "line_search_backtracks": sum(n.get("line_search_backtracks", 0) for n in directions),
            "backtrack_r_histogram": dict(Counter(str(n["line_search_r"]) for n in directions
                if n.get("line_search_backtracks", 0) > 0)),
            "max_inner_iterations": max((o["inner_iterations"] for o in outers), default=0),
            "max_active_contacts": max(contacts, default=0),
            "safe_ccd_broad_pairs_sum": sum(o.get("safe_ccd_broad_pairs", 0) for o in outers),
            "active_update_broad_pairs_sum": sum(o.get("active_update_broad_pairs", 0) for o in outers),
            "active_added_sum": sum(o.get("active_added", 0) for o in outers),
            "active_removed_sum": sum(o.get("active_removed", 0) for o in outers),
            "alpha_full_count": sum(a == 1 for a in alpha),
            "alpha_lt_1e_4_count": sum(a < 1e-4 for a in alpha),
            "alpha_median": statistics.median(alpha) if alpha else None,
            "penalty_escalations": sum("penalty_escalation" in o for o in outers),
            "native_geometric_contact_frames": sum(f.get("contact_geometry", {}).get("geometric_contact", False) for f in frames),
            "graph_captures": max((n["pcg"].get("graph_captures_total", 0) for n in directions), default=0)
                - max((n["pcg"].get("graph_captures_total", 0) for n in earlier_directions), default=0),
            "graph_hits": sum(n["pcg"].get("graph_cache_hit", False) for n in directions),
            "graph_capture_instantiate_host_ms_sum": sum(n["pcg"].get("graph_capture_instantiate_host_ms", 0) for n in directions),
        }
        # Graph field names are implementation-specific; retain their actual keys.
        window["pcg_record_keys"] = sorted(set().union(*(n["pcg"].keys() for n in directions)))
        windows[name] = window
    return {"path": str(run.relative_to(WORKSPACE)), "requested": read_json(run / "requested.json"),
            "result": result, "windows": windows, "stats_sha256": digest(run / "output/stats.json"),
            "frame_csv_sha256": digest(run / "trace/frames.csv")}


def main():
    old_runs = []
    old_summary = []
    for scene in ["cloth_sphere7_l", "cloth_table_l", "cloth_hang_l"]:
        base = [old_run(OLD / "runs/four-arm-contact-20260927", scene, "base", i) for i in (1, 2, 3)]
        robust = [old_run(OLD / "runs/four-arm-contact-20260927", scene, "robust2026", i) for i in (1, 2, 3)]
        tuned = old_run(OLD / "runs/four-arm-robust-veltol005-20260927", scene, "robust2026", 1)
        old_runs.extend(base + robust + [tuned])
        old_summary.append({"scene": scene, "frames": base[0]["frames"],
            "base_step_median_s": statistics.median(b["step_seconds"] for b in base),
            "robust_step_median_s": statistics.median(r["step_seconds"] for r in robust),
            "paired_step_ratio_median": statistics.median(b["step_seconds"] / r["step_seconds"] for b, r in zip(base, robust)),
            "base_directions_r01": base[0]["directions"],
            "robust_directions_r01": robust[0]["directions"],
            "robust_pcg_iterations_r01": robust[0]["pcg_iterations_reported"],
            "robust_tightened_step_s": tuned["step_seconds"],
            "robust_tightened_directions": tuned["directions"],
            "robust_tightened_pcg_iterations": tuned["pcg_iterations_reported"]})
    current = [current_run(arm, repeat) for arm in ("base", "base_graph", "base_toi", "base_toi_graph")
               for repeat in (1, 2, 3)]
    source_manifest = read_json(OLD / "runs/four-arm-contact-20260927/cloth_sphere7_l/robust2026/r01/source_files.json")
    selected = ["src/backends/cuda/engine/advance_al.cu", "src/backends/cuda/linear_system/linear_fused_pcg.cu",
                "src/backends/cuda/active_set_system/global_active_set_manager.cu"]
    source_checks = []
    by_path = {r["path"]: r for r in source_manifest}
    for rel in selected:
        path = WORKSPACE / "stiffGIPC/barrier-free/source" / rel
        expected = by_path["barrier-free/source/" + rel]["sha256"]
        source_checks.append({"path": str(path.relative_to(WORKSPACE)), "historical_sha256": expected,
                              "current_sha256": digest(path), "matches_historical_run": digest(path) == expected})
    summary = []
    for arm in ("base", "base_graph", "base_toi", "base_toi_graph"):
        runs = [r for r in current if r["requested"]["arm"] == arm]
        summary.append({"arm": arm, "solver100_median_s": statistics.median(r["windows"]["all100"]["solver_seconds"] for r in runs),
                        "solver30_median_s": statistics.median(r["windows"]["first30"]["solver_seconds"] for r in runs),
                        "windows_r01": runs[0]["windows"]})
    payload = {"schema": "robust-speed-gap-readonly-v1", "date": "2026-09-30",
               "scope": "Reanalysis of saved results only; no new simulation and no modification to donor repositories",
               "qualification": "Raw timing ratios; historical cross-program models/stops and current trajectory/physics gates are not matched",
               "historical_same_gpu_summary": old_summary, "current_summary": summary,
               "historical_source_checks": source_checks, "historical_runs": old_runs, "current_runs": current}
    first_base = current[0]["windows"]["all100"]
    payload["rough_work_budgets_first_repeat"] = []
    for arm in ("base_toi", "base_toi_graph"):
        measured = next(r for r in current if r["requested"]["arm"] == arm)["windows"]["all100"]
        mean_cost = measured["solver_seconds"] / measured["directions"]
        payload["rough_work_budgets_first_repeat"].append({
            "arm": arm, "entire_solver_window_seconds_per_direction": mean_cost,
            "break_even_directions_at_unchanged_mean_cost": first_base["solver_seconds"] / mean_cost,
            "twofold_directions_at_unchanged_mean_cost": first_base["solver_seconds"] / (2 * mean_cost),
            "scope": "Planning approximation including all solver overhead; not isolated PCG timing or predicted speedup"})
    output = TASK / "reports/robust_speed_gap_evidence_20260930.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"historical_same_gpu": old_summary, "current_summary": summary,
                      "historical_source_checks": source_checks, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
