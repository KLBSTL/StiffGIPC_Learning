"""Prepare immutable contact-pool plans and analyze existing runs; never run GPU.

Stages: guards -> screen -> cost -> full. Full plans require all preceding gates.
Only requested new config/report files are written. No index or native edits.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

from config import ROOT, expand, read, sha
from ipc_benchmark import metrics, write
from component_tuning import state_comparison
from validate_run import validate

TAG = "ipc_contact_pool_20261006"
SCENES = {"hang": "cloth_hang_l", "fixed": "cloth_fixed_bunny_l", "mixed": "bunny_cloth_bunny_l"}
STEPS = {"hang": 51, "fixed": 59, "mixed": 35}
PHASES = ("assembly", "pcg", "ccd", "line_search", "state_update")
PROTOCOL = ROOT / "reports/active/ipc_revision_20261005_quality_protocol.json"
PROTOCOL_SHA = "1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f"
COST_LABELS = {"collision.contact_pool.prepare", "collision.contact_pool.guard",
               "collision.contact_pool.classify", "collision.discrete_query", "collision.swept_query"}
# This contract is checked explicitly. Missing telemetry is unknown, never zero.
POOL_SUM_FIELDS = ("attempts", "reused_queries", "old_queries", "validation_calls",
                   "validation_passed", "validation_failed", "pairs_compared",
                   "nonempty_validation_calls", "production_overflow_queries")
POOL_DETAIL_FIELDS = ("prepare_calls", "capture_passes", "generations", "pool_pairs", "vf_pairs", "ee_pairs")


def path_for(kind, stage):
    return ROOT / f"{kind}/active/{TAG}_{stage}{'_analysis' if kind == 'reports' else ''}.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def task(scene, variant, repeat, stage, steps, **changes):
    config = {"scene": SCENES[scene], "preset": "combined", "backend": "ipc",
              "execution": "conditional_graph", "mas": "legacy", "dt": .01,
              "steps": steps, "timeout_seconds": 120, "trace_velocity": True,
              "discrete_bvh_refit": True, "bounded_ccd": False,
              "bounded_ccd_validate": False, "bvh_eligibility": False,
              "bvh_eligibility_validate": False, "contact_pool": variant == "on",
              "contact_pool_validate": False, "ipc_termination": "legacy",
              "ipc_min_updates": 6, "ipc_newton_tol": .01,
              "ipc_cumulative_tol": .01, "pcg_rho_tol": 1e-4}
    binary = "active"
    if variant == "stiff":
        config.update(preset="base", execution="host", discrete_bvh_refit=False)
        binary = "base"
    config.update(changes)
    expand(config)
    return {"name": f"{TAG}_{stage}_{scene}_{variant}_{repeat}",
            "scene_key": scene, "variant": variant, "repeat": repeat,
            "arm": scene + "_" + variant, "binary": binary, "config": config}


def stage_tasks(stage):
    if stage == "guards":
        return [task("hang", "on", "guard", stage, 51, contact_pool_validate=True),
                task("fixed", "on", "guard", stage, 59, contact_pool_validate=True),
                task("mixed", "off", "guard", stage, 35),
                task("mixed", "on", "guard", stage, 35, contact_pool_validate=True)]
    if stage == "screen":
        return [task(s, v, f"r{i}", stage, STEPS[s])
                for i in range(1, 4)
                for s in (("hang", "fixed") if i % 2 else ("fixed", "hang"))
                for v in (("off", "on") if i % 2 else ("on", "off"))]
    if stage == "cost":
        return [task(s, v, "events", stage, frame + 1, diagnostics=["cost"],
                     cost_frames=str(frame), cost_events=True, profile="none")
                for s, frame in (("hang", 41), ("fixed", 49)) for v in ("off", "on")]
    if stage == "full":
        rows = []
        for i in range(1, 4):
            variants = ["stiff", "off", "on"]
            variants = variants[i-1:] + variants[:i-1]
            for scene in (("hang", "fixed") if i % 2 else ("fixed", "hang")):
                rows.extend(task(scene, v, f"r{i}", stage, 100) for v in variants)
        return rows
    raise ValueError(stage)


def prior_gate(stage, native):
    needed = {"guards": [], "screen": [], "cost": ["guards", "screen"],
              "full": ["guards", "screen", "cost"]}[stage]
    evidence = {}
    for previous in needed:
        path = path_for("reports", previous)
        report = read(path)
        require(report["native_identity"] == native, "Prerequisite native identity mismatch")
        require(report["quality_protocol_sha256"] == PROTOCOL_SHA, "Quality protocol mismatch")
        require(report["stage_gate_passed"], f"{previous} gate did not pass")
        if previous == "cost":
            require(report.get("full_performance_gate_passed"), "Cost report did not pass the complete prerequisite chain")
        require(sha(path_for("configs", previous)) == report["plan_sha256"], "Prerequisite plan changed")
        for source, digest in report["input_sha256"].items():
            require(sha(ROOT / source) == digest, "Prerequisite input changed: " + source)
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return evidence


def prepare(stage, native, dry_run=False):
    require(sha(PROTOCOL) == PROTOCOL_SHA, "Frozen quality protocol changed")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", native or "")), "Supply --native-sha256 with the frozen active executable hash")
    if not dry_run:
        require(sha(ROOT / "builds/active/Release/gipc.exe") == native, "Active executable differs from requested identity")
    prerequisites = prior_gate(stage, native) if not dry_run else {}
    runs = stage_tasks(stage)
    plan = {"schema": "contact_pool_plan.v1", "stage": stage,
            "report": f"reports/active/{TAG}_{stage}_batch.json", "stop_on_failure": True,
            "native_identity": native, "quality_protocol_sha256": PROTOCOL_SHA,
            "requires_before_execution": [] if stage == "guards" else ["completed same-identity guards with activation, typed validation and quality support"],
            "prerequisite_report_sha256": prerequisites,
            "finite_run_budget": len(runs), "gpu_execution_owner": "parent; serial only",
            "full_gate": "Three paired off/on medians >=1.05 for both whole prefixes and fixed contact windows, guards and material support, then cost-package evidence >=5% of each selected frame; no missing evidence allowed",
            "cost_semantics": "CUDA-event stream elapsed includes submission gaps; CPU separately. Select inclusive package roots by parent_scope_id; no nested addition.",
            "contact_windows": {"hang": [41, 50], "fixed": [22, 100 if stage == "full" else 59], "mixed": [24, 35]},
            "runs": runs}
    if not dry_run:
        write(path_for("configs", stage), plan)
    return {"stage": stage, "runs": len(runs), "native_identity": native,
            "dry_run": dry_run, "plan": plan if dry_run else str(path_for("configs", stage))}


def window(frames, csv_rows, low, high):
    require(high >= low and high <= len(frames), "Incomplete requested window")
    chosen = frames[low-1:high]
    times = {int(r["frame"]): float(r["solver_ms"]) for r in csv_rows}
    require(all(i in times for i in range(low, high+1)), "Missing frame timing")
    require(all(all(k in f.get("phase_ms", {}) for k in PHASES) for f in chosen), "Missing phase timing")
    phase = {k: sum(f["phase_ms"][k] for f in chosen) for k in PHASES}
    elapsed = sum(times[i] for i in range(low, high+1))
    terminal_available = all("ipc_exit_assembly_ms" in f for f in chosen)
    terminal = sum(f["ipc_exit_assembly_ms"] for f in chosen) if terminal_available else None
    pcgs = [n["pcg"] for f in chosen for n in f["newton"] if "pcg" in n]
    alphas = [n["alpha"] for f in chosen for n in f["newton"] if "alpha" in n]
    return {"frames": [low, high], "solver_ms": elapsed, "phase_ms": phase,
            "exit_assembly_ms": terminal,
            "unclassified_ms": elapsed-sum(phase.values())-(terminal or 0),
            "unclassified_includes_unreported_exit_assembly": not terminal_available,
            "directions": len(pcgs), "pcg_iterations": sum(p["iterations"] for p in pcgs),
            "pcg_iterations_per_direction": {"min": min((p["iterations"] for p in pcgs), default=None),
                "max": max((p["iterations"] for p in pcgs), default=None),
                "median": statistics.median(p["iterations"] for p in pcgs) if pcgs else None},
            "accepted_alpha_count": len(alphas), "alpha_sum": sum(alphas),
            "accepted_alphas_by_frame": [[n["alpha"] for n in f["newton"] if "alpha" in n] for f in chosen],
            "newton_exits": dict(collections.Counter(f.get("newton_exit") for f in chosen)),
            "native_candidate_active_frames": [low+i for i, f in enumerate(chosen)
                if f.get("contact_geometry", {}).get("native_narrow_self_pairs", 0)
                or f.get("contact_geometry", {}).get("native_narrow_ground_pairs", 0)],
            "newton_active_frames": [low+i for i, f in enumerate(chosen) if any(n.get("active_pairs", 0) > 0 for n in f["newton"])],
            "native_self_pair_peak": max((f.get("contact_geometry", {}).get("native_narrow_self_pairs", 0) for f in chosen), default=0),
            "native_ground_pair_peak": max((f.get("contact_geometry", {}).get("native_narrow_ground_pairs", 0) for f in chosen), default=0),
            "newton_active_pair_peak": max((n.get("active_pairs", 0) for f in chosen for n in f["newton"]), default=0),
            "physical_contact_certified": False}


def pool_evidence(frames, expected_on, validate_on):
    rows = [f.get("contact_pool") for f in frames]
    required = ("requested", "validate", *POOL_SUM_FIELDS, *POOL_DETAIL_FIELDS, "fallback_reasons", "pool_bytes_peak")
    missing = [{"frame": i+1, "fields": list(required) if not isinstance(row, dict)
                else [k for k in required if k not in row]} for i, row in enumerate(rows)
               if not isinstance(row, dict) or any(k not in row for k in required)]
    result = {"complete_contract": bool(rows) and not missing, "missing": missing, "passed": False,
              "same_state_energy_alpha_support": None}
    if missing or not rows:
        return result
    count_fields = (*POOL_SUM_FIELDS, *POOL_DETAIL_FIELDS, "pool_bytes_peak")
    require(all(type(r[k]) is int and r[k] >= 0 for r in rows for k in count_fields), "Invalid contact-pool counters")
    sums = {k: sum(r[k] for r in rows) for k in (*POOL_SUM_FIELDS, *POOL_DETAIL_FIELDS)}
    reasons = collections.Counter()
    for r in rows:
        reasons.update(r["fallback_reasons"])
    result.update(totals=sums, fallback_reasons=dict(reasons),
                  pool_bytes_peak=max((r["pool_bytes_peak"] for r in rows), default=0),
                  raw_frame_evidence=rows,
                  production_overflow_scope="Measured only by validation reference.count > caller capacity; when validate=false this is not the total production overflow count")
    flags = all(r["requested"] == expected_on and r["validate"] == validate_on for r in rows)
    valid = sums["validation_failed"] == 0
    if expected_on:
        valid &= sums["reused_queries"] > 0
    else:
        valid &= sums["reused_queries"] == 0
    if validate_on:
        valid &= (sums["validation_calls"] > 0 and sums["validation_calls"] == sums["validation_passed"]
                  and sums["pairs_compared"] > 0 and sums["nonempty_validation_calls"] > 0)
        valid &= sums["validation_calls"] == sums["reused_queries"]
        energies = [n for f in frames for entry in f.get("newton", [])
                    for n in entry.get("contact_pool_energy", [])]
        energy_frames = [r.get("energy_audit") for r in rows]
        energy_contract = all(isinstance(r, dict) and all(k in r for k in
            ("calls", "failed", "max_relative_error", "max_barrier_relative_error")) for r in energy_frames)
        energy_ok = bool(energy_contract and energies)
        if energy_contract:
            expected_energy = sums["validation_passed"]-sums["production_overflow_queries"]
            energy_ok &= 0 < expected_energy and sum(r["calls"] for r in energy_frames) == len(energies) == expected_energy
            energy_ok &= sum(r["failed"] for r in energy_frames) == 0
            energy_ok &= all(math.isfinite(r[k]) and 0 <= r[k] <= 1e-10 for r in energy_frames
                             for k in ("max_relative_error", "max_barrier_relative_error"))
            for frame, counts, audit in zip(frames, rows, energy_frames):
                frame_energy = [e for n in frame.get("newton", []) for e in n.get("contact_pool_energy", [])]
                energy_ok &= (audit["calls"] == len(frame_energy) == counts["validation_passed"]-counts["production_overflow_queries"])
        keys = ("alpha", "candidate_energy", "legacy_energy", "relative_error", "barrier_relative_error",
                "armijo_branch_equal", "production_restored", "passed")
        for e in energies:
            energy_ok &= all(k in e for k in keys)
            if all(k in e for k in keys):
                energy_ok &= all(isinstance(e[k], (int, float)) and math.isfinite(e[k]) for k in keys[:5])
                energy_ok &= e["passed"] and e["armijo_branch_equal"] and e["production_restored"]
                energy_ok &= abs(e["relative_error"]) <= 1e-10 and abs(e["barrier_relative_error"]) <= 1e-10
        result["same_state_energy_alpha_support"] = {"passed": bool(energy_ok), "records": energies,
            "frame_summaries": energy_frames, "overflow_queries_excluded": sums["production_overflow_queries"],
            "expected_energy_calls": sums["validation_passed"]-sums["production_overflow_queries"],
            "tolerance": 1e-10, "tolerance_is_physical_relaxation": False}
        valid &= energy_ok
    result["passed"] = bool(flags and valid)
    return result


def velocity_evidence(run, steps):
    return export_evidence(run, steps, "velocity")


def export_evidence(run, steps, kind):
    topology = np.fromfile(run / "trace/topology.bin", dtype="<u4")
    require(len(topology) >= 3, "Topology header missing")
    vertices = int(topology[0])
    expected = [f"{kind}_{i:04d}.bin" for i in range(steps+1)]
    actual = sorted(p.name for p in (run / "trace").glob(kind+"_*.bin"))
    problems = []
    for name in expected:
        path = run / "trace" / name
        if not path.exists():
            problems.append({"file": name, "reason": "missing"})
            continue
        if path.stat().st_size != vertices*24:
            problems.append({"file": name, "reason": "wrong_byte_count", "actual": path.stat().st_size, "expected": vertices*24})
        elif not np.isfinite(np.fromfile(path, dtype="<f8")).all():
            problems.append({"file": name, "reason": "nonfinite"})
    return {"available": bool(actual), "passed": actual == expected and not problems,
            "expected_frames": steps+1, "actual_frames": len(actual), "vertices": vertices,
            "expected_bytes_per_frame": vertices*24, "problems": problems}


def material_window(rows, low, high):
    selected = rows[low-1:high]
    def minimum(key):
        values = [r[key] for r in selected if r[key] is not None]
        return min(values) if values else None
    return {"frames": [low, high], "max_stretch": max(r["max_stretch"] for r in selected),
            "p99_stretch": max(r["p99_stretch"] for r in selected),
            "fixed_drift_m": max(r["fixed_drift_m"] for r in selected),
            "fem_min_J": minimum("fem_min_J"), "abd_min_J": minimum("abd_min_J"),
            "fem_nonpositive_sum": sum(r["fem_nonpositive"] for r in selected),
            "fem_negative_volume_peak": max(r["fem_negative_volume"] for r in selected)}


def initial_inputs(a, b):
    """The off/on execution flags may differ; physical inputs may not."""
    files = ("trace/topology.bin", "trace/masses.bin", "trace/boundary_types.bin", "trace/body_ids.bin",
             "trace/state_0000.bin", "trace/velocity_0000.bin", "trace/metadata.json", "output/scene.json")
    evidence = {}
    for name in files:
        pa, pb = a / name, b / name
        present = pa.exists() and pb.exists()
        left, right = (sha(pa), sha(pb)) if present else (None, None)
        evidence[name] = {"off_sha256": left, "on_sha256": right, "equal": present and left == right}
    return {"passed": all(r["equal"] for r in evidence.values()), "files": evidence}


def package_cost(rows):
    """Choose disjoint semantic roots, not a sum of inclusive child scopes."""
    ids = {r["scope_id"]: r for r in rows}
    require(len(ids) == len(rows), "Duplicate cost scope IDs")
    selected = {i for i, r in ids.items() if r["stage"] in COST_LABELS}
    require(all(isinstance(ids[i].get("cpu_submit_ms"), (int, float))
                and math.isfinite(ids[i]["cpu_submit_ms"]) and ids[i]["cpu_submit_ms"] >= 0 for i in selected),
            "Invalid CPU cost interval")
    require(all(ids[i].get("gpu_interval_ms") is None or (isinstance(ids[i]["gpu_interval_ms"], (int, float))
                and math.isfinite(ids[i]["gpu_interval_ms"]) and ids[i]["gpu_interval_ms"] >= 0) for i in selected),
            "Invalid GPU cost interval")
    roots = []
    for i in sorted(selected):
        parent = ids[i]["parent_scope_id"]
        visited = {i}
        nested = False
        while parent:
            require(parent not in visited, "Cyclic cost parent chain")
            visited.add(parent)
            require(parent in ids, "Missing cost parent")
            if parent in selected:
                nested = True
            parent = ids[parent]["parent_scope_id"]
        if not nested:
            roots.append(ids[i])
    available = bool(roots) and all(r.get("gpu_interval_ms") is not None for r in roots)
    return {"available": available, "selected_scopes": len(selected), "inclusive_roots": len(roots),
            "root_scope_ids": [r["scope_id"] for r in roots],
            "root_labels": dict(collections.Counter(r["stage"] for r in roots)),
            "gpu_stream_elapsed_ms": sum(r["gpu_interval_ms"] for r in roots) if available else None,
            "cpu_submit_ms_including_waits": sum(r["cpu_submit_ms"] for r in roots),
            "labels_observed": dict(collections.Counter(ids[i]["stage"] for i in selected)),
            "inclusive_by_label_not_additive": {label: {
                "calls": sum(ids[i]["stage"] == label for i in selected),
                "gpu_stream_elapsed_ms": sum(ids[i]["gpu_interval_ms"] for i in selected if ids[i]["stage"] == label)
                    if all(ids[i].get("gpu_interval_ms") is not None for i in selected if ids[i]["stage"] == label) else None,
                "cpu_submit_ms_including_waits": sum(ids[i]["cpu_submit_ms"] for i in selected if ids[i]["stage"] == label)}
                for label in sorted({ids[i]["stage"] for i in selected})},
            "measurement": "Inclusive root CUDA-event stream elapsed, including launch gaps; not pure GPU busy time or a critical-path bound. CPU cannot be added."}


def read_cost(run, expected_frame):
    path = run / "cost.jsonl"
    if not path.exists():
        return {"available": False, "reason": "cost.jsonl missing"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selected = [r for r in rows if r.get("frame") == expected_frame]
    frames = [r for r in selected if r["stage"] == "ipc.physical_frame"]
    require(len(frames) == 1, "Expected exactly one cost physical frame")
    require(all(r.get("gpu_events_enabled") and not r.get("operator_probe_enabled") for r in selected), "Expected events mode without operator probe")
    result = package_cost(selected)
    result.update(frame=expected_frame, physical_frame_gpu_stream_elapsed_ms=frames[0].get("gpu_interval_ms"),
                  cost_sha256=sha(path))
    return result


def analyze_run(t, plan):
    run = ROOT / "runs/active" / t["name"]
    rec = {k: t[k] for k in ("name", "scene_key", "variant", "repeat", "binary")}
    rec.update(status="result_unavailable", recorded_frames=0,
               hard_checks_passed=False, quality_support_passed=False, missing_or_failures=[])
    failures = rec["missing_or_failures"]
    try:
        result = read(run / "result.json")
        req = read(run / "requested.json")
        rec.update(status=result["status"], recorded_frames=result.get("recorded_frames", 0))
    except (ValueError, KeyError, FileNotFoundError) as error:
        failures.append(type(error).__name__ + ": " + str(error))
        return rec
    if req.get("expanded_config") != expand(t["config"]):
        failures.append("requested_config_differs_from_plan")
    if t["binary"] == "active" and req.get("exe_sha256") != plan["native_identity"]:
        failures.append("native_identity_mismatch")
    if result["status"] != "completed" or rec["recorded_frames"] != t["config"]["steps"]:
        failures.append("run_incomplete")
    if not rec["recorded_frames"]:
        return rec
    try:
        frames = read(run / "output/stats.json")["frames"]
        with (run / "trace/frames.csv").open(encoding="utf-8") as stream:
            times = list(csv.DictReader(stream))
        require(len(frames) == rec["recorded_frames"], "Stats/result frame mismatch")
        rec["whole"] = window(frames, times, 1, len(frames))
        rec["actual_positions"] = export_evidence(run, len(frames), "state")
        if not rec["actual_positions"]["passed"]:
            failures.append("position_exports_incomplete_or_nonfinite")
        low, high = plan["contact_windows"][t["scene_key"]]
        rec["selected_window"] = window(frames, times, low, min(high, len(frames))) if len(frames) >= low else None
        m = metrics(run)
        rec["metrics"] = {k: m[k] for k in ("max_stretch", "p99_stretch", "fixed_drift_m", "finite", "directions", "pcg", "pcg_failures", "velocity_frames", "exits")}
        rec["tetrahedra_checks"] = {"fem_nonpositive": sum(x["fem_nonpositive"] for x in m["frames"]),
            "abd_nonpositive_frames": sum(x["abd_min_J"] is not None and x["abd_min_J"] <= 0 for x in m["frames"])}
        if not m["finite"] or m["pcg_failures"] or rec["tetrahedra_checks"]["abd_nonpositive_frames"] or (t["scene_key"] != "mixed" and rec["tetrahedra_checks"]["fem_nonpositive"]):
            failures.append("finite_pcg_or_tetrahedra_guard")
        rec["material_windows"] = {"whole": material_window(m["frames"], 1, len(frames))}
        rec["material_by_frame"] = m["frames"]
        if rec["selected_window"]:
            rec["material_windows"]["selected"] = material_window(m["frames"], low, min(high, len(frames)))
        extra_windows = {"hang": {"late": (41, len(frames))}, "fixed": {"dense": (45, 59)},
                         "mixed": {"initial": (1, 3), "middle": (24, 26), "late": (33, 35)}}[t["scene_key"]]
        rec["extra_windows"] = {k: {"work": window(frames, times, lo, hi), "material": material_window(m["frames"], lo, hi)}
                                for k, (lo, hi) in extra_windows.items() if len(frames) >= hi >= lo}
        if any(not math.isfinite(n["alpha"]) for f in frames for n in f["newton"] if "alpha" in n):
            failures.append("nonfinite_accepted_alpha")
        if t["binary"] == "active":
            rec["configuration"] = validate(run)
            resolved = read(run / "resolved_config.json").get("report_components", {})
            cfg = req["expanded_config"]
            if not rec["configuration"]["passed"] or any(resolved.get(k) != cfg[k] for k in ("contact_pool", "contact_pool_validate")):
                failures.append("configuration_execution_guard")
            rec["pool"] = pool_evidence(frames, cfg["contact_pool"], cfg["contact_pool_validate"])
            if not rec["pool"]["passed"]:
                failures.append("pool_contract_or_activation_guard")
            if t["scene_key"] == "mixed" and cfg["contact_pool"]:
                rec["mixed_window_coverage"] = {}
                for name, lo, hi in (("middle", 24, 26), ("late", 33, 35)):
                    portion = frames[lo-1:hi]
                    covered = len(portion) == hi-lo+1 and any(
                        f.get("contact_geometry", {}).get("native_narrow_self_pairs", 0)
                        or f.get("contact_geometry", {}).get("native_narrow_ground_pairs", 0)
                        or any(n.get("active_pairs", 0) for n in f["newton"]) for f in portion)
                    reused = sum(f.get("contact_pool", {}).get("reused_queries", 0) for f in portion)
                    rec["mixed_window_coverage"][name] = {"active_pairs_observed": bool(covered), "reused_queries": reused}
                    if not covered or not reused:
                        failures.append("mixed_" + name + "_coverage_missing")
        key = {"fixed": "fixed_bunny"}.get(t["scene_key"], t["scene_key"])
        bounds = read(PROTOCOL)["scenes"].get(key, {}).get("bounds")
        rec["material_checks"] = {k: {"actual": m[k], "bound": v, "passed": m[k] <= v} for k, v in bounds.items()} if bounds else None
        rec["material_scope"] = "Frozen cloth limits applied to exported prefix" if bounds else "Mixed compatibility: compare FEM metrics to same-program off; no absolute zero-inversion requirement invented"
        material_ok = all(v["passed"] for v in rec["material_checks"].values()) if bounds else t["scene_key"] == "mixed"
        if not material_ok:
            failures.append("frozen_material_bounds")
        if t["binary"] == "active":
            rec["actual_velocity"] = velocity_evidence(run, len(frames))
            if not rec["actual_velocity"]["passed"]:
                failures.append("actual_velocity_exports_incomplete_or_nonfinite")
        else:
            rec["actual_velocity"] = {"available": False, "passed": None,
                "reason": "Frozen Stiff lacks actual velocity export; position differences are never substituted"}
        rec["hard_checks_passed"] = not failures
        rec["quality_support_passed"] = bool(not failures and material_ok)
        if t["scene_key"] == "mixed":
            rec["quality_support_passed"] = None
            rec["quality_support_status"] = "pending: no frozen mixed FEM/cloth difference tolerance; same-program comparison reported separately"
        if plan["stage"] == "cost":
            rec["query_package"] = read_cost(run, int(t["config"]["cost_frames"]))
            frame = int(t["config"]["cost_frames"])
            rec["cost_frame_work"] = window(frames, times, frame, frame)
            rec["cost_frame_work"]["contact_pool"] = frames[frame-1].get("contact_pool")
    except (ValueError, KeyError, FileNotFoundError, AssertionError, IndexError) as error:
        failures.append(type(error).__name__ + ": " + str(error))
    rec["hard_checks_passed"] = not failures
    if rec["quality_support_passed"] is not None:
        rec["quality_support_passed"] = rec["quality_support_passed"] and not failures
    return rec


def mixed_guard_comparison(records):
    pair = {r["variant"]: r for r in records if r["scene_key"] == "mixed"}
    result = {"available": False, "bounded_compatibility_support": False,
              "quality_support_passed": None, "quality_support_status": "pending: no frozen mixed difference tolerance",
              "single_pair_establishes_repeatability": False}
    if set(pair) != {"off", "on"} or not all(r["hard_checks_passed"] for r in pair.values()):
        result["reason"] = "Both complete hard-guarded same-program mixed trajectories are required"
        return result
    a, b = (ROOT / "runs/active" / pair[v]["name"] for v in ("off", "on"))
    result["initial_inputs"] = initial_inputs(a, b)
    if not result["initial_inputs"]["passed"]:
        result["reason"] = "Physical inputs differ"
        return result
    result["state_difference"] = state_comparison(a, b)
    comparisons = []
    for old, new in zip(pair["off"]["material_by_frame"], pair["on"]["material_by_frame"]):
        require(old["frame"] == new["frame"], "Mixed frame mismatch")
        fields = {}
        for key in ("max_stretch", "p99_stretch", "fixed_drift_m", "fem_min_J", "fem_nonpositive", "fem_negative_volume", "abd_min_J"):
            x, y = old[key], new[key]
            fields[key] = {"off": x, "on": y, "on_minus_off": y-x if x is not None and y is not None else None,
                "worsening_observed": (y < x if key in ("fem_min_J", "abd_min_J") else y > x)
                    if x is not None and y is not None else None}
        comparisons.append({"frame": old["frame"], "fields": fields})
    worsening = any(v["worsening_observed"] for row in comparisons for v in row["fields"].values())
    result.update(available=True, material_by_frame=comparisons, worsening_observed=worsening,
                  bounded_compatibility_support=not worsening,
                  interpretation="Any observed increase is retained as pending support and blocks automatic progression; exact non-worsening supports only this bounded mixed compatibility check, never mixed quality certification.")
    return result


def summarize_pairs(records, stage):
    summary = {}
    for scene in ("hang", "fixed"):
        pairs = []
        for repeat in (["events"] if stage == "cost" else ["r1", "r2", "r3"]):
            selected = [r for r in records if r["scene_key"] == scene and r["repeat"] == repeat]
            off = next((r for r in selected if r["variant"] == "off"), None)
            on = next((r for r in selected if r["variant"] == "on"), None)
            if not off or not on or not off.get("whole") or not on.get("whole"):
                continue
            pair = {"repeat": repeat, "off": off["name"], "on": on["name"],
                    "whole_speedup": off["whole"]["solver_ms"] / on["whole"]["solver_ms"],
                    "selected_speedup": off["selected_window"]["solver_ms"] / on["selected_window"]["solver_ms"] if off.get("selected_window") and on.get("selected_window") else None,
                    "pcg_off": off["whole"]["pcg_iterations"], "pcg_on": on["whole"]["pcg_iterations"],
                    "directions_off": off["whole"]["directions"], "directions_on": on["whole"]["directions"],
                    "checks_passed": all(r["hard_checks_passed"] and r["quality_support_passed"] for r in (off, on))}
            pair["initial_inputs"] = initial_inputs(ROOT / "runs/active" / off["name"], ROOT / "runs/active" / on["name"])
            pair["checks_passed"] &= pair["initial_inputs"]["passed"]
            if stage != "cost" and pair["checks_passed"]:
                try:
                    pair["state_difference"] = state_comparison(ROOT / "runs/active" / off["name"], ROOT / "runs/active" / on["name"])
                except (ValueError, AssertionError, FileNotFoundError) as error:
                    pair["state_difference_error"] = str(error)
                    pair["checks_passed"] = False
            if stage == "cost":
                a, b = off.get("query_package", {}), on.get("query_package", {})
                available = a.get("available", False) and b.get("available", False)
                maintenance = all(label in b.get("labels_observed", {}) for label in COST_LABELS if label != "collision.discrete_query")
                wall = a.get("physical_frame_gpu_stream_elapsed_ms")
                saved = a["gpu_stream_elapsed_ms"]-b["gpu_stream_elapsed_ms"] if available else None
                fraction = saved/wall if available and wall and wall > 0 else None
                pair["net_query_cost"] = {"available": bool(available and maintenance), "off": a, "on": b,
                    "net_saved_stream_ms": saved, "fraction_of_off_selected_frame": fraction,
                    "same_frozen_state": False, "pure_gpu_busy_time": False,
                    "selected_frame_work": {"off": off.get("cost_frame_work"), "on": on.get("cost_frame_work")},
                    "includes_swept_capture_writes": True,
                    "interpretation": "Two diagnostic trajectories, inclusive disjoint-root stream timing. Work differences are reported; not a frozen-operator causal speedup."}
                pair["cost_gate_passed"] = bool(pair["checks_passed"] and available and maintenance and fraction is not None and fraction >= .05)
            pairs.append(pair)
        whole = [p["whole_speedup"] for p in pairs]
        contact = [p["selected_speedup"] for p in pairs if p["selected_speedup"] is not None]
        summary[scene] = {"pairs": pairs, "median_whole_speedup": statistics.median(whole) if whole else None,
                          "median_selected_speedup": statistics.median(contact) if contact else None,
                          "paired_checks_passed": bool(pairs) and all(p["checks_passed"] for p in pairs)}
        if stage in ("screen", "full"):
            repeats = {}
            for variant in ("off", "on"):
                arm = sorted((r for r in records if r["scene_key"] == scene and r["variant"] == variant
                              and r["hard_checks_passed"]), key=lambda r: r["repeat"])
                comparisons = []
                for i, left in enumerate(arm):
                    for right in arm[i+1:]:
                        a, b = ROOT / "runs/active" / left["name"], ROOT / "runs/active" / right["name"]
                        entry = {"left": left["name"], "right": right["name"], "initial_inputs": initial_inputs(a, b)}
                        try:
                            require(entry["initial_inputs"]["passed"], "Within-arm physical input mismatch")
                            entry["state_difference"] = state_comparison(a, b)
                        except (ValueError, AssertionError, FileNotFoundError) as error:
                            entry["error"] = str(error)
                        comparisons.append(entry)
                repeats[variant] = comparisons
            summary[scene]["within_arm_state_differences"] = repeats
            summary[scene]["trajectory_interpretation"] = "Report cloth/FEM/ABD separately against off repetition; no new position/velocity equivalence tolerance is introduced. Independent quality certification remains pending."
        if stage == "full":
            for p in pairs:
                stiff = next((r for r in records if r["scene_key"] == scene and r["variant"] == "stiff" and r["repeat"] == p["repeat"]), None)
                if stiff and stiff.get("whole"):
                    off = next(r for r in records if r["name"] == p["off"])
                    on = next(r for r in records if r["name"] == p["on"])
                    p["whole_vs_stiff"] = {"off": stiff["whole"]["solver_ms"]/off["whole"]["solver_ms"],
                                           "on": stiff["whole"]["solver_ms"]/on["whole"]["solver_ms"]}
                    p["selected_vs_stiff"] = {"off": stiff["selected_window"]["solver_ms"]/off["selected_window"]["solver_ms"],
                                              "on": stiff["selected_window"]["solver_ms"]/on["selected_window"]["solver_ms"]}
            for field in ("whole_vs_stiff", "selected_vs_stiff"):
                summary[scene]["median_"+field] = {v: statistics.median(p[field][v] for p in pairs if field in p)
                    for v in ("off", "on")} if all(field in p for p in pairs) and pairs else None
    return summary


def analyze(stage):
    require(sha(PROTOCOL) == PROTOCOL_SHA, "Frozen quality protocol changed")
    plan_path = path_for("configs", stage)
    plan = read(plan_path)
    batch_path = ROOT / plan["report"]
    batch = read(batch_path)
    require(batch.get("plan_sha256") == sha(plan_path), "Batch plan identity mismatch")
    tasks = {t["name"]: t for t in plan["runs"]}
    names = [t["name"] for t in batch["runs"]]
    require(len(set(names)) == len(names) and set(names) <= set(tasks), "Unexpected or duplicate batch run")
    records = [analyze_run(tasks[n], plan) for n in names]
    inputs = {plan_path.relative_to(ROOT).as_posix(): sha(plan_path), batch_path.relative_to(ROOT).as_posix(): sha(batch_path)}
    for name in names:
        folder = ROOT / "runs/active" / name
        for relative in ("requested.json", "resolved_config.json", "result.json", "output/stats.json", "output/scene.json",
                         "trace/frames.csv", "trace/topology.bin", "trace/masses.bin", "trace/boundary_types.bin",
                         "trace/body_ids.bin", "trace/metadata.json", "cost.jsonl"):
            source = folder / relative
            if source.exists():
                inputs[source.relative_to(ROOT).as_posix()] = sha(source)
        for pattern in ("state_*.bin", "velocity_*.bin"):
            for source in sorted((folder / "trace").glob(pattern)):
                inputs[source.relative_to(ROOT).as_posix()] = sha(source)
    complete = len(records) == len(tasks) and all(r["hard_checks_passed"] and
        (r["quality_support_passed"] or r["scene_key"] == "mixed") for r in records)
    summary = summarize_pairs(records, stage) if stage != "guards" else {}
    gate = complete
    if stage == "guards":
        try:
            summary["mixed_compatibility"] = mixed_guard_comparison(records)
        except (ValueError, AssertionError, FileNotFoundError) as error:
            summary["mixed_compatibility"] = {"bounded_compatibility_support": False, "error": str(error)}
        gate &= summary["mixed_compatibility"]["bounded_compatibility_support"]
    if stage in ("screen", "full"):
        gate &= all(len(s["pairs"]) == 3 and s["paired_checks_passed"]
                    and s["median_whole_speedup"] >= 1.05 and s["median_selected_speedup"] >= 1.05 for s in summary.values())
    if stage == "cost":
        gate &= all(len(s["pairs"]) == 1 and s["pairs"][0]["cost_gate_passed"] for s in summary.values())
    report = {"schema": "contact_pool_analysis.v1", "stage": stage,
              "native_identity": plan["native_identity"], "plan_sha256": sha(plan_path),
              "analyzer_sha256": sha(Path(__file__)),
              "quality_protocol_sha256": PROTOCOL_SHA, "input_sha256": inputs,
              "planned": len(tasks), "attempted": len(records), "unattempted": sorted(set(tasks)-set(names)),
              "runs": records, "summary": summary, "stage_gate_passed": bool(gate),
              "full_performance_gate_passed": False,
              "quality_scope": "Frozen material limits, exported states/actual velocities and same-state pool guards support a bounded experiment, not independent accepted-path CCD certification.",
              "performance_certified": False, "quality_certified": False, "default_promoted": False,
              "no_gpu_launched_by_analyzer": True}
    if stage == "cost" and gate:
        try:
            report["full_gate_prerequisites"] = prior_gate("cost", plan["native_identity"])
            report["full_performance_gate_passed"] = True
        except (ValueError, FileNotFoundError, KeyError) as error:
            report["full_gate_error"] = str(error)
    require(sha(PROTOCOL) == PROTOCOL_SHA, "Frozen quality protocol changed during analysis")
    write(path_for("reports", stage), report)
    return {"stage": stage, "attempted": len(records), "unattempted": report["unattempted"],
            "stage_gate_passed": report["stage_gate_passed"],
            "full_performance_gate_passed": report["full_performance_gate_passed"],
            "report": str(path_for("reports", stage))}


def self_test():
    assert [len(stage_tasks(s)) for s in ("guards", "screen", "cost", "full")] == [4, 12, 4, 18]
    assert stage_tasks("guards")[-1]["config"]["steps"] == 35
    assert all(not expand(t["config"])["bounded_ccd"] for s in ("guards", "screen", "cost", "full") for t in stage_tasks(s))
    def row(i, parent, stage, gpu):
        return {"scope_id": i, "parent_scope_id": parent, "stage": stage,
                "gpu_interval_ms": gpu, "cpu_submit_ms": 1}
    nested = [row(1, 0, "ipc.physical_frame", 20), row(2, 1, "collision.discrete_query", 8),
              row(3, 2, "collision.contact_pool.classify", 4), row(4, 1, "collision.contact_pool.prepare", 2)]
    result = package_cost(nested)
    assert result["gpu_stream_elapsed_ms"] == 10 and result["root_scope_ids"] == [2, 4]
    nested[-1]["gpu_interval_ms"] = None
    assert not package_cost(nested)["available"]
    assert not pool_evidence([{}], True, True)["passed"]
    nested[-1]["gpu_interval_ms"] = 2
    nested.append(row(5, 1, "collision.swept_query", 3))
    assert package_cost(nested)["gpu_stream_elapsed_ms"] == 13
    cycle = [row(1, 2, "collision.discrete_query", 1), row(2, 1, "collision.contact_pool.guard", 1)]
    try:
        package_cost(cycle)
        raise AssertionError("Cycle must be rejected")
    except ValueError:
        pass
    counters = {k: 0 for k in (*POOL_SUM_FIELDS, *POOL_DETAIL_FIELDS)}
    counters.update(requested=True, validate=True, fallback_reasons={}, pool_bytes_peak=64,
                    attempts=2, reused_queries=2, validation_calls=2, validation_passed=2,
                    pairs_compared=4, nonempty_validation_calls=2, production_overflow_queries=1,
                    energy_audit={"calls": 1, "failed": 0, "max_relative_error": 0, "max_barrier_relative_error": 0})
    energy = {"alpha": .5, "candidate_energy": 1., "legacy_energy": 1., "relative_error": 0.,
              "barrier_relative_error": 0., "armijo_branch_equal": True, "production_restored": True, "passed": True}
    frame = {"contact_pool": counters, "newton": [{"contact_pool_energy": [energy]}]}
    assert pool_evidence([frame], True, True)["passed"]
    counters["production_overflow_queries"] = 0
    assert not pool_evidence([frame], True, True)["passed"]
    counters["production_overflow_queries"] = 1
    energy["production_restored"] = False
    assert not pool_evidence([frame], True, True)["passed"]
    energy["production_restored"] = True
    frame["newton"] = []
    assert not pool_evidence([frame], True, True)["passed"]
    assert not mixed_guard_comparison([])["bounded_compatibility_support"]
    import tempfile
    with tempfile.TemporaryDirectory(prefix="contact_pool_cpu_") as temp:
        run = Path(temp)
        (run / "trace").mkdir()
        np.array([2, 0, 0], dtype="<u4").tofile(run / "trace/topology.bin")
        for i in range(2):
            np.zeros(6, dtype="<f8").tofile(run / f"trace/velocity_{i:04d}.bin")
        assert velocity_evidence(run, 1)["passed"]
        np.array([float("nan")]*6, dtype="<f8").tofile(run / "trace/velocity_0001.bin")
        assert not velocity_evidence(run, 1)["passed"]
        np.zeros(3, dtype="<f8").tofile(run / "trace/velocity_0001.bin")
        assert not velocity_evidence(run, 1)["passed"]
    print(json.dumps({"self_test": "passed", "GPU_runs": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "analyze", "self-test"))
    parser.add_argument("--stage", choices=("guards", "screen", "cost", "full"))
    parser.add_argument("--native-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
        return
    if args.stage is None:
        parser.error("--stage is required")
    result = prepare(args.stage, args.native_sha256, args.dry_run) if args.action == "prepare" else analyze(args.stage)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
