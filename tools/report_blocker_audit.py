"""Summarize read-only safe-alpha blocker diagnostics; no speed qualification."""
from __future__ import annotations
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from report_velocity_ablation import States, work

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    values = [float(v) for v in values if v is not None]
    return ({"count": len(values), "min": min(values), "median": statistics.median(values),
             "p95": float(np.percentile(values, 95)), "max": max(values)} if values else None)


def subset_summary(rows):
    counts = collections.Counter(r["classification"] for r in rows)
    return {"count": len(rows), "classes": dict(counts),
            "class_fraction": {k: v / len(rows) for k, v in counts.items()},
            "ccd_alpha": distribution([r["ccd_alpha"] for r in rows]),
            "constraint_trial_over_delta": distribution([r["constraint_trial_m"] / r["delta_m"] for r in rows]),
            "negative_trial_constraints": sum(r["constraint_trial_m"] < -1e-10 for r in rows),
            "lambda_used": distribution([r["lambda_used"] for r in rows]),
            "gamma_used": distribution([r["gamma_used"] for r in rows]),
            "effective_mu": distribution([r["effective_mu"] for r in rows])}


def prefix(state, count):
    obj = copy.copy(state)
    obj.positions = obj.positions[:count + 1]
    obj.indices = obj.indices[:count + 1]
    obj.velocity = obj.velocity[:count]
    return obj


def trajectory_difference(a, b):
    assert a.dt == b.dt and np.array_equal(a.positions[0], b.positions[0])
    distance = np.linalg.norm(a.positions - b.positions, axis=2)
    changed = np.flatnonzero(distance.max(axis=1) > 1e-8)
    return {"max_vertex_difference_m": float(distance.max()),
            "final_vertex_difference_m": float(distance[-1].max()),
            "first_state_difference_over_1e_minus8_m": int(changed[0]) if len(changed) else None,
            "cloth_max_mass_rms_over_scale": float(a.rms(a.positions - b.positions, a.cloth).max() / a.cloth_scale),
            "scope": "One off/on pair; known same-backend trajectory variability prevents attributing this difference to audit sampling."}


def main():
    cfg = read(ROOT / "configs/local_blocker_v30_20261001.json")
    manifest = read(ROOT / "IMPLEMENTATION_MANIFEST.json")
    assert cfg["source_digest"] == manifest["source_digest"]
    assert sha(ROOT / "builds/local-fused-v30/Release/gipc.exe") == cfg["fused_sha256"]
    assert not [f["path"] for f in manifest["files"] if sha(ROOT / f["path"]) != f["sha256"]]
    matrix = ROOT / "runs/local" / cfg["name"]
    assert all(read(matrix / f"{s}_status.json")["status"] == "completed" for s in ["smoke", "pilot", "diagnose"])
    reference4 = States(ROOT / "runs/local/local_velocity_20261001_base_reference_r01")
    reference2 = States(ROOT / "runs/local/local_velocity_20261001_base_reference_half_r01")
    csv_rows, summaries, frame_rows = [], [], []
    states = {}
    for branch in cfg["branches"]:
        variant = branch["id"]
        run = ROOT / "runs/local" / f"local_v30_blocker_{variant}_audit60"
        request, result = read(run / "requested.json"), read(run / "result.json")
        assert request["source_digest"] == cfg["source_digest"] and request["exe_sha256"] == cfg["fused_sha256"]
        assert result["status"] == "completed" and result["finite"] and result["recorded_frames"] == 60
        assert result["timing_is_diagnostic"] and request["substeps"] and request["suite"] == "0"
        for key, expected in [("dt", .01), ("tol", .01), ("pcg_tol", .0001), ("mu_mode", "diagonal"),
                              ("persist_contacts", branch["persist_contacts"]), ("robust_velocity_tol", branch["velocity_tol"]),
                              ("robust_velocity_stop", branch["velocity_tol"] is not None)]:
            assert request[key] == expected, (variant, key)
        detail = work(run)
        frames = detail.pop("frames")
        first, all_candidates, representatives = [], [], {}
        consecutive_max, previous_key, streak = 0, None, 0
        for number, frame in enumerate(frames, 1):
            expected = 22 <= number <= 60
            assert frame["toi_blocker_audit_enabled"] == expected
            assert frame["toi_cross_frame_contacts"] == branch["persist_contacts"]
            assert frame["robust_trial_velocity_tol"] == (branch["velocity_tol"] or 0)
            previous_key, streak = None, 0
            limited_count = 0
            for outer in frame.get("toi", []):
                assert ("blocker_audit" in outer) == expected
                if not expected:
                    continue
                audit = outer["blocker_audit"]
                assert audit["ccd_alpha"] == outer["full_ccd_alpha"]
                if not audit["limited"]:
                    continue
                limited_count += 1
                assert audit["candidates"] and audit["candidates"][0]["is_global_minimum"]
                for order, c in enumerate(audit["candidates"]):
                    row = {"variant": variant, "frame": number, "outer": outer["outer"], "candidate_order": order,
                           "ccd_alpha": audit["ccd_alpha"], "accepted_alpha": outer["alpha"], "inner_iterations": outer["inner_iterations"],
                           **c}
                    assert row["plane_valid"]
                    csv_rows.append(row)
                    all_candidates.append(row)
                    if order == 0:
                        first.append(row)
                        representatives.setdefault(c["classification"], row)
                        key = (c["kind"], tuple(c["ids"]))
                        streak = streak + 1 if key == previous_key else 1
                        previous_key = key
                        consecutive_max = max(consecutive_max, streak)
            frame_rows.append({"variant": variant, "frame": number, "outer": len(frame.get("toi", [])),
                               "directions": sum("pcg" in n for n in frame["newton"]), "limited_outer": limited_count,
                               "previous_frame_active_count": frame["toi_previous_frame_active_count"]})
        state = states[variant] = States(run)
        state_summaries = {"all_first_min": subset_summary(first),
                           "first5_outer": subset_summary([r for r in first if r["outer"] < 5]),
                           "after5_outer": subset_summary([r for r in first if r["outer"] >= 5]),
                           "warm_frame_first_min": subset_summary([r for r in first if r["frame"] >= 23]),
                           "active_first_min": subset_summary([r for r in first if r["in_trial_system"]])}
        consistency = {key: distribution([r.get(key) for r in all_candidates]) for key in
                       ["host_probe_vs_solver_plane_max_abs_error", "multiplier_update_relative_error", "slack_update_abs_error_m", "gamma_update_abs_error"]}
        assert consistency["host_probe_vs_solver_plane_max_abs_error"]["max"] < 1e-9
        assert consistency["multiplier_update_relative_error"]["max"] < 1e-9
        assert consistency["slack_update_abs_error_m"]["max"] < 1e-9
        assert consistency["gamma_update_abs_error"]["max"] < 1e-9
        row = {"variant": variant, "run": run.name, "request": request, "result": result, "work": detail,
               "audited_outer": sum(len(f["toi"]) for f in frames[21:]), "alpha_limited_outer": len(first),
               "largest_consecutive_same_first_min_within_frame": consecutive_max,
               "first_min_unique_keys": len({(r["kind"], tuple(r["ids"])) for r in first}),
               "subsets": state_summaries, "all_top4_classes": dict(collections.Counter(r["classification"] for r in all_candidates)),
               "consistency": consistency, "representatives": representatives, "physics": state.physical(),
               "vs_refined_base_dt4": state.compare(reference4),
               "raw_stats_sha256": sha(run / "output/stats.json"), "trace_csv_sha256": sha(run / "trace/frames.csv")}
        ccd_path = OUT / f"v30_{variant}_accepted_ccd.json"
        if ccd_path.exists():
            validation = read(ccd_path)
            assert validation["passed"] and validation["finite"]
            assert validation["conservative_collision_flags"] == validation["tet_inversions"] == 0
            row["independent_accepted_ccd"] = validation
            row["independent_accepted_ccd_sha256"] = sha(ccd_path)
        summaries.append(row)
    for row in summaries:
        if row["variant"] != "full":
            row["vs_default_toi"] = states[row["variant"]].compare(states["full"])
    off_on = {part: trajectory_difference(States(ROOT / "runs/local" / f"local_v30_blocker_full_{part}_off"),
                                         States(ROOT / "runs/local" / f"local_v30_blocker_full_{part}_on"))
              for part in ["smoke2", "pilot30"]}
    payload = {"protocol": cfg, "runs": summaries, "frame_rows": frame_rows, "off_on": off_on,
               "refined_reference_dt2_vs_dt4_first_0_6_s": prefix(reference2, 120).compare(reference4),
               "source_files_verified": len(manifest["files"]),
               "audit_scope": "First candidate is one canonical global CCD argmin per limited outer iteration; tied minima may exist. Top four are sampled, not all contacts. Missing planes are diagnostic only. Audit times are not speed results.",
               "causal_scope": "One run per branch, known MAS/backend variability. Persist changes C/lambda/gamma and frame friction snapshots together. It cannot identify a C-only causal benefit.",
               "timing_quality_matched_speedup": None}
    (OUT / "TOI_BLOCKER_AUDIT_V30_20261001.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = ["variant", "frame", "outer", "candidate_order", "kind", "ids", "classification", "ccd_alpha", "accepted_alpha",
              "in_trial_system", "in_active_set_after_update", "constraint_safe_m", "constraint_trial_m", "delta_m", "effective_mu",
              "lambda_used", "lambda_after_update", "gamma_used", "inner_iterations", "plane_scope"]
    with (OUT / "TOI_BLOCKER_AUDIT_V30_20261001.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    for row in summaries:
        print(json.dumps({k: row[k] for k in ["variant", "work", "audited_outer", "alpha_limited_outer", "subsets", "consistency"]}))


if __name__ == "__main__":
    main()
