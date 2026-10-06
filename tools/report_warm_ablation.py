"""Analyze effective warm histories and blocker/cloth diagnostics for v31."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from report_velocity_ablation import States, work
from report_blocker_audit import distribution, prefix, subset_summary

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "diagnose"], default="diagnose")
    args = parser.parse_args()
    cfg = read(ROOT / "configs/local_warm_v31_20261001.json")
    manifest = read(ROOT / "IMPLEMENTATION_MANIFEST.json")
    assert manifest["source_digest"] == cfg["source_digest"]
    assert sha(ROOT / "builds/local-fused-v31/Release/gipc.exe") == cfg["fused_sha256"]
    assert all(sha(ROOT / f["path"]) == f["sha256"] for f in manifest["files"])
    matrix = ROOT / "runs/local" / cfg["name"]
    status = read(matrix / f"{args.stage}_status.json")
    assert status["status"] in ["completed", "completed_with_failures"]
    expected_steps = 30 if args.stage == "pilot" else 60
    reference4 = States(ROOT / "runs/local/local_velocity_20261001_base_reference_r01")
    reference2 = States(ROOT / "runs/local/local_velocity_20261001_base_reference_half_r01")
    summaries, candidate_rows, frame_rows, initial, states = [], [], [], None, {}
    for branch in cfg["branches"]:
        entry = next(r for r in status["runs"] if r["variant"] == branch["id"])
        row = {"variant": branch["id"], "policy": branch, "run": entry["name"], "status": entry["status"]}
        summaries.append(row)
        if row["status"] == "skipped_pilot_failure":
            continue
        run = ROOT / "runs/local" / row["run"]
        requested = read(run / "requested.json")
        result = read(run / "result.json")
        assert requested["source_digest"] == cfg["source_digest"] and requested["exe_sha256"] == cfg["fused_sha256"]
        assert requested["runner_sha256"] == sha(ROOT / "tools/run_local.py")
        assert requested["dt"] == .01 and requested["tol"] == .01 and requested["pcg_tol"] == .0001 and requested["suite"] == "0"
        row.update(result=result, requested=requested, stats_sha256=sha(run / "output/stats.json"))
        if result["status"] != "completed":
            row["failure_log_tail"] = (run / "run.log").read_text(errors="replace").splitlines()[-12:]
            continue
        assert result["recorded_frames"] == expected_steps and result["finite"] and result["timing_is_diagnostic"]
        detail = work(run)
        frames = detail.pop("frames")
        assert detail["directions"] == sum(t["inner_iterations"] for f in frames for t in f["toi"])
        first, all_candidates, warm_states = [], [], []
        for number, frame in enumerate(frames, 1):
            policy = frame["toi_warm_history"]
            expected_policy = {"contacts": branch["contacts"], "lambda": branch["lambda"], "gamma": branch["gamma"],
                               "friction_mode": "coupled_al" if branch["mode"] == "coupled" else "independent_proxy",
                               "friction": None if branch["mode"] == "coupled" else branch["friction"]}
            assert policy == expected_policy, (branch["id"], policy, expected_policy)
            warm = frame["toi_warm_start_state"]
            warm_states.append(warm)
            assert warm["contacts"] == frame["toi_previous_frame_active_count"]
            if not branch["contacts"]:
                assert warm["contacts"] == 0
            if not branch["lambda"]:
                assert warm["lambda_sum"] == warm["positive_lambda_contacts"] == 0
            if not branch["gamma"]:
                assert warm["gamma_below_one_contacts"] == 0
            if branch["mode"] == "independent" and not branch["friction"]:
                assert warm["friction_force_proxy_sum"] == 0
            if branch["id"] in ["cold", "all_independent"]:
                assert warm["independent_proxy_vs_al_max_abs_difference"] == 0
            assert frame["toi_blocker_audit_enabled"] == (number >= 22)
            local_first = []
            for outer in frame["toi"]:
                if branch["id"] in ["cold", "all_independent"]:
                    assert outer["independent_proxy_vs_al_max_abs_difference"] == 0
                assert ("blocker_audit" in outer) == (number >= 22)
                if number < 22:
                    continue
                audit = outer["blocker_audit"]
                assert audit["ccd_alpha"] == outer["full_ccd_alpha"]
                if not audit["limited"]:
                    continue
                assert audit["candidates"][0]["is_global_minimum"]
                for order, candidate in enumerate(audit["candidates"]):
                    assert candidate["plane_valid"]
                    record = {"variant": branch["id"], "frame": number, "outer": outer["outer"], "candidate_order": order,
                              "ccd_alpha": audit["ccd_alpha"], "accepted_alpha": outer["alpha"], **candidate}
                    candidate_rows.append(record)
                    all_candidates.append(record)
                    if order == 0:
                        first.append(record)
                        local_first.append(record)
            frame_rows.append({"variant": branch["id"], "frame": number, "outer": len(frame["toi"]),
                               "directions": sum("pcg" in n for n in frame["newton"]), "warm_start": warm,
                               "first_min_classes": dict(collections.Counter(r["classification"] for r in local_first))})
        state = states[branch["id"]] = States(run)
        if initial is None:
            initial = state
        assert np.array_equal(state.positions[0], initial.positions[0]) and np.array_equal(state.mass, initial.mass)
        assert np.array_equal(state.faces, initial.faces) and np.array_equal(state.tets, initial.tets)
        consistency = {key: distribution([c.get(key) for c in all_candidates]) for key in
                       ["host_probe_vs_solver_plane_max_abs_error", "multiplier_update_relative_error", "slack_update_abs_error_m", "gamma_update_abs_error"]}
        # Probe coefficients can be sensitive when unsigned distance is close
        # to floating-point resolution. Keep the original check as a reported
        # failure; classify active contacts using their actual GPU plane.
        probe_outliers = [c for c in all_candidates if c.get("host_probe_vs_solver_plane_max_abs_error", 0) >= 1e-8]
        assert all(v is None or v["max"] < 1e-8 for k, v in consistency.items() if k != "host_probe_vs_solver_plane_max_abs_error")
        row.update(work=detail, physics=state.physical(), vs_refined_base_dt4=state.compare(reference4),
                   subsets={"all_first_min": subset_summary(first), "first5_outer": subset_summary([r for r in first if r["outer"] < 5]),
                            "after5_outer": subset_summary([r for r in first if r["outer"] >= 5])},
                   consistency=consistency, probe_coefficient_1e_minus8_check_pass=not probe_outliers,
                   probe_coefficient_outliers=probe_outliers,
                   warm_start_summary={key: distribution([w[key] for w in warm_states[22:]])
                       for key in ["contacts", "lambda_sum", "positive_lambda_contacts", "gamma_below_one_contacts", "friction_force_proxy_sum"]})
        if args.stage == "diagnose":
            paths = sorted((run / "trace/substeps").glob("safe_*.bin"))
            counts = {i: 0 for i in range(expected_steps)}
            digest = hashlib.sha256()
            for path in paths:
                _, frame, index = path.stem.split("_")
                assert int(index) == counts[int(frame)]
                counts[int(frame)] += 1
                digest.update(path.name.encode())
                digest.update(bytes.fromhex(sha(path)))
            assert all(counts[i] == len(frames[i]["toi"]) + 1 for i in range(expected_steps))
            row["trace_coverage"] = {"all_frames_and_outer": True, "substep_states": len(paths), "substep_joint_sha256": digest.hexdigest()}
            safety = ROOT / "reports" / f"v31_{branch['id']}_accepted_ccd.json"
            if safety.exists():
                validation = read(safety)
                assert validation["paths_checked"] == len(paths) - 1
                row["independent_ccd"] = validation
                row["independent_ccd_sha256"] = sha(safety)
    for row in summaries:
        if row["variant"] in states and "cold" in states and row["variant"] != "cold":
            row["vs_cold_toi"] = states[row["variant"]].compare(states["cold"])
    payload = {"protocol": cfg, "stage": args.stage, "runs": summaries, "frames": frame_rows,
               "source_files_verified": len(manifest["files"]), "component_tests": read(ROOT / "builds/v31_toi_components.json"),
               "refined_reference_dt2_vs_dt4": prefix(reference2, expected_steps * 2).compare(reference4),
               "scope": "One diagnostic run per policy; optional friction force proxy is an implementation ablation. Current trial still depends on normal/friction history. No formal speed or causal qualification.",
               "quality_matched_speedup": None}
    target = ROOT / "reports" / f"TOI_WARM_HISTORY_V31_{args.stage.upper()}_20261001.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.stage == "diagnose":
        fields = ["variant", "frame", "outer", "candidate_order", "kind", "ids", "classification", "ccd_alpha", "accepted_alpha",
                  "in_trial_system", "in_active_set_after_update", "constraint_trial_m", "delta_m", "lambda_used", "gamma_used"]
        with target.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(candidate_rows)
    for row in summaries:
        print(json.dumps({"variant": row["variant"], "status": row["status"],
                          "work": row.get("work"), "subsets": row.get("subsets"), "warm_start": row.get("warm_start_summary")}))


if __name__ == "__main__":
    main()
