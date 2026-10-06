"""Audit saved experimental coverage and identity; never launch a simulation."""
from pathlib import Path
from report_velocity_ablation import read, sha, work, States
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = read(ROOT / "configs/local_velocity_20261001.json")
    matrix = ROOT / "runs/local" / cfg["name"]
    frozen = read(matrix / "implementation_manifest.json")
    assert frozen["source_digest"] == cfg["source_digest"]
    mismatches = [f["path"] for f in frozen["files"] if sha(ROOT / f["path"]) != f["sha256"]]
    assert not mismatches, mismatches
    for label, directory in [("base", "local-base"), ("fused", "local-fused")]:
        assert sha(ROOT / "builds" / directory / "Release/gipc.exe") == cfg["binary_sha256"][label]
    cases = []
    initial = None
    topology_hash = None
    for stage, count in [("pilot", 4), ("measure", 12), ("reference_half", 1), ("reference", 1), ("safety", 4)]:
        status = read(matrix / f"{stage}_status.json")
        assert status["status"] == "completed" and len(status["runs"]) == count
        for item in status["runs"]:
            assert item["status"] == "completed"
            run = ROOT / "runs/local" / item["name"]
            result = read(run / "result.json")
            request = read(run / "requested.json")
            assert result["status"] == "completed" and result["finite"]
            assert result["recorded_frames"] == item["steps"] and request["dt"] == item["dt"]
            assert request["source_digest"] == cfg["source_digest"]
            assert request["exe_sha256"] == cfg["binary_sha256"]["base" if item["variant"] == "base" else "fused"]
            assert request["suite"] == "0" and request["persist_contacts"] is False
            assert request["substeps"] == (stage == "safety")
            variant = next(v for v in cfg["variants"] if v["id"] == item["variant"])
            assert request["robust_velocity_tol"] == variant["velocity_tol"]
            assert request["robust_velocity_stop"] == variant["velocity_stop"]
            strict = stage in ["reference_half", "reference"]
            assert request["tol"] == cfg["reference_newton_tol" if strict else "newton_tol"]
            assert request["pcg_tol"] == cfg["reference_pcg_tol" if strict else "pcg_tol"]
            details = work(run)
            frames = details.pop("frames")
            assert len(frames) == item["steps"]
            assert details["pcg_limit_hits"] == details["outer_limit_hits"] == 0
            if item["variant"] != "base":
                assert details["effective_inner_policies"] == ["robust_velocity" if variant["velocity_stop"] else "full_step"]
                assert details["effective_velocity_tolerances"] == [variant["velocity_tol"] or 0]
            states = States(run)
            assert np.isfinite(states.positions).all()
            if initial is None:
                initial = states.positions[0]
                topology_hash = sha(run / "trace/topology.bin")
            assert np.array_equal(initial, states.positions[0])
            assert topology_hash == sha(run / "trace/topology.bin")
            physical = states.physical()
            assert physical["fixed_abd_max_displacement_all_frames_m"] < 1e-12
            assert physical["tet_nonpositive_jacobian_frame_events"] == 0
            entry = {"run": item["name"], "stage": stage, "variant": item["variant"],
                     "frames": result["recorded_frames"], "finite": True, "pcg_limit_hits": 0,
                     "outer_limit_hits": 0, "solver_seconds": result["solver_seconds"],
                     "fixed_abd_max_displacement_m": physical["fixed_abd_max_displacement_all_frames_m"],
                     "all_frame_tet_inversions": 0}
            if stage == "safety":
                output = ROOT / "reports" / f"velocity_20261001_{item['variant']}_accepted_ccd.json"
                validation = read(output)
                paths = sorted((run / "trace/substeps").glob("safe_*.bin"))
                assert {int(p.stem.split("_")[1]) for p in paths} == set(range(cfg["measure_frames"]))
                assert validation["paths_checked"] == len(paths) - 1
                assert validation["scope"] == "accepted substeps"
                assert validation["passed"] and validation["conservative_collision_flags"] == 0
                assert validation["tet_inversions"] == 0 and validation["finite"]
                entry.update(accepted_substep_files=len(paths), ccd_paths=validation["paths_checked"],
                             ccd_json_sha256=sha(output), ccd_passed=True)
            cases.append(entry)
    payload = {"passed": True, "verified_unix": time.time(), "runs_checked": len(cases),
               "source_files_checked": len(frozen["files"]), "source_digest": cfg["source_digest"],
               "binary_sha256": cfg["binary_sha256"], "identical_initial_state": True,
               "identical_topology_sha256": topology_hash, "runs": cases,
               "validator_binary_sha256": sha(ROOT / "builds/validator/Release/validate_path.exe"),
               "validator_source_sha256": sha(ROOT / "tools/validator/validate_path.cpp"),
               "reporter_sha256": sha(ROOT / "tools/report_velocity_ablation.py"),
               "scope": "Saved-result identity/completion/geometry and separate safety-path audit; not physical fidelity qualification"}
    path = ROOT / "reports/velocity_20261001_verification.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "runs_checked": len(cases), "source_files_checked": len(frozen["files"]), "output": str(path)}))


if __name__ == "__main__":
    main()
