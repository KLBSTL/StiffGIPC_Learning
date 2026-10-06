"""Final source, archive, policy, completion and path coverage verification."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = read(ROOT / "configs/local_warm_v31_20261001.json")
    manifest = read(ROOT / "IMPLEMENTATION_MANIFEST.json")
    assert manifest["source_digest"] == cfg["source_digest"]
    assert all(sha(ROOT / f["path"]) == f["sha256"] for f in manifest["files"])
    assert sha(ROOT / "builds/local-fused-v31/Release/gipc.exe") == cfg["fused_sha256"]
    archive = ROOT / "research/v30_before_state_split"
    old = read(archive / "IMPLEMENTATION_MANIFEST.json")
    for f in old["files"]:
        current = ROOT / f["path"]
        saved = archive / current.name if current.name in ["toi_solver.cu", "toi_solver.cuh"] and "stiff_fused/" in f["path"] else current
        assert sha(saved) == f["sha256"], f["path"]
    assert sha(ROOT / "builds/local-fused-v30/Release/gipc.exe") == "b1c359d1da5a85dbde0d1eeb38f1522f0896091dee7f50e2e177a38003783786"
    assert sha(archive / "run_local.py") == read(ROOT / "runs/local/local_v30_blocker_full_audit60/requested.json")["runner_sha256"]
    component = read(ROOT / "builds/v31_toi_components.json")
    assert component["passed"] and [len(component[k]) for k in ["tests", "warm_history_tests", "volume_step_tests"]] == [6, 7, 17]
    diagnosis = read(ROOT / "reports/TOI_WARM_HISTORY_V31_DIAGNOSE_20261001.json")
    matrix = ROOT / "runs/local" / cfg["name"]
    validation = read(matrix / "validation_status.json")
    assert validation["status"] == "completed"
    rows, common_initial = [], None
    for stage in ["smoke", "pilot", "diagnose"]:
        status = read(matrix / f"{stage}_status.json")
        assert status["status"] in ["completed", "completed_with_failures"] and status["single_gpu_serial"]
        for entry in status["runs"]:
            row = {"stage": stage, "run": entry["name"], "variant": entry["variant"], "status": entry["status"]}
            rows.append(row)
            if entry["status"] != "completed":
                continue
            run = ROOT / "runs/local" / entry["name"]
            requested, result = read(run / "requested.json"), read(run / "result.json")
            assert requested["source_digest"] == cfg["source_digest"] and requested["exe_sha256"] == cfg["fused_sha256"]
            assert requested["runner_sha256"] == sha(ROOT / "tools/run_local.py")
            assert result["recorded_frames"] == entry["steps"] and result["finite"]
            frames = read(run / "output/stats.json")["frames"]
            assert len(frames) == entry["steps"]
            assert all(f["toi_exit"] != "iteration_limit" for f in frames)
            assert not any(n["pcg"].get("iteration_limit", False) for f in frames for n in f["newton"] if "pcg" in n)
            initial = [sha(run / "trace" / name) for name in ["state_0000.bin", "topology.bin", "masses.bin", "body_ids.bin", "boundary_types.bin"]]
            common_initial = initial if common_initial is None else common_initial
            assert initial == common_initial
            paths = sorted((run / "trace").glob("state_*.bin"))
            assert len(paths) == entry["steps"] + 1
            assert all(np.isfinite(np.fromfile(path, dtype="<f8")).all() for path in paths)
            row.update(frames=len(frames), initial_identical=True, all_native_states_finite=True,
                       stats_sha256=sha(run / "output/stats.json"))
            if stage == "diagnose":
                detail = next(r for r in diagnosis["runs"] if r["variant"] == entry["variant"])
                assert detail["trace_coverage"]["all_frames_and_outer"]
                assert detail["work"]["directions"] == sum(t["inner_iterations"] for f in frames for t in f["toi"])
                safety = read(ROOT / "reports" / f"v31_{entry['variant']}_accepted_ccd.json")
                assert safety["paths_checked"] == detail["trace_coverage"]["substep_states"] - 1
                row.update(effective_warm_policy_validated=True, full_substep_coverage=True, independent_ccd=safety)
    smoke_a = ROOT / "runs/local" / f"{cfg['run_prefix']}_cold_coupled_smoke2/trace/state_0002.bin"
    smoke_b = ROOT / "runs/local" / f"{cfg['run_prefix']}_cold_smoke2/trace/state_0002.bin"
    a, b = [np.fromfile(p, dtype="<f8").reshape(-1, 3) for p in [smoke_a, smoke_b]]
    smoke_error = float(np.linalg.norm(a - b, axis=1).max())
    assert smoke_error < 1e-8
    safety_rows = [r for r in rows if "independent_ccd" in r]
    output = {"execution_identity_coverage_verified": True, "source_digest": cfg["source_digest"],
              "binary_sha256": cfg["fused_sha256"], "source_files": len(manifest["files"]),
              "v30_reconstructed_source_verified": True, "component_tests": [6, 7, 17],
              "completed_runs": sum(r["status"] == "completed" for r in rows),
              "failed_or_skipped_runs": [r for r in rows if r["status"] != "completed"],
              "independent_ccd_all_passed": all(r["independent_ccd"]["passed"] for r in safety_rows),
              "accepted_paths_checked": sum(r["independent_ccd"]["paths_checked"] for r in safety_rows),
              "probe_coefficient_check_failures": [{"variant": r["variant"], "outliers": len(r["probe_coefficient_outliers"])}
                  for r in diagnosis["runs"] if r.get("probe_coefficient_1e_minus8_check_pass") is False],
              "coupled_independent_smoke_max_vertex_error_m": smoke_error, "runs": rows,
              "scope": "Numerical update and runtime/safety audit; physical quality and performance remain separate qualifications."}
    (ROOT / "reports/v31_warm_verification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ["execution_identity_coverage_verified", "completed_runs", "source_files", "component_tests",
                                            "independent_ccd_all_passed", "accepted_paths_checked", "probe_coefficient_check_failures"]}))


if __name__ == "__main__":
    main()
