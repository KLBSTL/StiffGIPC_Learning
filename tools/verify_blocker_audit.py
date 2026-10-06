"""Verify identities, effective options and full accepted-path export coverage."""
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
    cfg = read(ROOT / "configs/local_blocker_v30_20261001.json")
    manifest = read(ROOT / "IMPLEMENTATION_MANIFEST.json")
    assert manifest["source_digest"] == cfg["source_digest"]
    assert all(sha(ROOT / f["path"]) == f["sha256"] for f in manifest["files"])
    assert sha(ROOT / "builds/local-fused-v30/Release/gipc.exe") == cfg["fused_sha256"]
    assert sha(ROOT / "builds/local-fused-v29/Release/gipc.exe") == "5b13fc3bffde9208d148bb360a4137ab295c140f167d3dbfce2794277ec1e907"
    assert read(ROOT / "research/v29_before_blocker_audit/IMPLEMENTATION_MANIFEST.json")["source_digest"] == "b72075846c78895020d7f724718ebe7bffc6d5514b6eceb14853400001e1ed62"
    component = read(ROOT / "builds/v30_toi_components.json")
    assert component["passed"] and len(component["tests"]) == 6 and len(component["volume_step_tests"]) == 17
    rows, baseline = [], None
    matrix = ROOT / "runs/local" / cfg["name"]
    for stage in ["smoke", "pilot", "diagnose"]:
        status = read(matrix / f"{stage}_status.json")
        assert status["status"] == "completed" and status["single_gpu_serial"]
        for entry in status["runs"]:
            run = ROOT / "runs/local" / entry["name"]
            result, request = read(run / "result.json"), read(run / "requested.json")
            frames = read(run / "output/stats.json")["frames"]
            assert result["status"] == "completed" and result["finite"] and result["recorded_frames"] == entry["steps"]
            assert request["exe_sha256"] == cfg["fused_sha256"] and request["source_digest"] == cfg["source_digest"]
            assert request["runner_sha256"] == sha(ROOT / "tools/run_local.py")
            assert len(frames) == entry["steps"]
            assert all(f["toi_exit"] != "iteration_limit" for f in frames)
            assert not any(n["pcg"].get("iteration_limit", False) for f in frames for n in f["newton"] if "pcg" in n)
            states = sorted((run / "trace").glob("state_*.bin"))
            assert len(states) == entry["steps"] + 1
            signature = [sha(run / "trace" / f) for f in ["state_0000.bin", "masses.bin", "topology.bin", "body_ids.bin", "boundary_types.bin"]]
            baseline = signature if baseline is None else baseline
            assert signature == baseline
            row = {"name": entry["name"], "status": result["status"], "frames": len(frames),
                   "finite_native_states": all(np.isfinite(np.fromfile(f, dtype="<f8")).all() for f in states),
                   "native_trace_count": len(states), "initial_geometry_topology_mass_boundary_identical": True}
            assert row["finite_native_states"]
            if stage == "diagnose":
                variant = entry["variant"]
                paths = sorted((run / "trace/substeps").glob("safe_*.bin"))
                counts = {i: 0 for i in range(60)}
                for path in paths:
                    _, frame, inner = path.stem.split("_")
                    assert int(inner) == counts[int(frame)]
                    counts[int(frame)] += 1
                assert all(counts[i] == len(frames[i]["toi"]) + 1 for i in range(60))
                ccd = read(ROOT / "reports" / f"v30_{variant}_accepted_ccd.json")
                assert ccd["passed"] and ccd["finite"] and ccd["conservative_collision_flags"] == ccd["tet_inversions"] == 0
                assert ccd["paths_checked"] == len(paths) - 1
                digest = hashlib.sha256()
                for path in paths:
                    digest.update(path.name.encode())
                    digest.update(bytes.fromhex(sha(path)))
                row.update(substep_states=len(paths), substep_paths_checked=ccd["paths_checked"],
                           all_60_frames_and_every_outer_exported=True, substep_files_joint_sha256=digest.hexdigest(), independent_ccd=ccd)
            rows.append(row)
    output = {"passed": True, "source_digest": cfg["source_digest"], "source_files": len(manifest["files"]),
              "binary_sha256": cfg["fused_sha256"], "archived_v29_verified": True, "component_tests": [6, 17],
              "runs": rows, "analysis_script_sha256": sha(ROOT / "tools/report_blocker_audit.py"),
              "scope": "Execution identity and export/safety audit; not a physical accuracy or speed qualification."}
    (ROOT / "reports/v30_blocker_verification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "source_files": len(manifest["files"]), "completed_runs": len(rows),
                      "accepted_paths": [r["substep_paths_checked"] for r in rows if "substep_paths_checked" in r]}))


if __name__ == "__main__":
    main()
