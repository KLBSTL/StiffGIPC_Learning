"""Serial, staged v31 cross-frame state ablation with preserved results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "diagnose"], required=True)
    args = parser.parse_args()
    cfg = read(ROOT / "configs/local_warm_v31_20261001.json")
    manifest = read(ROOT / "IMPLEMENTATION_MANIFEST.json")
    assert manifest["source_digest"] == cfg["source_digest"]
    assert sha(ROOT / "builds/local-fused-v31/Release/gipc.exe") == cfg["fused_sha256"]
    assert all(sha(ROOT / f["path"]) == f["sha256"] for f in manifest["files"])
    component = read(ROOT / "builds/v31_toi_components.json")
    assert component["passed"] and len(component["warm_history_tests"]) == 7
    matrix = ROOT / "runs/local" / cfg["name"]
    matrix.mkdir(exist_ok=True)
    write(matrix / "protocol.json", cfg)
    if not (matrix / "implementation_manifest.json").exists():
        (matrix / "implementation_manifest.json").write_bytes((ROOT / "IMPLEMENTATION_MANIFEST.json").read_bytes())
    previous = "smoke" if args.stage == "pilot" else "pilot" if args.stage == "diagnose" else None
    previous_runs = {}
    if previous:
        last = read(matrix / f"{previous}_status.json")
        assert last["status"] in ["completed", "completed_with_failures"]
        if previous == "smoke":
            assert last["status"] == "completed"
        previous_runs = {r["variant"]: r for r in last["runs"]}
    state = {"stage": args.stage, "status": "running", "runs": [], "source_files_checked": len(manifest["files"]),
             "source_digest": cfg["source_digest"], "fused_sha256": cfg["fused_sha256"],
             "started_unix": time.time(), "timing_scope": "diagnostic; no speed claim",
             "runner_sha256": sha(Path(__file__)), "single_gpu_serial": True}
    status_file = matrix / f"{args.stage}_status.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIPC_")}
    cold = cfg["branches"][0]
    branches = [{**cold, "id": "cold_coupled", "mode": "coupled"}, cold] if args.stage == "smoke" else cfg["branches"]
    steps = {"smoke": 2, "pilot": 30, "diagnose": 60}[args.stage]
    failures = 0
    for branch in branches:
        name = f"{cfg['run_prefix']}_{branch['id']}_{args.stage}{steps}"
        run = ROOT / "runs/local" / name
        entry = {"name": name, "variant": branch["id"], "steps": steps, "status": "running"}
        state["runs"].append(entry)
        if args.stage == "diagnose" and previous_runs[branch["id"]]["status"] != "completed":
            entry["status"] = "skipped_pilot_failure"
            failures += 1
            write(status_file, state)
            continue
        command = [sys.executable, str(ROOT / "tools/run_local.py"), "--arm", "base_toi", "--build-tag", "v31",
                   "--scene", cfg["scene"], "--steps", str(steps), "--name", name, "--timeout", str(cfg["timeout_seconds"]),
                   "--trace", "--trace-stride", "1", "--suite", cfg["suite"], "--tol", str(cfg["newton_tol"]),
                   "--pcg-tol", str(cfg["pcg_tol"]), "--dt", str(cfg["dt"]), "--mu-mode", cfg["mu_mode"]]
        if branch["id"] == "legacy":
            command += ["--persist-contacts"]
        else:
            for flag, field in [("--warm-contacts", "contacts"), ("--warm-lambda", "lambda"), ("--warm-gamma", "gamma")]:
                command += [flag, "keep" if branch[field] else "reset"]
            command += ["--friction-history", branch["mode"]]
            if branch["mode"] == "independent":
                command += ["--warm-friction", "keep" if branch["friction"] else "reset"]
        if args.stage != "smoke":
            command += ["--blocker-audit-from-frame", str(cfg["audit_from_frame"]), "--blocker-audit-to-frame", str(steps)]
        if args.stage == "diagnose":
            command += ["--substeps"]
        entry["command"] = command
        write(status_file, state)
        if (run / "result.json").exists():
            result = read(run / "result.json")
            assert read(run / "requested.json")["exe_sha256"] == cfg["fused_sha256"]
            entry["reused_saved_result"] = True
        else:
            assert not run.exists(), "Inspect partial run before resuming"
            with (matrix / f"{name}.launcher.log").open("wb") as log:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry["launcher_exit_code"] = process.returncode
            result = read(run / "result.json") if (run / "result.json").exists() else {"status": "launch_failed"}
        entry["result"] = result
        entry["status"] = result["status"]
        if result["status"] != "completed" or result.get("recorded_frames") != steps or not result.get("finite"):
            entry["status"] = "failed_validation"
            failures += 1
        write(status_file, state)
        print(json.dumps({"name": name, "result": result}), flush=True)
    state["status"] = "completed_with_failures" if failures else "completed"
    state["finished_unix"] = time.time()
    write(status_file, state)
    if args.stage == "smoke" and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
