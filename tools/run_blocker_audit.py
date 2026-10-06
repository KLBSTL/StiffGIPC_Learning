"""Serial staged v30 blocker diagnostics; resume completed runs without overwriting."""
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_blocker_v30_20261001.json")
    parser.add_argument("--stage", choices=["smoke", "pilot", "diagnose"], required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    manifest = read(ROOT / "IMPLEMENTATION_MANIFEST.json")
    assert manifest["source_digest"] == cfg["source_digest"]
    assert sha(ROOT / "builds/local-fused-v30/Release/gipc.exe") == cfg["fused_sha256"]
    changed = [f["path"] for f in manifest["files"] if sha(ROOT / f["path"]) != f["sha256"]]
    assert not changed, changed
    assert read(ROOT / "builds/v30_toi_components.json")["passed"]
    matrix = ROOT / "runs/local" / cfg["name"]
    matrix.mkdir(exist_ok=True)
    write(matrix / "protocol.json", cfg)
    if not (matrix / "implementation_manifest.json").exists():
        (matrix / "implementation_manifest.json").write_bytes((ROOT / "IMPLEMENTATION_MANIFEST.json").read_bytes())
    previous = "smoke" if args.stage == "pilot" else "pilot" if args.stage == "diagnose" else None
    if previous:
        assert read(matrix / f"{previous}_status.json")["status"] == "completed"
    state = {"stage": args.stage, "status": "running", "runs": [], "source_files_checked": len(manifest["files"]),
             "source_digest": cfg["source_digest"], "fused_sha256": cfg["fused_sha256"],
             "started_unix": time.time(), "timing_scope": "diagnostic; no speed claim",
             "runner_sha256": sha(Path(__file__)), "single_gpu_serial": True}
    status_file = matrix / f"{args.stage}_status.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIPC_")}
    full = cfg["branches"][0]
    cases = [(full, 2, False), (full, 2, True)] if args.stage == "smoke" else [(full, 30, False), (full, 30, True)] if args.stage == "pilot" else [(b, cfg["diagnostic_frames"], True) for b in cfg["branches"]]
    for branch, steps, audit in cases:
        suffix = f"smoke2_{'on' if audit else 'off'}" if args.stage == "smoke" else f"pilot30_{'on' if audit else 'off'}" if args.stage == "pilot" else "audit60"
        name = f"local_v30_blocker_{branch['id']}_{suffix}"
        run = ROOT / "runs/local" / name
        command = [sys.executable, str(ROOT / "tools/run_local.py"), "--arm", "base_toi", "--build-tag", "v30",
                   "--scene", cfg["scene"], "--steps", str(steps), "--name", name, "--timeout", str(cfg["timeout_seconds"]),
                   "--trace", "--trace-stride", "1", "--suite", cfg["suite"], "--tol", str(cfg["newton_tol"]),
                   "--pcg-tol", str(cfg["pcg_tol"]), "--dt", str(cfg["dt"]), "--mu-mode", cfg["mu_mode"]]
        if audit:
            start = 1 if args.stage == "smoke" else cfg["audit_from_frame"]
            command += ["--blocker-audit-from-frame", str(start), "--blocker-audit-to-frame", str(steps)]
        if args.stage == "diagnose":
            command += ["--substeps"]
        if branch["velocity_tol"] is not None:
            command += ["--robust-velocity-tol", str(branch["velocity_tol"]), "--robust-velocity-stop"]
        if branch["persist_contacts"]:
            command += ["--persist-contacts"]
        entry = {"name": name, "variant": branch["id"], "steps": steps, "audit": audit, "command": command, "status": "running"}
        state["runs"].append(entry)
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
        write(status_file, state)
        print(json.dumps({"name": name, "result": result}), flush=True)
        if result["status"] != "completed" or result.get("recorded_frames") != steps or not result.get("finite"):
            state["status"] = "needs_diagnosis"
            write(status_file, state)
            raise SystemExit(1)
    state["status"] = "completed"
    state["finished_unix"] = time.time()
    write(status_file, state)


if __name__ == "__main__":
    main()
