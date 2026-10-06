"""Serial, resumable local velocity-stop ablation using frozen v29 binaries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_sample():
    fields = "name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return {"unix": time.time(), "fields": fields, "line": result.stdout.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_velocity_20261001.json")
    parser.add_argument("--stage", choices=["pilot", "measure", "reference_half", "reference", "safety"], required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    matrix = ROOT / "runs/local" / cfg["name"]
    matrix.mkdir(exist_ok=True)
    manifest = read(ROOT / "IMPLEMENTATION_MANIFEST.json")
    if manifest["source_digest"] != cfg["source_digest"]:
        raise RuntimeError("Source version changed; create a new protocol rather than mix versions")
    mismatches = [item["path"] for item in manifest["files"] if sha(ROOT / item["path"]) != item["sha256"]]
    if mismatches:
        raise RuntimeError("Source files differ from frozen manifest: " + repr(mismatches[:10]))
    for label, folder in [("base", "local-base"), ("fused", "local-fused")]:
        binary = ROOT / "builds" / folder / "Release/gipc.exe"
        if sha(binary) != cfg["binary_sha256"][label]:
            raise RuntimeError("Binary changed: " + label)
    write(matrix / "protocol.json", cfg)
    snapshot = matrix / "implementation_manifest.json"
    if not snapshot.exists():
        snapshot.write_bytes((ROOT / "IMPLEMENTATION_MANIFEST.json").read_bytes())
    write(matrix / "source_verification.json", {"files": len(manifest["files"]), "mismatches": mismatches,
        "source_digest": manifest["source_digest"], "binary_sha256": cfg["binary_sha256"], "verified_unix": time.time()})
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIPC_")}
    state = {"stage": args.stage, "status": "running", "runs": [], "started_unix": time.time(),
             "runner_sha256": sha(Path(__file__)), "gpu_initial": gpu_sample(),
             "background_graphics_are_not_terminated": True}
    status_file = matrix / (args.stage + "_status.json")

    def save():
        write(status_file, state)

    def name(stage, variant, rep):
        return f"{cfg['name']}_{variant}_{stage}_r{rep:02d}"

    def eligible(variant, stage):
        prerequisite = "pilot" if stage == "measure" else "measure"
        path = ROOT / "runs/local" / name(prerequisite, variant, 1) / "result.json"
        if not path.exists():
            return False
        prior = read(path)
        frames = cfg["pilot_frames"] if prerequisite == "pilot" else cfg["measure_frames"]
        return prior["status"] == "completed" and prior.get("finite", False) and prior.get("recorded_frames") == frames

    def launch(variant, rep):
        stage = args.stage
        label = variant["id"]
        run_name = name(stage, label, rep)
        run = ROOT / "runs/local" / run_name
        reference = stage in ["reference_half", "reference"]
        steps = cfg["pilot_frames"] if stage == "pilot" else cfg["reference_half_frames"] if stage == "reference_half" else cfg["reference_frames"] if stage == "reference" else cfg["measure_frames"]
        dt = cfg["reference_half_dt"] if stage == "reference_half" else cfg["reference_dt"] if stage == "reference" else cfg["dt"]
        timeout = cfg["reference_timeout_seconds"] if reference else cfg["timeout_seconds"]
        newton_tol = cfg["reference_newton_tol"] if reference else cfg["newton_tol"]
        pcg_tol = cfg["reference_pcg_tol"] if reference else cfg["pcg_tol"]
        command = [sys.executable, str(ROOT / "tools/run_local.py"), "--arm", variant["arm"],
            "--scene", cfg["scene"], "--steps", str(steps), "--name", run_name,
            "--timeout", str(timeout), "--trace", "--trace-stride", "1", "--suite", cfg["suite"],
            "--tol", str(newton_tol), "--pcg-tol", str(pcg_tol), "--dt", str(dt),
            "--mu-mode", cfg["mu_mode"]]
        if variant["velocity_tol"] is not None:
            command += ["--robust-velocity-tol", str(variant["velocity_tol"]), "--robust-velocity-stop"]
        if stage == "safety":
            command += ["--substeps"]
        item = {"variant": label, "stage": stage, "repeat": rep, "name": run_name,
                "steps": steps, "dt": dt, "command": command, "status": "running", "gpu_before": gpu_sample()}
        state["runs"].append(item)
        save()
        if (run / "result.json").exists():
            item["result"] = read(run / "result.json")
            item["reused_saved_result"] = True
        elif run.exists():
            item["result"] = {"status": "incomplete_previous_launch"}
        else:
            samples = []
            with (matrix / (run_name + ".launcher.log")).open("wb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                item["launcher_pid"] = process.pid
                save()
                while process.poll() is None:
                    samples.append(gpu_sample())
                    time.sleep(5)
                item["launcher_exit_code"] = process.returncode
            write(matrix / (run_name + ".gpu.json"), samples)
            item["result"] = read(run / "result.json") if (run / "result.json").exists() else {"status": "launch_failed"}
        item["status"] = item["result"]["status"]
        item["gpu_after"] = gpu_sample()
        save()
        print(json.dumps({"stage": stage, "variant": label, "repeat": rep, "result": item["result"]}), flush=True)
        return item

    save()
    variants = [cfg["variants"][0]] if args.stage in ["reference_half", "reference"] else cfg["variants"]
    repeats = cfg["repeats"] if args.stage == "measure" else 1
    failed = set()
    for rep in range(1, repeats + 1):
        order = variants[(rep - 1) % len(variants):] + variants[:(rep - 1) % len(variants)]
        for variant in order:
            label = variant["id"]
            if label in failed or (args.stage in ["measure", "safety"] and not eligible(label, args.stage)):
                state["runs"].append({"variant": label, "repeat": rep, "status": "skipped_failed_prerequisite"})
                save()
                continue
            item = launch(variant, rep)
            if item["status"] != "completed" or not item["result"].get("finite", False):
                failed.add(label)
    state["status"] = "finished_with_failures" if failed else "completed"
    state["finished_unix"] = time.time()
    save()
    print(json.dumps({"stage": args.stage, "status": state["status"], "failed_variants": sorted(failed)}), flush=True)


if __name__ == "__main__":
    main()
