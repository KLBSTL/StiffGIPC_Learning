"""Serial isolated potential probe. Cross-engine ratios are diagnostic.

Reuses a hash-matched historical binary, never rebuilds or modifies donors.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "stiffGIPC"
SUITE = OLD / "benchmarks/stiff4-v1"
MATRIX = ROOT / "runs/local/robust_port_probe_20261001"
STEPS = 30


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    global STEPS, MATRIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=[30, 60], default=30)
    args = parser.parse_args()
    STEPS = args.steps
    if STEPS == 60:
        MATRIX = ROOT / "runs/local/robust_port_probe_20261001_60f"
    MATRIX.mkdir(exist_ok=True)
    bins = MATRIX / "frozen_bin"
    bins.mkdir(exist_ok=True)
    source_bins = OLD / "barrier-free/build-ninja2/Release/bin"
    original = source_bins / "stiff4_cloth_cases.exe"
    expected = read(SUITE / "runs/four-arm-contact-20260927/cloth_sphere7_l/robust2026/r01/environment.json")["exe_sha256"]
    assert sha(original) == expected
    copied = []
    for src in sorted(source_bins.iterdir()):
        if src.is_file() and (src.suffix.lower() == ".dll" or src.name == original.name):
            dst = bins / src.name
            if dst.exists():
                assert sha(dst) == sha(src)
            else:
                shutil.copy2(src, dst)
            copied.append({"name": src.name, "sha256": sha(dst), "bytes": dst.stat().st_size})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIPC_", "STIFF4_"))}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip()
    assert int(gpu.split(",")[2]) > 3500
    protocol = {"frames": STEPS, "repeats": 3, "dt_s": .01, "scene": "cloth_sphere7_l",
                "qualification": "standalone potential only; cross-engine physics, geometry, stopping and CCD not matched",
                "gpu_preflight": gpu, "serial_gpu": True, "frozen_runtime": copied,
                "runner_sha256": sha(Path(__file__)), "diagnostic_exports": "Stiff trace; Robust timing only",
                "ordering": "base, Robust .05, Graph, Robust 1; reverse next repeat",
                "source_mutations": "none"}
    write(MATRIX / "protocol.json", protocol)
    records = []
    sequence = ["base", "robust_v005", "base_graph", "robust_v1"]
    for repeat in range(1, 4):
        for arm in sequence if repeat % 2 else sequence[::-1]:
            name = f"robust_port_probe_{arm}_{STEPS}_r{repeat:02d}"
            run = ROOT / "runs/local" / name
            if arm.startswith("robust"):
                result_file = run / "probe_result.json"
                if not result_file.exists():
                    run.mkdir(exist_ok=False)
                    run_env = env.copy()
                    run_env.update(STIFF4_CASE_ID="cloth_sphere7_l",
                                   STIFF4_SCENE_MANIFEST=str(SUITE / "scene_manifests/cloth_sphere7_l.json"),
                                   STIFF4_OBJECT_MANIFEST=str(SUITE / "scene_object_manifests/cloth_sphere7_l.json"),
                                   STIFF4_ROBUST_INITIAL_DIR=str(run), STIFF4_ROBUST_TIMING_DIR=str(run),
                                   STIFF4_ROBUST_DIAGNOSTIC_VELOCITY_TOL="0.05" if arm == "robust_v005" else "1.0")
                    torch_dlls = Path(sys.executable).parent / "Lib/site-packages/torch/lib"
                    run_env["PATH"] = os.pathsep.join([str(bins), str(torch_dlls), env.get("PATH", "")])
                    command = [str(bins / original.name), "sphere7", str(STEPS)]
                    write(run / "probe_command.json", {"command": command, "cwd": str(run), "exe_sha256": expected})
                    started = time.monotonic()
                    with (run / "process.log").open("wb") as stream:
                        proc = subprocess.run(command, cwd=run, env=run_env, stdout=stream,
                                              stderr=subprocess.STDOUT, timeout=180)
                    log = (run / "process.log").read_text(encoding="utf-8", errors="replace")
                    observed = [int(x) for x in re.findall(r"^STIFF4_CLOTH_FRAME case=\S+ frame=(\d+)$", log, re.MULTILINE)]
                    data = {"exit_code": proc.returncode, "observed_frames": observed,
                            "process_wall_seconds": time.monotonic() - started,
                            "newton_cap_log_hits": log.count("Newton Iteration Exits with Max Iteration"),
                            "linear_cap_log_hits": log.count("Linear Solver Exits with Max Iteration")}
                    write(result_file, data)
                result = read(result_file)
                assert result["exit_code"] == 0 and result["observed_frames"] == list(range(1, STEPS + 1)), result
                with (run / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == STEPS and all(r["status"] == "completed_diagnostic" for r in rows)
                log = (run / "process.log").read_text(encoding="utf-8", errors="replace")
                cg = [int(x) for x in re.findall(r"Iterative linear solver iteration count: (\d+)", log)]
                directions = sum(int(r["newton_iterations"]) for r in rows)
                assert len(cg) == directions
                config = read(run / "robust_requested_config.json")
                assert config["scene_manifest_consumed"] and config["object_manifest_consumed"]
                assert config["config"]["newton"]["velocity_tol"] == (.05 if arm == "robust_v005" else 1.)
                seconds = sum(float(r["step_wall_ms"]) for r in rows) / 1000
                inner_seconds = sum(float(r["newton_wall_ms"]) for r in rows) / 1000
                cg_count = sum(cg)
            else:
                if not (run / "result.json").exists():
                    assert not run.exists()
                    command = [sys.executable, str(ROOT / "tools/run_local.py"), "--arm", arm,
                               "--build-tag", "v31", "--scene", "cloth_sphere7_l", "--steps", str(STEPS),
                               "--name", name, "--trace", "--suite", "0", "--tol", "0.01", "--pcg-tol", "0.0001", "--dt", "0.01"]
                    with (MATRIX / f"{name}.launcher.log").open("wb") as stream:
                        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
                    assert proc.returncode == 0, name
                result = read(run / "result.json")
                assert result["status"] == "completed" and result["finite"] and result["recorded_frames"] == STEPS
                frames = read(run / "output/stats.json")["frames"]
                solves = [n for f in frames for n in f["newton"] if "pcg" in n]
                directions = len(solves)
                cg_count = sum(n["pcg"]["iterations"] for n in solves)
                seconds = result["solver_seconds"]
                inner_seconds = None
            row = {"arm": arm, "repeat": repeat, "run": str(run.relative_to(ROOT)), "frames": STEPS,
                   "measured_window_seconds": seconds, "directions": directions, "cg_iterations": cg_count,
                   "inner_seconds": inner_seconds, "result": result}
            records.append(row)
            write(MATRIX / "progress.json", {"status": "running", "runs": records})
            print(json.dumps({k: v for k, v in row.items() if k != "result"}), flush=True)
    summaries = []
    for arm in sequence:
        rows = [r for r in records if r["arm"] == arm]
        paired = [next(r for r in records if r["arm"] == "base" and r["repeat"] == row["repeat"])["measured_window_seconds"] / row["measured_window_seconds"] for row in rows]
        summaries.append({"arm": arm, "seconds_median": statistics.median(r["measured_window_seconds"] for r in rows),
                          "paired_raw_ratio_median": statistics.median(paired), "paired_raw_ratio_range": [min(paired), max(paired)],
                          "directions_median": statistics.median(r["directions"] for r in rows),
                          "cg_iterations_median": statistics.median(r["cg_iterations"] for r in rows)})
    output = {"protocol": protocol, "runs": records, "summary": summaries,
              "timing_boundary": "Stiff solver window versus Robust world.advance step; not identical instrumentation",
              "quality_verified": False, "speed_qualified": False, "all_12_completed": True}
    suffix = "_60F" if STEPS == 60 else ""
    write(ROOT / f"reports/ROBUST_STANDALONE_POTENTIAL_20261001{suffix}.json", output)
    write(MATRIX / "progress.json", {"status": "completed", "runs": records})
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
