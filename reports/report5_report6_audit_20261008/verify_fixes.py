"""Bounded Windows correctness checks, using the existing GPU lock and Job.

No simulations, cleanup, automatic retries, or performance certification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for folder in ("bench", "diagnostic", "full_eval", "local"):
    sys.path.insert(0, str(ROOT / "tools" / folder))
from windows_runner import gpu_lock, gpu_processes, driver_model
from linux_runner import gpu_query
from windows_owned_job import OwnedJob


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("narrow", "full"), required=True)
    args = ap.parse_args()
    if os.name != "nt":
        raise RuntimeError("This check uses the project's Windows owned-process contract")
    build, out = (ROOT / args.build).resolve(), (ROOT / args.out).resolve()
    if ROOT not in build.parents or ROOT not in out.parents:
        raise RuntimeError("Paths must be inside the repository")
    if out.exists():
        raise RuntimeError("One attempt per output directory; existing evidence is preserved")
    out.mkdir(parents=True)
    ctest = Path("D:/computer/cmake/bin/ctest.exe")
    sanitizer = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/compute-sanitizer/compute-sanitizer.exe")
    commands = [("ctest", [str(ctest), "--test-dir", str(build), "-C", "Release", "--output-on-failure"])]
    if args.stage == "narrow":
        commands[0][1].extend(["-R", "^(sym_spmv_equivalence|line_search_acceptance)$"])
        for tool in ("memcheck", "racecheck", "synccheck"):
            commands.append((tool, [str(sanitizer), "--tool", tool, "--error-exitcode", "42",
                                    str(build / "Release/sym_spmv_tests.exe")]))
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIPC_")}
    result = {"stage": args.stage, "scope": "correctness only; no physical quality or performance certification",
              "commands": [], "source_sha256": {name: sha(ROOT / name) for name in (
                  "StiffGIPC/core/GIPC.cu", "StiffGIPC/linear_system/utils/spmv.cu",
                  "StiffGIPC/solver/line_search_acceptance.h", "StiffGIPC/cuda_tools/scoped_cuda_events.h",
                  "tests/sym_spmv_tests.cu", "tests/line_search_acceptance_test.cpp")},
              "binaries": {p.name: sha(p) for p in (build / "Release").glob("*.exe")}}
    report = out / "RESULT.json"
    def save():
        report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    mode = driver_model(0)
    with gpu_lock(ROOT):
        initial = gpu_query(0)
        foreign = gpu_processes(0, None, mode)
        result.update(initial_gpu=initial, initial_processes=foreign)
        save()
        if initial["memory.free"] < 2048 or foreign["blocking_foreign_pids"] or shutil.disk_usage(out).free < 4 * 1024**3:
            raise RuntimeError("GPU/disk reserve or foreign-load guard failed")
        for name, argv in commands:
            started = time.monotonic()
            row = {"name": name, "argv": argv, "timeout_seconds": 120, "status": "running"}
            result["commands"].append(row)
            save()
            with OwnedJob() as job, (out / (name + ".log")).open("xb") as log:
                job.launch(argv, build, env, log)
                while not job.finished():
                    job.observe()
                    if time.monotonic() - started >= 120:
                        row["status"] = "timeout"
                        job.terminate()
                        break
                    sample = gpu_query(0, timeout=5)
                    if sample["memory.free"] < 1536 or shutil.disk_usage(out).free < 1024**3:
                        row["status"] = "resource_failed"
                        job.terminate()
                        break
                    time.sleep(.5)
                row.update(exit_code=job.poll(), wall_seconds=time.monotonic() - started,
                           process_evidence=job.observe())
                if row["status"] == "running":
                    row["status"] = "passed" if row["exit_code"] == 0 else "failed"
            save()
            if row["status"] != "passed":
                raise RuntimeError(name + " check failed; evidence preserved, no retry")
    result["status"] = "passed"
    save()
    print(json.dumps({"status": "passed", "checks": len(commands), "result": str(report)}))


if __name__ == "__main__":
    main()
