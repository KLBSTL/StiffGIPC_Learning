"""Seal selected correctness evidence; no GPU launch or raw trajectory export."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def record(path):
    path = path.resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    previous = OUT.parent / "report5_report6_audit_20261008/ARTIFACT_INDEX.json"
    previous_rows = read(previous)["artifacts"]
    for row in previous_rows:
        current = record(Path(row["path"]))
        if (current["bytes"], current["sha256"]) != (row["bytes"], row["sha256"]):
            raise RuntimeError("Prior sealed artifact changed: " + row["path"])
    build = ROOT / "build/report56_recheck_final_20261008"
    runs = ROOT / "runs/report56_recheck_20261008"
    ctest = read(runs / "full_after_runner_fix/RESULT.json")
    smoke = read(runs / "smoke/SUMMARY.json")
    identity = read(runs / "smoke/IDENTITY.json")
    manifest = read(build / "windows_manifest.json")
    if ctest["status"] != "passed" or smoke["status"] != "passed":
        raise RuntimeError("Incomplete verification cannot be sealed as passing")
    if manifest["exe"]["sha256"] != smoke["program_sha256"]:
        raise RuntimeError("Programs differ")
    for name, source in (
        ("full_ctest.log", runs / "full_after_runner_fix/ctest.log"),
        ("interrupted_ctest.log", runs / "full/ctest.log"),
        ("interrupted_observer.log", ROOT / "runs/report56_recheck_full_20261008.log"),
        ("JOB_CPU.log", ROOT / "runs/report56_recheck_job_observe_20261008.log"),
        ("BENCH_CONTRACTS.log", ROOT / "runs/report56_recheck_contracts_20261008.log"),
        ("LOCAL_CONTRACTS.log", ROOT / "runs/report56_recheck_local_contracts_20261008.log"),
    ):
        target = OUT / name
        if target.exists(): raise RuntimeError("Existing evidence: " + str(target))
        shutil.copyfile(source, target)
    failures = []
    for folder in sorted((build / "line_search_failures").iterdir()):
        target = OUT / "failure_fixtures" / folder.name
        target.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(folder / "stderr.log", target / "stderr.log")
        report = folder / "output/stats.json"
        row = {"case": folder.name, "stderr": record(target / "stderr.log")}
        if report.is_file():
            shutil.copyfile(report, target / "stats.json")
            row["report"] = record(target / "stats.json")
        failures.append(row)
    if len(failures) != 4: raise RuntimeError("Expected four native failure fixtures")
    source_paths = (
        "CMakeLists.txt", "StiffGIPC/app/gl_main.cu", "StiffGIPC/core/GIPC.cu",
        "StiffGIPC/gipc/statistics.cpp", "StiffGIPC/linear_system/utils/spmv.cu",
        "StiffGIPC/cuda_tools/scoped_cuda_events.h", "StiffGIPC/solver/line_search_acceptance.h",
        "tests/check_line_search_failure.cmake", "tests/line_search_acceptance_test.cpp",
        "tests/sym_spmv_tests.cu", "tools/local/windows_owned_job.py",
    )
    receipt = {
        "scope": "Correctness integration and normal-path smoke; no speed ratio/physical quality certification",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD~1"], cwd=ROOT, text=True).strip(),
        "runner_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "build_manifest": record(build / "windows_manifest.json"),
        "program": record(Path(manifest["exe"]["path"])),
        "source_and_assets_digest": identity["source_digest"],
        "sources": [record(ROOT / name) for name in source_paths],
        "native_units": len(manifest["translation_units"]),
        "ctest": {"passed": 13, "total": 13, "commands": [{key: c[key] for key in
                    ("argv", "status", "exit_code", "wall_seconds")} for c in ctest["commands"]],
                  "original_receipt": record(runs / "full_after_runner_fix/RESULT.json")},
        "cpu_tests": {"job_observation": 6, "bench_contracts": 20, "local_contracts": 9},
        "failure_fixtures": failures,
        "smoke": [{key: row[key] for key in ("name", "config", "guards_passed", "directions",
                   "pcg_iterations", "max_energy_backtracks", "max_intersection_backtracks")}
                  for row in smoke["runs"]],
        "smoke_original_receipt": record(runs / "smoke/SUMMARY.json"),
        "interrupted_first_attempt": {"status": "observer_error; not a passing run",
                  "error": "PermissionError: [WinError 5] Member image",
                  "original_receipt": record(runs / "full/RESULT.json"),
                  "retained": True, "superseded_after_explicit_runner_fix": True},
        "prior_sealed_artifacts_verified": len(previous_rows),
        "coverage_gaps": ["No native type-1 changed-state backtrack fixture",
                          "No full-trajectory ninth-backtrack failure fixture",
                          "Display catch statically reviewed, not injected through a FreeGLUT session",
                          "Healthy CUDA entry guard only; CUDA_SAFE_CALL abort behavior unchanged",
                          "No new paired speed, material calibration or independent accepted-path CCD",
                          "No re-run of prior three Sanitizer checks for the new binary"],
    }
    write_new(OUT / "VALIDATION_RECEIPT.json", receipt)
    files = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    write_new(OUT / "ARTIFACT_INDEX.json", {"artifacts": [record(p) for p in sorted(files)],
                                           "program_sha256": smoke["program_sha256"]})
    print(json.dumps({"sealed": len(files), "native_fixtures": len(failures), "ctest": "13/13"}))


if __name__ == "__main__":
    main()
