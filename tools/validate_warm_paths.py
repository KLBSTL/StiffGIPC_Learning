"""Independent CPU path audit for completed v31 runs, at most two jobs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate(entry):
    variant = entry["variant"]
    run = ROOT / "runs/local" / entry["name"]
    output = ROOT / "reports" / f"v31_{variant}_accepted_ccd.json"
    command = [str(ROOT / "builds/validator/Release/validate_path.exe"), str(run / "trace"), str(output), "substeps", "--stable-nh1"]
    if output.exists():
        raise RuntimeError(f"Preserve existing validation before rerunning: {output}")
    with (ROOT / "builds" / f"v31_{variant}_ccd.log").open("wb") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return {"variant": variant, "run": entry["name"], "command": command,
            "exit_code": process.returncode, "validation": read(output) if output.exists() else None}


def main():
    cfg = read(ROOT / "configs/local_warm_v31_20261001.json")
    matrix = ROOT / "runs/local" / cfg["name"]
    stage = read(matrix / "diagnose_status.json")
    assert stage["status"] in ["completed", "completed_with_failures"]
    tasks = [entry for entry in stage["runs"] if entry["status"] == "completed"]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(validate, entry) for entry in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            validation = row["validation"] or {}
            print(json.dumps({"variant": row["variant"], "exit_code": row["exit_code"],
                              "passed": validation.get("passed"), "paths": validation.get("paths_checked"),
                              "flags": validation.get("conservative_collision_flags")}), flush=True)
            (matrix / "validation_status.json").write_text(json.dumps({"status": "running", "runs": rows}, indent=2), encoding="utf-8")
    (matrix / "validation_status.json").write_text(json.dumps({"status": "completed", "runs": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
