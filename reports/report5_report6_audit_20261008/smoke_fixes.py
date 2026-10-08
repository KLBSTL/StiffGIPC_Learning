"""One explicit serial smoke suite for the corrected native solver.

Uses the unchanged Windows runner, guards and configuration validator.
No baseline comparison or speed/quality certification; no retries on failure.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in ("bench", "diagnostic", "full_eval", "local"):
    sys.path.insert(0, str(ROOT / "tools" / folder))
from config import expand
from local_identity import active_identity, verify
from windows_runner import execute, gpu_lock


def main():
    build = ROOT / "build/report56_audit_20261008"
    out = ROOT / "runs/report56_audit_20261008/smoke"
    if out.exists():
        raise RuntimeError("Smoke evidence exists; no automatic retry or overwrite")
    out.mkdir()
    identity = active_identity(ROOT, build, build / "Release/gipc.exe", build / "build.log")
    (out / "IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    tasks = []
    for name, scene, steps, preset in (
        ("sphere_host", "cloth_sphere7_l", 30, "host"),
        ("sphere_graph", "cloth_sphere7_l", 30, "graph"),
        ("fixed_all", "cloth_fixed_bunny_l", 60, "combined"),
        ("mixed_all", "bunny_cloth_bunny_l", 35, "combined"),
    ):
        c = {"scene": scene, "steps": steps, "preset": preset}
        if preset == "combined": c["discrete_bvh_refit"] = True
        tasks.append({"name": name, "binary": "active", "config": expand(c)})
    summary = {"scope": "From-zero shared-desktop correctness smoke only; no speed ratio or physical quality certification",
               "program_sha256": identity["exe"]["sha256"], "tasks": tasks, "runs": [], "status": "running"}
    report = out / "SUMMARY.json"
    with gpu_lock(ROOT):
        for task in tasks:
            result = execute(out, task, identity, 0)
            row = {"name": task["name"], "result": result, "numeric_guards_passed": False}
            summary["runs"].append(row)
            if result["status"] == "completed":
                frames = json.loads((out / task["name"] / "output/stats.json").read_text(encoding="utf-8"))["frames"]
                newtons = [n for f in frames for n in f.get("newton", [])]
                pcgs = [n["pcg"] for n in newtons if "pcg" in n]
                row.update(numeric_guards_passed=bool(pcgs) and
                    not any(p.get("iteration_limit") or p.get("breakdown") for p in pcgs) and
                    not any(f.get("newton_exit") == "iteration_limit" for f in frames) and
                    not any(n.get("line_search_failure") for n in newtons),
                    directions=len(pcgs), pcg_iterations=sum(p["iterations"] for p in pcgs),
                    max_energy_backtracks=max((n.get("energy_backtracks", 0) for n in newtons), default=0),
                    max_intersection_backtracks=max((n.get("intersection_backtracks", 0) for n in newtons), default=0))
            summary["status"] = "running" if row["numeric_guards_passed"] else "failed"
            report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if not row["numeric_guards_passed"]:
                raise RuntimeError("Smoke failed: " + task["name"] + "; no retry, remaining tasks skipped")
        verify(identity["sources"] + [identity["exe"]] + identity["dlls"])
    summary["status"] = "passed"
    summary["runner_sha256"] = hashlib.sha256((ROOT / "tools/local/windows_runner.py").read_bytes()).hexdigest()
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "runs": len(summary["runs"]), "summary": str(report)}))


if __name__ == "__main__":
    main()
