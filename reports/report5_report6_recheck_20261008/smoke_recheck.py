"""One bounded normal-path check after failure-boundary changes; no retry."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in ("bench", "diagnostic", "full_eval", "local"):
    sys.path.insert(0, str(ROOT / "tools" / folder))
from config import expand
from local_identity import active_identity, verify
from windows_runner import execute, gpu_lock


def main():
    build = ROOT / "build/report56_recheck_final_20261008"
    out = ROOT / "runs/report56_recheck_20261008/smoke"
    out.mkdir(parents=True, exist_ok=False)
    identity = active_identity(ROOT, build, build / "Release/gipc.exe", build / "build.log")
    (out / "IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    summary = {"scope": "Shared-desktop normal-path correctness only; no performance/quality certification",
               "program_sha256": identity["exe"]["sha256"], "runs": [], "status": "running"}
    report = out / "SUMMARY.json"
    with gpu_lock(ROOT):
        for name, scene, steps, preset in (
            ("sphere_graph", "cloth_sphere7_l", 30, "graph"),
            ("fixed_all", "cloth_fixed_bunny_l", 60, "combined"),
        ):
            config = {"scene": scene, "steps": steps, "preset": preset}
            if preset == "combined": config["discrete_bvh_refit"] = True
            task = {"name": name, "binary": "active", "config": expand(config)}
            result = execute(out, task, identity, 0)
            row = {"name": name, "config": task["config"], "result": result, "guards_passed": False}
            summary["runs"].append(row)
            if result["status"] == "completed":
                stats = json.loads((out / name / "output/stats.json").read_text(encoding="utf-8"))
                frames = stats["frames"]
                newtons = [n for f in frames for n in f.get("newton", [])]
                pcgs = [n["pcg"] for n in newtons if "pcg" in n]
                row.update(guards_passed=bool(pcgs) and not stats.get("failure") and
                    not any(p.get("iteration_limit") or p.get("breakdown") for p in pcgs) and
                    not any(f.get("newton_exit") == "iteration_limit" for f in frames) and
                    not any(n.get("line_search_failure") for n in newtons),
                    directions=len(pcgs), pcg_iterations=sum(p["iterations"] for p in pcgs),
                    max_energy_backtracks=max((n.get("energy_backtracks", 0) for n in newtons), default=0),
                    max_intersection_backtracks=max((n.get("intersection_backtracks", 0) for n in newtons), default=0))
            summary["status"] = "running" if row["guards_passed"] else "failed"
            report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if not row["guards_passed"]:
                raise RuntimeError("Normal-path smoke failed; evidence preserved, no retry")
        verify(identity["sources"] + [identity["exe"]] + identity["dlls"])
    summary["status"] = "passed"
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "runs": len(summary["runs"]), "summary": str(report)}))


if __name__ == "__main__":
    main()
