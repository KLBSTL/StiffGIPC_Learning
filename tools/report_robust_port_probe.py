"""Verify the isolated probe records and plot measured, unqualified costs."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = [ROOT / "reports/ROBUST_STANDALONE_POTENTIAL_20261001.json",
              ROOT / "reports/ROBUST_STANDALONE_POTENTIAL_20261001_60F.json"]
    all_data = [read(path) for path in inputs]
    verified = []
    output_rows = []
    for data in all_data:
        steps = data["protocol"]["frames"]
        assert not data["speed_qualified"] and not data["quality_verified"]
        assert len(data["runs"]) == 12 and data["all_12_completed"]
        for row in data["runs"]:
            run = ROOT / row["run"]
            with (run / ("frames.csv" if row["arm"].startswith("robust") else "trace/frames.csv")).open(encoding="utf-8-sig", newline="") as stream:
                frames = list(csv.DictReader(stream))
            assert len(frames) == steps
            assert row["measured_window_seconds"] > 0 and row["directions"] > 0
            if row["arm"].startswith("robust"):
                result = read(run / "probe_result.json")
                assert result["newton_cap_log_hits"] == result["linear_cap_log_hits"] == 0
                assert result["exit_code"] == 0
                cfg = read(run / "robust_requested_config.json")["config"]
                assert cfg["dt"] == .01 and cfg["linear_system"]["tol_rate"] == .001
            else:
                request = read(run / "requested.json")
                assert request["tol"] == .01 and request["pcg_tol"] == 1e-4
                stats = read(run / "output/stats.json")["frames"]
                assert not any(n["pcg"].get("iteration_limit", False) for f in stats for n in f["newton"] if "pcg" in n)
            verified.append({"run": str(run), "frames": steps, "frames_sha256": sha(run / ("frames.csv" if row["arm"].startswith("robust") else "trace/frames.csv"))})
        for summary in data["summary"]:
            rows = [r for r in data["runs"] if r["arm"] == summary["arm"]]
            assert len(rows) == 3
            assert statistics.median(r["measured_window_seconds"] for r in rows) == summary["seconds_median"]
            output_rows.append({"frames": steps, **summary})
    audit = read(ROOT / "reports/ROBUST_PORT_SOURCE_AUDIT_20261001.json")
    assert audit["finite_differences"][-1]["public_gradient_relative_error"] > .4
    assert audit["finite_differences"][-1]["edge_edge_gradient_relative_error"] < 1e-7
    data = all_data[1]
    arms = ["base", "base_graph", "robust_v005", "robust_v1"]
    labels = ["Base", "Base + Graph", "Robust\nv_tol=0.05", "Robust\nv_tol=1.0"]
    colors = ["#3f5266", "#227a9f", "#e09335", "#2d8a64"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, field, title, unit in zip(axes, ["measured_window_seconds", "directions"],
                                    ["Measured window cost", "Direction solves"], ["seconds", "count"]):
        values = [[r[field] for r in data["runs"] if r["arm"] == arm] for arm in arms]
        medians = [statistics.median(v) for v in values]
        ax.bar(range(4), medians, color=colors, width=.65, alpha=.82)
        for i, (v, median) in enumerate(zip(values, medians)):
            ax.scatter([i - .1, i, i + .1], v, color="#162839", s=16, zorder=3)
            ax.text(i, max(v) * 1.04, f"{median:.3f}" if field.endswith("seconds") else str(median), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(4), labels)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(max(v) for v in values) * 1.2)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("60-frame local probe: 3 serial interleaved repeats", fontsize=13)
    fig.text(.5, .015, "Diagnostic only: cross-engine physics and timing boundaries are not matched.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .07, 1, .92])
    plots = ROOT / "reports/figures"
    plots.mkdir(exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(plots / f"ROBUST_PORT_POTENTIAL_20261001.{suffix}", dpi=220)
    plt.close(fig)
    result = {"runs_verified": len(verified), "total_measured_frames": sum(r["frames"] for r in verified),
              "source_formula_check_verified": True, "no_observed_iteration_cap": True,
              "quality_gate_checked": False, "qualified_speedup": False,
              "raw_inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs], "verified": verified}
    write_path = ROOT / "reports/ROBUST_PORT_PROBE_VERIFICATION_20261001.json"
    write_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "reports/ROBUST_PORT_POTENTIAL_20261001.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [k for k in output_rows[0] if k != "paired_raw_ratio_range"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ["verified", "raw_inputs"]}))


if __name__ == "__main__":
    main()
