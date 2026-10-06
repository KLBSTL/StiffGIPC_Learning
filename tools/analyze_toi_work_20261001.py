"""Read-only decomposition and stopping replay of the completed velocity experiment."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def stopping_replay(records, kmin):
    beta = 1.
    for i, record in enumerate(records, 1):
        if i >= kmin:
            beta *= 1 - record["alpha"]
        if beta <= .01:
            return i
    return len(records)


def main():
    rows = []
    for label in ["base", "full", "vel005", "vel100"]:
        for repeat in [1, 2, 3]:
            name = f"local_velocity_20261001_{label}_measure_r{repeat:02d}"
            frames = json.loads((ROOT / "runs/local" / name / "output/stats.json").read_text())["frames"]
            directions = [[n for n in f.get("newton", []) if "pcg" in n] for f in frames]
            outer = [t for f in frames for t in f.get("toi", [])]
            late = frames[21:]
            alphas = [t["alpha"] for f in late for t in f.get("toi", [])]
            first5 = [t["alpha"] for f in late for t in f.get("toi", [])[:5]]
            after5 = [t["alpha"] for f in late for t in f.get("toi", [])[5:]]
            inner = sum(t["inner_iterations"] for t in outer)
            row = {"run": name, "variant": label, "repeat": repeat,
                   "total_directions": sum(map(len, directions)),
                   "first21_directions": sum(map(len, directions[:21])),
                   "last79_directions": sum(map(len, directions[21:])),
                   "outer": len(outer), "inner_iterations_sum": inner,
                   "extra_inner_directions": inner - len(outer),
                   "last79_outer": sum(len(f.get("toi", [])) for f in late),
                   "maximum_outer_per_frame": max((len(f.get("toi", [])) for f in frames), default=0),
                   "maximum_inner_iterations": max((t["inner_iterations"] for t in outer), default=0),
                   "active_added": sum(t.get("active_added", 0) for t in outer),
                   "active_removed": sum(t.get("active_removed", 0) for t in outer)}
            if outer:
                assert inner == row["total_directions"]
                replay = {k: sum(stopping_replay(f["toi"], k) for f in late) for k in [1, 2, 6]}
                assert replay[6] == row["last79_outer"]
                row.update(contact_first5_alpha_median=statistics.median(first5),
                           contact_after5_alpha_median=statistics.median(after5),
                           contact_alpha_median=statistics.median(alphas),
                           frozen_alpha_outer_stopping_replay=replay,
                           frozen_alpha_replay_K2_reduction_fraction=1 - replay[2] / replay[6])
                f = frames[49]
                row["frame50"] = {"directions": len(directions[49]), "outer": len(f["toi"]),
                                  "full_trial_energy_steps": sum(n.get("line_search_r") == 1 for n in directions[49]),
                                  "safe_alphas": [t["alpha"] for t in f["toi"]],
                                  "outer_records": [{k: t.get(k) for k in ["outer", "alpha", "beta", "mu", "active_self", "active_added"]} for t in f["toi"][:6]]}
            else:
                row["contact_alpha_median"] = statistics.median(n["alpha"] for ds in directions[21:] for n in ds)
                row["frame_exit_counts"] = {reason: sum(f.get("newton_exit") == reason for f in frames)
                                            for reason in ["movement", "cumulative_toi"]}
            rows.append(row)
    payload = {"runs": rows,
               "count_definition": "One direction is one Hessian/global linear-system PCG solve; CG iterations are excluded from this count. TOI directions equal outer count plus repeated inner directions.",
               "replay_scope": "Apply alternative Kmin to recorded safe alpha sequences only. No new simulation, no predicted runtime/accuracy; changing stopping changes subsequent trajectories.",
               "alpha_scope": "TOI alpha interpolates safe toward unrestricted trial; base alpha applies a Newton direction. Their geometric targets differ; both drive their own cumulative stopping weights."}
    output = ROOT / "reports/TOI_WORK_DECOMPOSITION_20261001.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k in ["variant", "repeat", "total_directions", "outer", "extra_inner_directions", "contact_first5_alpha_median", "frozen_alpha_replay_K2_reduction_fraction"]}))


if __name__ == "__main__":
    main()
