"""Read-only cost bounds from the completed seven-scene experiment.

These are Amdahl planning bounds, not speed predictions or quality certificates.
Phase intervals are the original solver fields; they are not kernel attribution.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "autodl_seven_20261008" / "TIMINGS.csv"
SCENES = (
    "cloth_hang_l", "cloth_hang_m", "cloth_sphere7_l", "cloth_sphere7_m",
    "cloth_fixed_bunny_l", "cloth_fixed_bunny_m", "bunny_cloth_bunny_l",
)


def analyze(rows):
    index = {}
    for row in rows:
        key = (row["scene"], row["arm"])
        if key in index or key[0] not in SCENES or key[1] not in ("base", "graph", "all"):
            raise ValueError("Unexpected or duplicate experiment row")
        if row["status"] != "completed" or row["hard_checks_passed"] != "True":
            raise ValueError("Incomplete experiment cannot define a cost bound")
        for name in ("solver_seconds", "linear_seconds", "ccd_seconds", "line_search_seconds", "assembly_seconds"):
            value = float(row[name])
            if not math.isfinite(value) or value < 0 or (name == "solver_seconds" and value == 0):
                raise ValueError("Invalid timing field: " + name)
        index[key] = row
    if len(index) != 21:
        raise ValueError("Exactly seven scenes and three arms are required")
    result = []
    for scene in SCENES:
        base, graph, all_ = (index[(scene, arm)] for arm in ("base", "graph", "all"))
        tb, tg, ta = (float(row["solver_seconds"]) for row in (base, graph, all_))
        linear = float(all_["linear_seconds"])
        collision_ls = float(all_["ccd_seconds"]) + float(all_["line_search_seconds"])
        required = max(0.0, ta - tb / 2)
        if linear >= ta or collision_ls >= ta:
            raise ValueError("Invalid phase envelope; cannot calculate Amdahl bound")
        result.append({
            "scene": scene, "base_seconds": tb, "graph_seconds": tg, "all_seconds": ta,
            "graph_vs_base": tb / tg, "all_vs_base": tb / ta,
            "components_vs_graph": tg / ta, "two_x_target_seconds": tb / 2,
            "seconds_still_to_save": required, "fraction_of_all_to_save": required / ta,
            "linear_fraction": linear / ta, "ccd_line_search_fraction": collision_ls / ta,
            "assembly_fraction": float(all_["assembly_seconds"]) / ta,
            "linear_only_fraction_to_remove": required / linear if linear else None,
            "ideal_zero_linear_speedup_vs_base": tb / (ta - linear),
            "speedup_vs_base_if_linear_and_collision_ls_halved": tb / (ta - .5 * (linear + collision_ls)),
        })
    return result


def main():
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = analyze(list(csv.DictReader(stream)))
    output = {
        "status": "verified_arithmetic_only", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "scope": "Single-sample planning bounds; no kernel attribution, speed prediction, or quality certification",
        "rows": rows,
    }
    (HERE / "COST_BUDGET.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("scene all/base remaining% linear% CCD+LS% linear-only-removal%")
    for row in rows:
        print(f"{row['scene']} {row['all_vs_base']:.3f} {100*row['fraction_of_all_to_save']:.1f} "
              f"{100*row['linear_fraction']:.1f} {100*row['ccd_line_search_fraction']:.1f} "
              f"{100*row['linear_only_fraction_to_remove']:.1f}")


if __name__ == "__main__":
    main()
