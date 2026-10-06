"""Summarize the velocity ablation without treating timing as quality proof."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
from physical_metrics import snapshot

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranges(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values} if values else None


def work(run):
    frames = read(run / "output/stats.json")["frames"]
    directions = [n for f in frames for n in f.get("newton", []) if "pcg" in n]
    outer = [t for f in frames for t in f.get("toi", [])]
    return {
        "directions": len(directions), "cg_iterations": sum(n["pcg"]["iterations"] for n in directions),
        "pcg_limit_hits": sum(n["pcg"].get("iteration_limit", False) for n in directions),
        "outer_limit_hits": sum(f.get("toi_exit") == "iteration_limit" or f.get("newton_exit") == "iteration_limit" for f in frames),
        "outer": len(outer), "velocity_exits": sum(n.get("inner_exit_reason") == "velocity_converged" for n in directions),
        "full_step_exits": sum(n.get("inner_exit_reason") == "full_step" for n in directions),
        "backtrack_directions": sum(n.get("line_search_backtracks", 0) > 0 for n in directions),
        "backtracks": sum(n.get("line_search_backtracks", 0) for n in directions),
        "maximum_inner_iterations": max((t["inner_iterations"] for t in outer), default=0),
        "safe_ccd_pair_visits": sum(t.get("safe_ccd_broad_pairs", 0) for t in outer),
        "active_update_pair_visits": sum(t.get("active_update_broad_pairs", 0) for t in outer),
        "active_added": sum(t.get("active_added", 0) for t in outer),
        "active_removed": sum(t.get("active_removed", 0) for t in outer),
        "maximum_active_contacts": max((t.get("active_self", 0) + t.get("active_ground", 0) for t in outer), default=0),
        "effective_inner_policies": sorted(set(f.get("toi_inner_stop_policy", "ipc") for f in frames)),
        "effective_velocity_tolerances": sorted(set(f.get("robust_trial_velocity_tol", 0) for f in frames)),
        "geometric_contact_frames": sum(f.get("contact_geometry", {}).get("geometric_contact", False) for f in frames),
        "frames": frames,
    }


class States:
    def __init__(self, run):
        self.run = run
        self.requested = read(run / "requested.json")
        self.result = read(run / "result.json")
        self.dt = self.requested["dt"]
        directory = run / "trace"
        paths = sorted(directory.glob("state_*.bin"))
        self.indices = [int(p.stem[-4:]) for p in paths]
        assert self.indices == list(range(self.result["recorded_frames"] + 1)), run
        self.positions = np.stack([np.fromfile(p, dtype="<f8").reshape(-1, 3) for p in paths])
        self.mass = np.fromfile(directory / "masses.bin", dtype="<f8")
        self.fixed = np.fromfile(directory / "boundary_types.bin", dtype="<i4") == 1
        self.moving = ~self.fixed
        self.scale = float(np.linalg.norm(np.ptp(self.positions[0], axis=0)))
        self.velocity = np.diff(self.positions, axis=0) / self.dt
        raw = np.fromfile(directory / "topology.bin", dtype="<u4")
        nv, nf, nt = map(int, raw[:3])
        self.faces = raw[3:3 + 3 * nf].reshape(-1, 3)
        self.tets = raw[3 + 3 * nf:].reshape(-1, 4)
        assert len(self.positions[0]) == nv and len(self.tets) == nt
        in_tet = np.zeros(nv, dtype=bool)
        in_tet[self.tets.ravel()] = True
        cloth = self.faces[~in_tet[self.faces].any(axis=1)]
        self.cloth = np.zeros(nv, dtype=bool)
        self.cloth[np.unique(cloth)] = True
        self.cloth_scale = float(np.linalg.norm(np.ptp(self.positions[0, self.cloth], axis=0)))
        # This ablation's only tetrahedral object is an entirely fixed ABD sphere.
        # Native boundary_types does not encode its generalized-coordinate lock.
        scene = read(run / "output/scene.json")
        abd = [o for o in scene["objects"] if o["body_type"] == "ABD"]
        fem = [o for o in scene["objects"] if o["body_type"] == "FEM"]
        body_ids = np.fromfile(directory / "body_ids.bin", dtype="<i4")
        assert self.requested["scene"] == "cloth_sphere7_l"
        assert len(abd) == len(fem) == 1 and abd[0]["fixed_mode"] == "all"
        assert fem[0]["dimension"] == 2 and fem[0]["fixed_mode"] == "none"
        assert np.all(body_ids[in_tet] == 0) and np.all(body_ids[self.cloth] == -1)
        assert np.all(in_tet | self.cloth), "Unexpected unclassified vertices"
        self.fixed_geometry = in_tet | self.fixed
        edges = np.concatenate([cloth[:, [0, 1]], cloth[:, [1, 2]], cloth[:, [2, 0]]])
        self.edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.edge_rest = np.linalg.norm(self.positions[0, self.edges[:, 0]] - self.positions[0, self.edges[:, 1]], axis=1)

    def rms(self, values, mask=None):
        mask = self.moving if mask is None else mask
        return np.sqrt(np.average(np.sum(values[:, mask] ** 2, axis=2), axis=1, weights=self.mass[mask]))

    def physical(self):
        stretch = np.linalg.norm(self.positions[:, self.edges[:, 0]] - self.positions[:, self.edges[:, 1]], axis=2) / self.edge_rest
        fixed_motion = np.linalg.norm(self.positions[:, self.fixed] - self.positions[0, self.fixed], axis=2)
        kinetic = .5 * np.sum(self.mass[self.moving] * np.sum(self.velocity[:, self.moving] ** 2, axis=2), axis=1)
        tet = self.positions[:, self.tets]
        determinant = np.sum((tet[:, :, 1] - tet[:, :, 0]) * np.cross(tet[:, :, 2] - tet[:, :, 0], tet[:, :, 3] - tet[:, :, 0]), axis=2)
        jacobian = determinant / determinant[0]
        final = snapshot(self.run)[0]
        return {
            "finite_all_frames": bool(np.isfinite(self.positions).all()),
            "fixed_max_displacement_all_frames_m": float(fixed_motion.max(initial=0)),
            "free_ground_min_gap_all_frames_m": float((self.positions[:, self.moving, 1] + 1).min()),
            "cloth_max_edge_stretch_all_frames": float(stretch.max()),
            "cloth_max_edge_stretch_final": float(stretch[-1].max()),
            "cloth_edge_stretch_p95_max_all_frames": float(np.percentile(stretch, 95, axis=1).max()),
            "cloth_edge_stretch_p95_final": float(np.percentile(stretch[-1], 95)),
            "tet_relative_jacobian_min_all_frames": float(jacobian.min()),
            "tet_nonpositive_jacobian_frame_events": int((jacobian <= 0).sum()),
            "cloth_vertices": int(self.cloth.sum()),
            "cloth_mass_kg": float(self.mass[self.cloth].sum()),
            "fixed_abd_mass_kg": float(self.mass[self.fixed_geometry].sum()),
            "cloth_initial_bbox_diagonal_m": self.cloth_scale,
            "fixed_abd_max_displacement_all_frames_m": float(np.linalg.norm(self.positions[:, self.fixed_geometry] - self.positions[0, self.fixed_geometry], axis=2).max(initial=0)),
            "cloth_max_native_step_velocity_rms_m_s": float(self.rms(self.velocity, self.cloth).max()),
            "cloth_final_native_step_velocity_rms_m_s": float(self.rms(self.velocity, self.cloth)[-1]),
            "maximum_free_native_step_velocity_rms_m_s": float(self.rms(self.velocity).max()),
            "kinetic_proxy_final_J": float(kinetic[-1]),
            "kinetic_proxy_max_J": float(kinetic.max()),
            "final": final,
            "velocity_scope": "one native-step position difference, matching v=(x_n-x_previous)/dt; internal q/v not exported",
            "energy_scope": "mass-weighted position-derived kinetic proxy; not total mechanical energy",
        }

    def compare(self, reference):
        assert np.array_equal(self.positions[0], reference.positions[0]), (self.run, reference.run)
        assert np.array_equal(self.mass, reference.mass) and np.array_equal(self.fixed, reference.fixed)
        ratio = self.dt / reference.dt
        indices = np.rint(np.asarray(self.indices) * ratio).astype(int)
        assert np.allclose(indices * reference.dt, np.asarray(self.indices) * self.dt, atol=1e-12)
        assert indices[-1] < len(reference.positions)
        difference = self.positions - reference.positions[indices]
        rms = self.rms(difference)
        velocity_difference = self.velocity - reference.velocity[indices[1:] - 1]
        vrms = self.rms(velocity_difference)
        ref_v = self.rms(reference.velocity[indices[1:] - 1])
        relative = vrms / np.maximum(ref_v, 1e-12)
        all_mass_rms = np.sqrt(np.average(np.sum(difference ** 2, axis=2), axis=1, weights=self.mass))
        assert np.array_equal(self.cloth, reference.cloth)
        cloth_rms = self.rms(difference, self.cloth)
        cloth_vrms = self.rms(velocity_difference, self.cloth)
        cloth_ref_v = self.rms(reference.velocity[indices[1:] - 1], self.cloth)
        cloth_relative = cloth_vrms / np.maximum(cloth_ref_v, 1e-12)
        cloth_vertex_error = np.linalg.norm(difference[:, self.cloth], axis=2)
        return {
            "reference_run": reference.run.name, "reference_dt": reference.dt, "compared_states": len(indices),
            "physical_time_end_s": float(indices[-1] * reference.dt), "initial_state_identical": True,
            "max_free_mass_rms_m": float(rms.max()), "max_free_mass_rms_over_scale": float(rms.max() / self.scale),
            "final_free_mass_rms_m": float(rms[-1]), "final_free_mass_rms_over_scale": float(rms[-1] / self.scale),
            "max_vertex_difference_m": float(np.linalg.norm(difference, axis=2).max()),
            "max_all_mass_rms_over_scale": float(all_mass_rms.max() / self.scale),
            "max_velocity_mass_rms_m_s": float(vrms.max()), "final_velocity_mass_rms_m_s": float(vrms[-1]),
            "max_velocity_relative_reference": float(relative.max()),
            "final_velocity_relative_reference": float(relative[-1]),
            "reference_final_velocity_rms_m_s": float(ref_v[-1]),
            "existing_1pct_position_budget_pass": bool(rms.max() / self.scale <= .01),
            "existing_1pct_velocity_budget_pass": bool(relative.max() <= .01),
            "existing_1e6_trajectory_equivalence_diagnostic_pass": bool(all_mass_rms.max() / self.scale <= 1e-6),
            "cloth_only": {
                "vertices": int(self.cloth.sum()), "mass_kg": float(self.mass[self.cloth].sum()),
                "initial_bbox_diagonal_m": self.cloth_scale,
                "max_mass_rms_m": float(cloth_rms.max()), "final_mass_rms_m": float(cloth_rms[-1]),
                "max_mass_rms_over_cloth_scale": float(cloth_rms.max() / self.cloth_scale),
                "final_mass_rms_over_cloth_scale": float(cloth_rms[-1] / self.cloth_scale),
                "max_vertex_difference_m": float(cloth_vertex_error.max()),
                "final_vertex_error_p95_m": float(np.percentile(cloth_vertex_error[-1], 95)),
                "max_velocity_mass_rms_m_s": float(cloth_vrms.max()),
                "final_velocity_mass_rms_m_s": float(cloth_vrms[-1]),
                "reference_final_velocity_rms_m_s": float(cloth_ref_v[-1]),
                "max_velocity_relative_reference": float(cloth_relative.max()),
                "final_velocity_relative_reference": float(cloth_relative[-1]),
                "suggested_1pct_position_budget_pass": bool(cloth_rms.max() / self.cloth_scale <= .01),
                "suggested_1pct_velocity_budget_pass": bool(cloth_relative.max() <= .01),
                "scope": "Deformable cloth only, excluding scene-locked ABD sphere. Boundary flags alone fail to encode the sphere lock. Both absolute and relative velocity errors are reported; the refined reference still needs convergence qualification.",
            },
            "legacy_boundary_mask_warning": "Legacy free/all-mass RMS includes the fixed ABD sphere (boundary_types=0), diluting cloth position and absolute velocity errors. Do not use it alone to qualify deformable-body quality.",
            "scope": "All output times aligned; velocities use each solver's own native step. Different-stop ablations need physical quality, not bitwise equivalence. Refined base is diagnostic, not an exact solution.",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_velocity_20261001.json")
    parser.add_argument("--stage", choices=["pilot", "measure"], default="measure")
    args = parser.parse_args()
    cfg = read(args.config)
    matrix = ROOT / "runs/local" / cfg["name"]
    status = read(matrix / (args.stage + "_status.json"))
    cache = {}

    def states(run):
        if run not in cache:
            cache[run] = States(run)
        return cache[run]

    def path(label, stage, rep=1):
        return ROOT / "runs/local" / f"{cfg['name']}_{label}_{stage}_r{rep:02d}"

    rows = []
    for item in status["runs"]:
        row = {k: item.get(k) for k in ["variant", "repeat", "name", "status"]}
        if not item.get("name"):
            rows.append(row)
            continue
        run = ROOT / "runs/local" / item["name"]
        result = read(run / "result.json")
        row["result"] = result
        if result["status"] == "completed":
            expected = cfg["pilot_frames"] if args.stage == "pilot" else cfg["measure_frames"]
            assert result["recorded_frames"] == expected and result["finite"]
            requested = read(run / "requested.json")
            assert requested["source_digest"] == cfg["source_digest"]
            assert requested["exe_sha256"] == cfg["binary_sha256"]["base" if row["variant"] == "base" else "fused"]
            for key, expected_value in [("dt", cfg["dt"]), ("tol", cfg["newton_tol"]), ("pcg_tol", cfg["pcg_tol"]), ("suite", cfg["suite"])]:
                assert requested[key] == expected_value, (run, key)
            details = work(run)
            frame_records = details.pop("frames")
            row["work"] = details
            row["physics"] = states(run).physical()
            variant = next(v for v in cfg["variants"] if v["id"] == row["variant"])
            assert requested["robust_velocity_tol"] == variant["velocity_tol"]
            assert requested["robust_velocity_stop"] == variant["velocity_stop"]
            if row["variant"] != "base":
                assert details["effective_velocity_tolerances"] == [variant["velocity_tol"] or 0]
                assert details["effective_inner_policies"] == ["robust_velocity" if variant["velocity_stop"] else "full_step"]
            base = path("base", args.stage, row["repeat"])
            full = path("full", args.stage, row["repeat"])
            if (base / "result.json").exists() and read(base / "result.json")["status"] == "completed":
                row["vs_same_dt_base"] = states(run).compare(states(base))
                with (base / "trace/frames.csv").open() as stream:
                    bt = [float(r["solver_ms"]) / 1000 for r in csv.DictReader(stream)]
                with (run / "trace/frames.csv").open() as stream:
                    ct = [float(r["solver_ms"]) / 1000 for r in csv.DictReader(stream)]
                bf = work(base)["frames"]
                near = [f.get("contact_geometry", {}).get("native_narrow_self_pairs", 0) + f.get("contact_geometry", {}).get("native_narrow_ground_pairs", 0) > 0 for f in bf]
                geom = [f.get("contact_geometry", {}).get("geometric_contact", False) for f in bf]
                row["common_base_phases"] = {}
                for label, mask in [("no_native_near", [not x for x in near]), ("native_near", near), ("geometric_contact", geom)]:
                    bs = sum(t for t, keep in zip(bt, mask) if keep)
                    cs = sum(t for t, keep in zip(ct, mask) if keep)
                    row["common_base_phases"][label] = {"frames": sum(mask), "base_seconds": bs, "candidate_seconds": cs, "raw_base_over_candidate": bs / cs if cs else None}
            if row["variant"] != "full" and (full / "result.json").exists() and read(full / "result.json")["status"] == "completed":
                row["vs_default_toi"] = states(run).compare(states(full))
            if args.stage == "measure":
                for label, stage in [("vs_refined_base_dt2", "reference_half"), ("vs_refined_base_dt4", "reference")]:
                    ref = path("base", stage)
                    if (ref / "result.json").exists() and read(ref / "result.json")["status"] == "completed":
                        row[label] = states(run).compare(states(ref))
            row["raw_csv_sha256"] = sha(run / "trace/frames.csv")
            row["raw_stats_sha256"] = sha(run / "output/stats.json")
        rows.append(row)
    summaries = []
    for variant in cfg["variants"]:
        label = variant["id"]
        selected = [r for r in rows if r["variant"] == label and r["status"] == "completed"]
        paired_base, paired_full = [], []
        for row in selected:
            for ref_label, target in [("base", paired_base), ("full", paired_full)]:
                ref = next((r for r in rows if r["variant"] == ref_label and r["repeat"] == row["repeat"] and r["status"] == "completed"), None)
                if ref:
                    target.append(ref["result"]["solver_seconds"] / row["result"]["solver_seconds"])
        summaries.append({"variant": label, "completed_runs": len(selected),
            "solver_seconds": ranges([r["result"]["solver_seconds"] for r in selected]),
            "directions": ranges([r["work"]["directions"] for r in selected]),
            "cg_iterations": ranges([r["work"]["cg_iterations"] for r in selected]),
            "outer": ranges([r["work"]["outer"] for r in selected]),
            "velocity_exits": ranges([r["work"]["velocity_exits"] for r in selected]),
            "paired_base_over_candidate": ranges(paired_base), "paired_default_toi_over_candidate": ranges(paired_full),
            "worst_cloth_edge_stretch": max((r["physics"]["cloth_max_edge_stretch_all_frames"] for r in selected), default=None),
            "quality_matched_speedup": "N/A: algorithm, reference convergence and full physical gates require separate qualification"})
    controls = {}
    if args.stage == "measure":
        for label in [v["id"] for v in cfg["variants"]]:
            r1, r2 = path(label, "measure", 1), path(label, "measure", 2)
            if all((r / "result.json").exists() and read(r / "result.json")["status"] == "completed" for r in [r1, r2]):
                controls[label + "_same_mode_repeat"] = states(r2).compare(states(r1))
        half, fine = path("base", "reference_half"), path("base", "reference")
        if all((r / "result.json").exists() and read(r / "result.json")["status"] == "completed" for r in [half, fine]):
            controls["strict_base_dt2_vs_dt4"] = states(half).compare(states(fine))
    safety = {}
    for variant in cfg["variants"]:
        report = ROOT / "reports" / f"velocity_20261001_{variant['id']}_accepted_ccd.json"
        if report.exists():
            safety[variant["id"]] = read(report)
    payload = {"config": cfg, "stage": args.stage, "status": status["status"], "summaries": summaries,
        "runs": rows, "controls": controls, "independent_safety_runs": safety,
        "scope": "Same native-step dt/physics for timed arms; separate strict dt/2 and dt/4 base diagnostics; quality is not inferred from time.",
        "known_limitations": ["Existing desktop graphics load; serial rotated repeats do not guarantee GPU exclusivity", "MAS same-mode trajectory noise exists", "refined-dt base is not an exact reference", "Kinetic proxy does not establish total energy or friction equivalence", "Separate substep safety runs can have different trajectories from timing runs", "Legacy boundary-mask whole-scene RMS includes the fixed ABD sphere; cloth-only metrics must be used for cloth fidelity"]}
    output = ROOT / "reports" / f"VELOCITY_ABLATION_20261001_{args.stage}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["variant", "repeat", "status", "solver_seconds", "directions", "cg_iterations", "outer", "velocity_exits", "max_stretch", "legacy_ref_dt4_position_pct", "ref_dt4_cloth_position_pct", "ref_dt4_cloth_velocity_pct", "ref_dt4_cloth_final_velocity_error_m_s"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ref = row.get("vs_refined_base_dt4", {})
            writer.writerow({"variant": row["variant"], "repeat": row["repeat"], "status": row["status"],
                "solver_seconds": row.get("result", {}).get("solver_seconds"),
                **{k: row.get("work", {}).get(k) for k in ["directions", "cg_iterations", "outer", "velocity_exits"]},
                "max_stretch": row.get("physics", {}).get("cloth_max_edge_stretch_all_frames"),
                "legacy_ref_dt4_position_pct": 100 * ref["max_free_mass_rms_over_scale"] if ref else None,
                "ref_dt4_cloth_position_pct": 100 * ref["cloth_only"]["max_mass_rms_over_cloth_scale"] if ref else None,
                "ref_dt4_cloth_velocity_pct": 100 * ref["cloth_only"]["max_velocity_relative_reference"] if ref else None,
                "ref_dt4_cloth_final_velocity_error_m_s": ref["cloth_only"]["final_velocity_mass_rms_m_s"] if ref else None})
    print(json.dumps({"output": str(output), "stage": args.stage, "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
