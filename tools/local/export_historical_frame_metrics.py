"""Read existing native logs and positions; never launch or modify a simulation."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(session, output):
    ledger_path = session / "performance_delivery_ledger.json"
    ledger = read(ledger_path)
    assert ledger["complete"] and not ledger["skipped"]
    assert len(ledger["runs"]) == 90
    assert set(ledger["selected_scenes"]) == {"Sphere", "Bunny", "Table", "Hang", "Mixed", "Boxes"}
    output.mkdir(parents=True, exist_ok=False)
    frames_out, newton_out, sources = [], [], []
    position_hashes = {}

    def state(path):
        data = path.read_bytes()
        position_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        values = np.frombuffer(data, dtype=np.float64).reshape(-1, 3)
        assert len(values) and np.isfinite(values).all(), str(path)
        return values

    for run in ledger["runs"]:
        task = run["task"]
        folder = session / task["name"]
        native = read(folder / "result.json")
        assert native["status"] == "completed" and native["recorded_frames"] == 100
        request = read(folder / "requested.json")
        config = request["expanded_config"]
        assert config["steps"] == 100 and config["dt"] == .01 and config["pcg_rho_tol"] == 1e-4
        stats = read(folder / "output/stats.json")["frames"]
        with (folder / "trace/frames.csv").open(newline="", encoding="utf8") as stream:
            timings = list(csv.DictReader(stream))
        assert len(stats) == len(timings) == 100
        sources.append(dict(run=task["name"], exe_sha256=request["exe_sha256"],
            config_sha256=request["config_sha256"], files={str((folder / name).relative_to(ROOT)): digest(folder / name)
            for name in ("requested.json", "result.json", "output/stats.json", "trace/frames.csv")}))
        for frame, (stat, timing) in enumerate(zip(stats, timings), 1):
            assert int(timing["frame"]) == frame
            directions = [n["pcg"] for n in stat["newton"] if "pcg" in n]
            assert directions and all(p.get("frame", frame) == frame for p in directions)
            assert all(not p.get("iteration_limit") and not p.get("breakdown") for p in directions)
            assert stat.get("newton_exit") != "iteration_limit"
            phase = stat.get("phase_ms", {})
            assert all(k in phase for k in ("assembly", "pcg", "ccd", "line_search"))
            stopping = stat.get("ipc_stopping", {})
            count = sum(p["iterations"] for p in directions)
            row = dict(scene=task["scene_key"], method=task["arm"], repeat=task["repeat"],
                frame=frame, dt=config["dt"], pcg_rho_tolerance=config["pcg_rho_tol"],
                outer_rule=config["ipc_termination"], solver_ms=float(timing["solver_ms"]),
                **{k + "_ms": phase.get(k) for k in ("assembly", "pcg", "ccd", "line_search", "state_update")},
                terminal_assembly_ms=stat.get("ipc_exit_assembly_ms"),
                N_pcg_directions=len(directions), newton_record_count=len(stat["newton"]),
                accepted_updates=stopping.get("accepted_updates"), P_pcg_iterations=count,
                P_per_direction=count / len(directions),
                pcg_stage_us_per_iteration=1000 * phase["pcg"] / count if count else None,
                exit_reason=stat.get("newton_exit"), final_beta=stopping.get("final_beta"),
                **{k: sum(n.get(k, 0) for n in stat["newton"]) for k in
                   ("energy_evaluations", "energy_backtracks", "intersection_backtracks")},
                candidate_pairs=int(timing["candidate_pairs"]), ground_candidates=int(timing["ground_candidates"]),
                graph_cache_hits=sum(bool(p.get("graph_cache_hit")) for p in directions) if task["arm"] != "stiff" else None,
                graph_capture_instantiate_host_ms=sum(p.get("graph_capture_instantiate_host_ms", 0) for p in directions) if task["arm"] != "stiff" else None,
                graph_body_executions=sum(p.get("graph_body_executions", 0) for p in directions) if config["execution"] == "conditional_graph" else None,
                trajectory_rms_vs_stiff_mm=None, trajectory_max_vs_stiff_mm=None,
                trajectory_difference_scope=None, source_run=str(folder.relative_to(ROOT)))
            row["unclassified_envelope_ms"] = row["solver_ms"] - sum(phase.values())
            assert math.isfinite(row["solver_ms"]) and row["solver_ms"] > 0
            if task["arm"] == "ours":
                reference = session / f'performance_{task["scene_key"].lower()}_stiff_r{task["repeat"]}'
                a = state(folder / f"trace/state_{frame:04d}.bin")
                b = state(reference / f"trace/state_{frame:04d}.bin")
                assert a.shape == b.shape
                distance = np.linalg.norm(a - b, axis=1)
                row.update(trajectory_rms_vs_stiff_mm=1000 * float(np.sqrt(np.mean(distance ** 2))),
                    trajectory_max_vs_stiff_mm=1000 * float(distance.max()),
                    trajectory_difference_scope="all_vertices_same_round_same_frame; trajectory_difference_not_exact_solution_error")
            frames_out.append(row)
            for index, n in enumerate(stat["newton"]):
                p = n.get("pcg", {})
                newton_out.append(dict(scene=row["scene"], method=row["method"], repeat=row["repeat"],
                    frame=frame, newton_record_index_0=index, pcg_iterations=p.get("iterations"),
                    rho_initial=p.get("rho_initial"), rho_stop=p.get("rho_stop"),
                    alpha=n.get("alpha"), beta=n.get("beta"), active_pairs=n.get("active_pairs"),
                    kappa_before=n.get("kappa_before"), kappa_after=n.get("kappa_after"),
                    native_newton_json=json.dumps(n, ensure_ascii=False, separators=(",", ":"))))
    assert len(frames_out) == 9000

    def csv_file(name, rows):
        with (output / name).open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    csv_file("per_frame.csv", frames_out)
    csv_file("frame100.csv", [r for r in frames_out if r["frame"] == 100])
    csv_file("per_newton.csv", newton_out)
    median_rows = []
    for scene in ledger["selected_scenes"]:
        selected = [r for r in frames_out if r["scene"] == scene and r["method"] == "ours" and r["frame"] == 100]
        assert len(selected) == 3
        median_rows.append(dict(scene=scene, solver_ms=statistics.median(r["solver_ms"] for r in selected),
            N=statistics.median(r["N_pcg_directions"] for r in selected),
            P=statistics.median(r["P_pcg_iterations"] for r in selected),
            observed_min_ms=min(r["solver_ms"] for r in selected), observed_max_ms=max(r["solver_ms"] for r in selected)))
    csv_file("current_graph_frame100_medians.csv", median_rows)
    for source in sources:
        assert all(digest(ROOT / path) == checksum for path, checksum in source["files"].items())
    manifest = dict(complete=True, simulation_runs_launched=0, frames=len(frames_out),
        frame100_rows=90, newton_records=len(newton_out),
        trajectory_difference_rows=1800, strict_reference_error_available=False,
        source_ledger=str(ledger_path.relative_to(ROOT)), source_ledger_sha256=digest(ledger_path),
        exporter_sha256=digest(Path(__file__)), sources=sources, position_sha256=position_hashes,
        outputs={p.name: digest(p) for p in output.glob("*.csv")})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    table = "\n".join(f'| {r["scene"]} | {r["solver_ms"]:.4f} | {r["N"]} | {r["P"]} | {r["observed_min_ms"]:.4f}–{r["observed_max_ms"]:.4f} |' for r in median_rows)
    text = f"""# 历史逐帧数据离线读出（2026-10-11）

读取上轮六场景五种实现、每场景100帧×三轮的90次完整运行，得到9000条逐帧数据。
此次没有启动仿真，没有修改native源码或原始结果。原始日志SHA256复核通过。

| 当前Graph，第100物理帧 | 时间中位数(ms) | N | P | 三轮时间范围(ms) |
|---|---:|---:|---:|---:|
{table}

这六行是当前执行组合、legacy累计TOI .01，dt=.01、PCG rho=1e-4；没有启用残差门控。
各列分别取三轮中位数，不代表某次实际运行；实际三轮值见frame100.csv。

- per_frame.csv：9000条，每次每帧的同步host求解包络、五项阶段时间、N/P、退出原因、线搜索次数、接触数量及Graph统计。
- frame100.csv：90条实际第100帧结果；筛选method=ours得到当前Graph的18条。
- per_newton.csv：每个Newton记录；保留完整native_newton_json，包括原始PCG、alpha/beta、kappa、线搜索及可用诊断。
- current_graph_frame100_medians.csv：上表的六行摘要。
- manifest.json：原始日志、输入位置和输出文件校验摘要。

method含义：stiff=冻结原版；old_host/old=历史组合Host/Graph；host/ours=当前组合Host/Graph。
这些主矩阵方法全用legacy累计规则，不能替代同一检查点的三种外层规则对照。
已找到的八份完整检查点都是Table第28步或Bunny第46步，没有第100步的完整检查点。
只导出state_0099.bin的位置不能恢复第100步所需的完整物理上下文与缓存。

N_pcg_directions只数实际带PCG的方向；P_pcg_iterations是该帧总PCG迭代数。
newton_record_count还包含可能只组装、没有PCG的末次退出记录；accepted_updates单独保留。
pcg_stage_us_per_iteration是线性求解阶段/迭代数的均摊值，包含准备、MAS、归约、同步和Graph等，不能当作纯PCG单次kernel时间。
terminal_assembly_ms是assembly的子项，不能再次相加；graph_capture_instantiate_host_ms亦已包含在相关求解包络中。
unclassified_envelope_ms是host包络减CUDA阶段时间之和的差值，不能解释为已分离的CPU纯计算时间。
现有日志没有逐Newton的分项时间、完整PCG残差历史或逐帧写盘时间；缺失字段保留为空，没有补零伪造。

trajectory_*字段仅对当前Graph计算：同轮、同帧、全体顶点相对冻结Stiff轨迹的RMS/最大欧氏位置差，米转毫米。
这不是相对精确解或较严数值参考的误差，也不是布料分区指标。后续同检查点精度表必须另跑较严参考。
实际位置/速度仍以原trace中的float64二进制为准；碰撞候选数或候选距离不等于完整无穿透认证。

当前GIPC_DEFER_STATS=1把JSON写盘移到求解计时之外，但仍每帧重写累计stats.json。
位置/速度导出也在solver_ms之外。它们增加整次进程实际时间并可能间接影响GPU状态，开销未做开/关配对定量测试。
求解器内现有计时、JSON统计和残差门控计算仍包含在实际执行中。细粒度探针应另行诊断，不能混入正式加速比。
"""
    (output / "README.md").write_text(text, encoding="utf8")
    print(json.dumps({k: manifest[k] for k in ("complete", "simulation_runs_launched", "frames", "frame100_rows", "newton_records", "trajectory_difference_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=ROOT / "runs/local_combination_seven_20261011")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/historical_frame_metrics_20261011")
    args = parser.parse_args()
    export(args.session.resolve(), args.output.resolve())
