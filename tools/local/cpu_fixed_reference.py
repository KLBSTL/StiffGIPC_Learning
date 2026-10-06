"""CPU-only reference for the four saved repeat-1 solutions of a frozen system.

No materials, GPU code, PCG thresholds or frozen experiment files are changed.
SuperLU is float64; independent residuals use Decimal precision 50 because
Windows numpy.longdouble need not be wider than float64. No condition-number or
arbitrary-precision forward-solution certificate is inferred from a residual.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


CASES = ("fixed_cloth_legacy", "fixed_cloth_cholesky", "fixed_mixed_legacy")
REFERENCE_RESIDUAL_LIMIT = 1e-8


def record(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def fnv1a64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def expand_blocks(rows, cols, blocks, dofs):
    import numpy as np
    from scipy.sparse import coo_matrix
    if dofs < 0 or dofs % 3 or len(rows) != len(cols) or blocks.shape != (len(rows), 3, 3):
        raise ValueError("Invalid block dimensions")
    if len(rows) and (min(rows.min(), cols.min()) < 0 or max(rows.max(), cols.max()) >= dofs // 3):
        raise ValueError("Block index outside full global DOFs")
    if not np.isfinite(blocks).all():
        raise ValueError("Nonfinite matrix")
    rr = np.broadcast_to(3 * rows[:, None, None] + np.arange(3)[None, :, None], blocks.shape).ravel()
    cc = np.broadcast_to(3 * cols[:, None, None] + np.arange(3)[None, None, :], blocks.shape).ravel()
    vv = blocks.ravel()
    off = np.repeat(rows != cols, 9)
    # Match production SpMV for every stored block, including lower entries.
    # Diagonal blocks occur once, exactly as stored: do not symmetrize them.
    matrix = coo_matrix((np.r_[vv, vv[off]], (np.r_[rr, cc[off]], np.r_[cc, rr[off]])),
                        shape=(dofs, dofs)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def active_dofs(matrix, rhs):
    import numpy as np
    zero_rows = np.flatnonzero(np.diff(matrix.indptr) == 0)
    zero_cols = np.flatnonzero(np.diff(matrix.tocsc().indptr) == 0)
    if not np.array_equal(zero_rows, zero_cols):
        raise ValueError("Zero row/column mismatch: cannot eliminate DOFs")
    if np.any(rhs[zero_rows] != 0):
        raise ValueError("Nonzero RHS on exact zero row")
    active = np.ones(len(rhs), dtype=bool)
    active[zero_rows] = False
    return active, zero_rows


def decimal_residual(matrix, rhs, solution):
    """Independent scalar evaluation; no scipy/GPU matvec used for residual."""
    import numpy as np
    if not np.isfinite(solution).all():
        raise ValueError("Nonfinite solution")
    with localcontext() as ctx:
        ctx.prec = 50
        x = [Decimal.from_float(float(v)) for v in solution]
        residual = []
        squared_r = Decimal(0)
        squared_b = Decimal(0)
        for i in range(len(rhs)):
            dot = Decimal(0)
            for j in range(matrix.indptr[i], matrix.indptr[i + 1]):
                dot += Decimal.from_float(float(matrix.data[j])) * x[matrix.indices[j]]
            b = Decimal.from_float(float(rhs[i]))
            r = b - dot
            squared_r += r * r
            squared_b += b * b
            residual.append(float(r))
        norm_r = squared_r.sqrt()
        relative = float((squared_r / squared_b).sqrt()) if squared_b else float(norm_r)
    if not math.isfinite(relative):
        raise ValueError("Nonfinite residual metric")
    return np.asarray(residual), {"relative_l2": relative, "absolute_l2": float(norm_r)}


def load_snapshot(prefix):
    import numpy as np
    prefix = Path(prefix)
    meta_path = Path(str(prefix) + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n, count = meta["dofs"], meta["blocks"]
    if meta.get("matrix_format") != "symmetric upper block COO, int32 indices, float64 column-major 3x3":
        raise ValueError("Unrecognized snapshot ABI")
    arrays = {}
    evidence = {"meta": record(meta_path)}
    for key, dtype, length in (("rows", "<i4", count), ("cols", "<i4", count),
                               ("values", "<f8", 9 * count), ("rhs", "<f8", n)):
        path = Path(str(prefix) + "_" + key + ".bin")
        data = path.read_bytes()
        expected = meta["buffers"][key]
        if len(data) != expected["bytes"] or fnv1a64(data) != expected["fnv1a64"]:
            raise ValueError("Snapshot fingerprint mismatch: " + key)
        arr = np.frombuffer(data, dtype=dtype)
        if len(arr) != length or not np.isfinite(arr).all():
            raise ValueError("Invalid snapshot buffer: " + key)
        arrays[key] = arr
        evidence[key] = record(path)
    blocks = arrays["values"].reshape(count, 3, 3).transpose(0, 2, 1)
    matrix = expand_blocks(arrays["rows"], arrays["cols"], blocks, n)
    # Unique global blocks ensure CSR duplicate summation has not rounded a
    # second stored contribution into an entry before Decimal evaluation.
    identities = np.column_stack((arrays["rows"], arrays["cols"]))
    if len(np.unique(identities, axis=0)) != count:
        raise ValueError("Snapshot is not unique block COO")
    if np.any(arrays["rows"] > arrays["cols"]):
        raise ValueError("Snapshot metadata claims upper storage but contains lower blocks")
    active, zero = active_dofs(matrix, arrays["rhs"])
    asymmetry = matrix - matrix.T
    details = {"dofs": n, "blocks": count, "scalar_nnz": matrix.nnz,
               "lower_blocks": int(np.sum(arrays["rows"] > arrays["cols"])),
               "exact_zero_rows": zero.tolist(), "active_dofs": int(active.sum()),
               "diagonal_blocks_not_symmetrized": True,
               "max_absolute_asymmetry": float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.0,
               "preconditioners": [{k: p[k] for k in ("kind", "offset", "count") if k in p}
                                   for p in meta["local_preconditioners"]]}
    return matrix, arrays["rhs"], active, zero, evidence, details


def solve_case(directory):
    import numpy as np
    import scipy
    from scipy.sparse.linalg import splu
    started = time.monotonic()
    directory = Path(directory)
    prefix = directory / "fixed" / "f2_n1"
    matrix, rhs, active, zero, evidence, details = load_snapshot(prefix)
    study_path = Path(str(prefix) + "_study.json")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if study.get("system_unchanged") is not True or study.get("primary_restored_bitwise") is not True:
        raise ValueError("Unprotected study")
    evidence["study"] = record(study_path)
    requested = json.loads((directory / "requested.json").read_text(encoding="utf-8"))
    output = {"case": directory.name, "status": "completed", "matrix": details,
              "source_digest": requested["source_digest"], "exe_sha256": requested["exe_sha256"],
              "inputs": evidence, "repeat2_covered": False,
              "coverage": "Only saved repeat=1: rho 1e-4/1e-16 x host/graph; repeat=2 solutions were not exported",
              "reference_method": "Float64 SuperLU with at most two Decimal50-residual iterative refinements",
              "residual_method": "Independent Decimal precision=50 scalar CSR dot and norms; no GPU/scipy matvec",
              "numpy": np.__version__, "scipy": scipy.__version__,
              "numpy_longdouble_precision_bits": np.finfo(np.longdouble).nmant + 1,
              "condition_number_certified": False, "physics_certified": False, "performance_certified": False}
    if not active.any():
        reference = np.zeros(len(rhs))
        factor = None
    else:
        factor = splu(matrix[active][:, active].tocsc())
        reference = np.zeros(len(rhs))
        reference[active] = factor.solve(rhs[active])
    residual, metrics = decimal_residual(matrix, rhs, reference)
    history = [metrics]
    # Fixed bounded refinement, not a parameter search or changed PCG tolerance.
    for _ in range(2):
        if metrics["relative_l2"] <= 1e-12 or factor is None:
            break
        candidate = reference.copy()
        candidate[active] += factor.solve(residual[active])
        next_residual, next_metrics = decimal_residual(matrix, rhs, candidate)
        history.append(next_metrics)
        if next_metrics["relative_l2"] >= metrics["relative_l2"]:
            break
        reference, residual, metrics = candidate, next_residual, next_metrics
    output["reference"] = {**metrics, "refinement_evaluations": history,
                           "residual_limit": REFERENCE_RESIDUAL_LIMIT,
                           "residual_passed": metrics["relative_l2"] <= REFERENCE_RESIDUAL_LIMIT,
                           "lu_nnz": int(factor.L.nnz + factor.U.nnz) if factor is not None else 0}
    records = []
    norm_ref = float(np.linalg.norm(reference[active]))
    for phase, tol in ((1, 1e-4), (3, 1e-16)):
        for mode_id, mode in enumerate(("host", "graph")):
            path = Path(str(prefix) + f"_p{phase}_m{mode_id}_x.bin")
            if path.stat().st_size != len(rhs) * 8:
                raise ValueError("Saved solution size mismatch")
            x = np.fromfile(path, dtype="<f8")
            native = [r for r in study["runs"] if r["mode"] == mode and r["repeat"] == 1 and r["rho_tolerance"] == tol]
            if len(native) != 1 or native[0]["error"] or native[0]["limit"]:
                raise ValueError("Saved solution has no unique clean study record")
            _, actual = decimal_residual(matrix, rhs, x)
            delta = float(np.linalg.norm((x - reference)[active]))
            records.append({"mode": mode, "repeat": 1, "rho_tolerance": tol,
                            "iterations": native[0]["iterations"], "solution": record(path),
                            "cpu_true_residual": actual, "native_true_relative_residual": native[0]["true_relative_residual"],
                            "relative_error_to_cpu_reference": delta / norm_ref if norm_ref else delta,
                            "max_abs_error_to_cpu_reference": float(np.max(np.abs(x - reference))) if len(x) else 0.0,
                            "exact_zero_row_solution_max_abs": float(np.max(np.abs(x[zero]))) if len(zero) else 0.0,
                            "residual_at_most_1e_minus_8": actual["relative_l2"] <= REFERENCE_RESIDUAL_LIMIT})
    output["saved_solutions"] = records
    output["wall_seconds_diagnostic_only"] = time.monotonic() - started
    return output


def bounded_case(directory, seconds, memory_mib):
    import psutil
    started = time.monotonic()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    peak = 0
    with tempfile.TemporaryDirectory(prefix="cpu_reference_") as tmp:
        out = Path(tmp) / "worker.json"
        with (Path(tmp) / "log.txt").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                                        "--worker", str(directory.resolve()), "--worker-output", str(out)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            guard = None
            while process.poll() is None:
                try:
                    owned = psutil.Process(process.pid)
                    rss = owned.memory_info().rss + sum(c.memory_info().rss for c in owned.children(recursive=True))
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - started > seconds:
                    guard = "timeout"
                elif peak > memory_mib * 2**20:
                    guard = "memory_budget"
                if guard:
                    process.kill()
                    process.wait()
                    break
                time.sleep(.1)
        if guard:
            result = {"case": directory.name, "status": guard, "saved_solutions": [],
                      "coverage": "Worker stopped before completed CPU evidence; no pass inferred"}
        elif out.exists():
            result = json.loads(out.read_text(encoding="utf-8"))
        else:
            result = {"case": directory.name, "status": "error", "error": (Path(tmp) / "log.txt").read_text(encoding="utf-8")[-2000:]}
    result["resource_guard"] = {"timeout_seconds": seconds, "memory_budget_mib": memory_mib,
                                "peak_sampled_rss_mib": peak / 2**20,
                                "elapsed_seconds": time.monotonic() - started}
    return result


def markdown(report):
    lines = ["# 独立 CPU 固定系统参考（2026-10-06）", "",
             "范围：每系统只核对保存的 4 条 repeat=1 解；repeat=2 没有解文件，明确未覆盖。未运行 GPU，未改生产 rho、材料或停止规则。", "",
             "矩阵取完整 global_triplet 与 m_b；每个非对角 3×3 块按生产 SpMV 增加其转置，对角块原样保留。ABD 已包含在该全局算子和 RHS 内，无额外旁路作用。只允许消除严格零行且零列、零 RHS 的孤立自由度；实际三项是否消除见下表。", "",
             "参考为 FP64 SuperLU，残差用独立 50 位 Decimal 标量点积和范数，最多两次有限迭代改进。Windows longdouble 不被当作更高精度。参考残差通过不等于前向误差已获条件数认证，也不等于 PCG 默认解达到同精度。", "",
             "| 系统 | 状态 | 完整 DOF / 零行 | CPU 参考真相对残差 | 参考 ≤1e-8 |", "|---|---|---:|---:|---|"]
    for c in report["cases"]:
        m, ref = c.get("matrix", {}), c.get("reference", {})
        res = ref.get("relative_l2")
        lines.append(f"| {c['case']} | {c['status']} | {m.get('dofs', '?')} / {len(m.get('exact_zero_rows', [])) if m else '?'} | {format(res, '.9e') if res is not None else '未获得'} | {ref.get('residual_passed', False)} |")
    lines += ["", "| 系统 | rho | 模式 | PCG 次数 | 独立 CPU 真残差 | 原 GPU 真残差 | 相对 CPU 参考解误差 | ≤1e-8 |",
              "|---|---:|---|---:|---:|---:|---:|---|"]
    for c in report["cases"]:
        for r in c.get("saved_solutions", []):
            lines.append(f"| {c['case']} | {r['rho_tolerance']:.0e} | {r['mode']} | {r['iterations']} | {r['cpu_true_residual']['relative_l2']:.9e} | {r['native_true_relative_residual']:.9e} | {r['relative_error_to_cpu_reference']:.9e} | {r['residual_at_most_1e_minus_8']} |")
    lines += ["", "每项硬上限 120 秒、4 GiB 工作进程 RSS；资源中止、奇异分解或非有限结果均不记通过。精确输入/解文件哈希、源码与程序身份、矩阵微小非对称、资源采样见同名 JSON。没有物理质量或性能认证，repeat=2 的缺口不会由 repeat=1 外推补齐。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--root", type=Path, default=Path("runs/local_step2_fixtures_20261006"))
    parser.add_argument("--output", type=Path, default=Path("reports/local_step2/CPU_REFERENCE.json"))
    args = parser.parse_args()
    if args.worker:
        try:
            result = solve_case(args.worker)
        except Exception as exc:
            result = {"case": args.worker.name, "status": "error", "error": type(exc).__name__ + ": " + str(exc)}
        args.worker_output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        return
    if args.output.exists() or args.output.with_suffix(".md").exists():
        raise FileExistsError("Preserve existing CPU reference evidence")
    report = {"schema": "cpu_fixed_reference.v1", "cpu_only": True, "repeat2_covered": False,
              "performance_certified": False, "physical_quality_certified": False,
              "tool": record(__file__), "cases": []}
    for name in CASES:
        result = bounded_case(args.root / name, 120, 4096)
        report["cases"].append(result)
        print(name + ": " + result["status"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
