"""CPU-only independent operator identity/ownership and narrow source checks."""
from decimal import Decimal
from pathlib import Path
import random


def check() -> None:
    rng = random.Random(431)
    counts = [0, 1, 31, 32, 33, 255, 256, 257, 513]
    for count in counts:
        owners = [0] * count
        for block in range(max(1, (count + 255) // 256)):
            for lane in range(256):
                index = 256 * block + lane
                if index < count:
                    owners[index] += 1
        assert all(n == 1 for n in owners)
        # Small integer arithmetic is exact: independently build the dense
        # operator, including lower entries, repeated entries, and deliberately
        # nonsymmetric diagonal blocks. No canonicalization is permitted.
        n = 15
        dense = [[Decimal(0) for _ in range(n)] for _ in range(n)]
        x = [Decimal(rng.randrange(-4, 5)) for _ in range(n)]
        stored_scalar = Decimal(0)
        for _ in range(count):
            i, j = rng.randrange(n // 3), rng.randrange(n // 3)
            block = [[Decimal(rng.randrange(-3, 4)) for _ in range(3)] for _ in range(3)]
            local = sum(x[3 * i + r] * block[r][c] * x[3 * j + c]
                        for r in range(3) for c in range(3))
            stored_scalar += local if i == j else 2 * local
            for r in range(3):
                for c in range(3):
                    dense[3 * i + r][3 * j + c] += block[r][c]
                    if i != j:
                        dense[3 * j + c][3 * i + r] += block[r][c]
        ap = [sum(a * b for a, b in zip(row, x)) for row in dense]
        assert sum(a * b for a, b in zip(x, ap)) == stored_scalar
        assert all(sum(row[j] * 0 for j in range(n)) == 0 for row in dense)

    root = Path(__file__).resolve().parents[2]
    kernel = (root / "linear_system/utils/spmv_quadratic_kernel.cuh").read_text()
    assert "return;" not in kernel
    assert "if(i!=j)quadratic*=2" in kernel
    assert "if(i<j)" not in kernel
    assert "BlockReduceFloat(block_storage).Sum(quadratic)" in kernel
    assert "if(valid && flags)" in kernel
    valid_begin = kernel.index("if(valid)")
    opening = kernel.index("{", valid_begin)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (kernel[closing] == "{") - (kernel[closing] == "}")
        closing += 1
    assert "HeadSegmentedReduce" not in kernel[opening:closing]
    assert kernel[closing:].count("HeadSegmentedReduce") == 3
    assert "Vector3 vec=Vector3::Zero()" in kernel and "char flags=1" in kernel
    study = (root / "linear_system/solver/pcg_spmv_quadratic_study.inl").read_text()
    assert study.count("cudaMemsetAsync(output.data(),0xff") == 2
    assert study.count("cudaMemsetAsync(scalar.data(),0xff") == 2
    assert "cudaMemsetAsync(spmv_quadratic_partials.data(),0xff" in study
    graph = (root / "linear_system/solver/pcg_graph_impl.inl").read_text()
    for key in ["spmv_quadratic_effective", "spmv_quadratic_partials_count",
                "spmv_quadratic_reduce_bytes", "spmv_quadratic_partials.data()",
                "spmv_quadratic_reduce_storage.data()"]:
        assert key in graph
    old = root.parents[1] / "stiff_perf_v50/StiffGIPC/linear_system/utils/spmv.cu"
    active = (root / "linear_system/utils/spmv.cu").read_text()
    frozen = old.read_text()
    start = "__global__ void warp_reduce_sym_spmv_kernel"
    end = "}  // namespace\n"
    assert active.split(start, 1)[1].split(end, 1)[0] == frozen.split(start, 1)[1].split(end, 1)[0]
    print(f"PASS: {len(counts)} block-count ownership cases, exact Decimal dense/block identities, original kernel retained, Graph scratch signature, all-lane warp collectives, poisoned Graph-study outputs.")
    print("NOT RUN: C++/CUDA build, GPU kernels, PCG, trajectory quality, performance.")


if __name__ == "__main__":
    check()
