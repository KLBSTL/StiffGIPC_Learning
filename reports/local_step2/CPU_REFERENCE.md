# 独立 CPU 固定系统参考（2026-10-06）

范围：每系统只核对保存的 4 条 repeat=1 解；repeat=2 没有解文件，明确未覆盖。未运行 GPU，未改生产 rho、材料或停止规则。

导出 ABI 为 little-endian int32 block row/col、逐块 column-major float64 3×3、float64 RHS/解；loader 核对实际字节数和 meta 中 FNV-1a64，并记录 SHA-256。`linear_system/linear_system/global_linear_system.cu:160–187` 导出最终全局矩阵和整个 RHS；同文件 `:412–426` 的生产 SpMV 只使用这些矩阵块。`linear_system/utils/spmv.cu:55–64` 定义非对角转置扩展；`linear_system/solver/pcg_fixed_study.inl:146–151` 只保存每臂首次解。没有使用额外 ABD 预条件逆块来伪造 A。

矩阵取完整 global_triplet 与 m_b；每个非对角 3×3 块按生产 SpMV 增加其转置，对角块原样保留。ABD 已包含在该全局算子和 RHS 内，无额外旁路作用。只允许消除严格零行且零列、零 RHS 的孤立自由度；实际三项是否消除见下表。

参考为 FP64 SuperLU，残差用独立 50 位 Decimal 标量点积和范数，最多两次有限迭代改进。Windows longdouble 不被当作更高精度。参考残差通过不等于前向误差已获条件数认证，也不等于 PCG 默认解达到同精度。

| 系统 | 状态 | 完整 DOF / 零行 | CPU 参考真相对残差 | 参考 ≤1e-8 |
|---|---|---:|---:|---|
| fixed_cloth_legacy | completed | 17340 / 0 | 2.876262056e-14 | True |
| fixed_cloth_cholesky | completed | 17340 / 0 | 2.925166331e-14 | True |
| fixed_mixed_legacy | completed | 60858 / 0 | 5.687667006e-16 | True |

| 系统 | rho | 模式 | PCG 次数 | 独立 CPU 真残差 | 原 GPU 真残差 | 相对 CPU 参考解误差 | ≤1e-8 |
|---|---:|---|---:|---:|---:|---:|---|
| fixed_cloth_legacy | 1e-04 | host | 6 | 4.198429558e-02 | 4.198429558e-02 | 6.967366527e-03 | False |
| fixed_cloth_legacy | 1e-04 | graph | 6 | 4.198429712e-02 | 4.198429712e-02 | 6.967366178e-03 | False |
| fixed_cloth_legacy | 1e-16 | host | 72 | 4.162395051e-07 | 4.162395051e-07 | 3.056111675e-08 | False |
| fixed_cloth_legacy | 1e-16 | graph | 72 | 3.481388428e-07 | 3.481388428e-07 | 3.058051892e-08 | False |
| fixed_cloth_cholesky | 1e-04 | host | 6 | 4.198429446e-02 | 4.198429446e-02 | 6.967365833e-03 | False |
| fixed_cloth_cholesky | 1e-04 | graph | 6 | 4.198429446e-02 | 4.198429446e-02 | 6.967365833e-03 | False |
| fixed_cloth_cholesky | 1e-16 | host | 70 | 2.738346179e-07 | 2.738346179e-07 | 3.096566519e-08 | False |
| fixed_cloth_cholesky | 1e-16 | graph | 70 | 2.733603144e-07 | 2.733603144e-07 | 3.096537754e-08 | False |
| fixed_mixed_legacy | 1e-04 | host | 6 | 1.439583564e-03 | 1.439583564e-03 | 2.342390282e-02 | False |
| fixed_mixed_legacy | 1e-04 | graph | 6 | 1.439583120e-03 | 1.439583120e-03 | 2.342390849e-02 | False |
| fixed_mixed_legacy | 1e-16 | host | 130 | 1.086152489e-09 | 1.086152493e-09 | 3.736620114e-08 | True |
| fixed_mixed_legacy | 1e-16 | graph | 130 | 1.080405240e-09 | 1.080405242e-09 | 3.731057683e-08 | True |

每项硬上限 120 秒、4 GiB 工作进程 RSS；资源中止、奇异分解或非有限结果均不记通过。精确输入/解文件哈希、源码与程序身份、矩阵微小非对称、资源采样见同名 JSON。没有物理质量或性能认证，repeat=2 的缺口不会由 repeat=1 外推补齐。

本次实际资源：cloth legacy 7.03 秒/252.82 MiB，cloth Cholesky 6.09 秒/253.00 MiB，mixed legacy 25.83 秒/1077.33 MiB；时间仅用于 CPU 预算记录。三项首次 LU 解已满足参考残差，无需改进迭代。两个布料保存的严格 PCG 解仍未到 `1e-8` 真残差，独立 CPU 结果与原 GPU 残差一致；不能将 CPU 参考解通过改写为生产 PCG 全通过。

验证命令：`E:/Anaconda/envs/DL/python.exe -X utf8 -m unittest discover -s tools/local -p cpu_fixed_reference_test.py -v`，8 项通过；随后执行 `E:/Anaconda/envs/DL/python.exe -X utf8 tools/local/cpu_fixed_reference.py`，三个 CPU worker 均 completed。原始证据禁止覆盖，工具再次运行需要新的输出路径。
