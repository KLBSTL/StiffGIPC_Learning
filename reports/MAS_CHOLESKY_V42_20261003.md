# v42：固定失败系统的 Cholesky 与确定顺序聚合

五个冻结失败系统已完成70次GPU固定求解及独立CPU核验。**对称输入＋double Cholesky三角求解＋固定顺序R聚合**通过固定算子门槛；在单独的rho=1e-16配置下，五系统的真残差和重复解门槛全部通过。默认停止参数未改变，完整仿真仍由后续v43验证。

## 方案、范围与身份

- 独立源码 `sources/mas_replay_v42`，实际测试目录AutoDL `v42a_20261003`。沿用v40原生kernel和v37 SpMV/ABD；没有修改A/b、原0.3N预算（18257次）或物理参数。
- mode5：原double显式逆、原子聚合，作为对照。
- mode6：局部输入取对称部分，double Cholesky因子作用，原子聚合。零填充对角仍按原kernel补1；非正/非有限pivot明确失败，不加正则。
- mode7：mode6加固定顺序CSR聚合；每个标量按原FEM节点排序串行求和，避免R的原子累加。因子作用为两次三角求解，不构造显式逆。
- 五快照：v39默认/严格失败，v41 double Graph、double host和float wide守卫失败。
- 主实验：5快照×3方案×2容差（1e-4/1e-14）×2重复=60次；每调用240秒上限。
- 主实验发现mode7两个系统的1e-14真残差超门槛后，追加5快照统一mode7、rho=1e-16、相同预算、2重复=10次；原失败项保留。
- GPU为RTX4090，CUDA12.8，SM89，Release。二进制SHA256：`0531ec7ae2c3ba3ef9c97201f5e2c9e9adb28232f845be7fc995a07dc0acfb16`。

首次v42构建及最小夹具成功，但首个快照捕获遇到同步cudaMemset不兼容错误。该源码、manifest、二进制及错误日志保留于远端v42和本地 `downloads/replay_v42_initial_20261003`。修复仅把清零改为同一per-thread stream的cudaMemsetAsync；重新冻结v42a并重建，数值算法不变。初始二进制SHA256为 `813ead164dadb5f1e85835ee7d926d0567c061512f4fcb914cc10cf1b0c9150f`。

## 固定算子结果

CTest 1/1通过：正定/非对称小矩阵、零填充对角、负pivot和NaN pivot分类、三角求解残差与Graph重放。最大绝对残差5.20e-18，Graph逐位相同。

每个快照10探针，各3次直接重复及3次Graph重复分别记录。mode7在五快照全部逐位一致；mode6仍有两个快照未通过1e-10重复门槛，证明仅改Cholesky没有消除原子聚合误差。

| mode7快照 | CPU按实际GPU因子重建作用量差 | 因子重建H最大相对差 | rho=1e-14最大CPU真残差 | 1e-8真残差门槛 |
|---|---:|---:|---:|---|
| v39默认 | 7.19e-13 | 5.65e-16 | 5.49e-9 | 通过 |
| v39严格 | 7.76e-12 | 5.69e-16 | 6.85e-9 | 通过 |
| v41 Graph | 3.04e-11 | 5.21e-16 | 2.35e-8 | 失败 |
| v41 host | 5.44e-13 | 7.68e-16 | 3.65e-9 | 通过 |
| v41 wide | 3.88e-12 | 6.02e-16 | 4.94e-8 | 失败 |

所有Cholesky因子对角为正，所有测试探针二次型为正，40次主实验Cholesky求解均按rho判停、零breakdown和预算上限。原double逆对照在v41 Graph/host各四次breakdown，其他12次rho判停；原double逆重复性也有两项失败。

mode6在v41 Graph的CPU作用量差1.25e-9、GPU重复差1.35e-9；固定顺序mode7将后者降为逐位一致。这里固定的是预条件器作用，不是整个PCG：全局SpMV仍有原子累加，因此不能声称完整求解确定。

## 更严格停止参数的独立补充

| 快照 | rho=1e-16最大CPU真残差 | 两次解相对差 | 迭代次数 |
|---|---:|---:|---|
| v39默认 | 4.41e-10 | 4.03e-9 | 6164 / 6156 |
| v39严格 | 5.85e-10 | 3.56e-8 | 4720 / 4728 |
| v41 Graph | 3.56e-9 | 4.00e-9 | 4155 / 4158 |
| v41 host | 7.08e-10 | 1.01e-8 | 5764 / 5758 |
| v41 wide | 4.25e-9 | 3.45e-8 | 4751 / 4746 |

10/10均在18257原预算内rho判停，真残差1e-8和重复解1e-6门槛通过。补充试验的因子与全部探针输出均与主实验mode7逐位相同。不能用这组结果把1e-14的失败改写为通过，也不能把更严格的额外工作量作为性能收益。

70次总计62次rho判停、8次原double逆breakdown、零预算上限。CPU使用独立对称稀疏A计算真残差，并用实际GPU导出的L重建H和预条件器作用，未只信任GPU自报值。

## 可复现证据

- 冻结及身份：`manifests/mas_replay_v42a.json`、`reports/VERIFICATION_V42_20261003.json`。119源码/依赖文件核验，初始119文件另行核验，原生kernel与v40字节一致。
- 主CPU分析：`reports/REPLAY_V42_CPU.json`；补充：`reports/REPLAY_V42_STRICT_CPU.json`。
- 主下载：`downloads/replay_v42_20261003`；严格：`downloads/replay_v42_strict_20261003`；初始失败：`downloads/replay_v42_initial_20261003`。
- 三包SHA256依次为 `3142dfb407466ae51a2f5f1dd09d31b46f10064c9d70d41bf4002862024e0a74`、`e29ef869437edb8e4e07a9f53acd15550be7cba4df0823751aa6dedac70ceefd`、`f935d99c889144a296b9b1900466ecdfaf7445703279c6b4ae7bf5e42fd9ab83`。
- 下载核验后移除三份远端重复传输包，回收263341786字节。随后对10份经SHA核验与对应原件完全相同的因子文件做硬链接去重，原路径和内容不变；原始数据和本地包保留。清理记录在v42a远端 `reports/final_verification.json` 和 `reports/factor_dedup.json`，最终空闲以v43结束记录为准。

实际执行（AutoDL新目录中）：

```sh
cmake -S sources/mas_replay_v42 -B builds/replay -DCMAKE_BUILD_TYPE=Release
cmake --build builds/replay --parallel 2
ctest --test-dir builds/replay --output-on-failure
python3 tools/benchmark_replay_v42.py
python3 tools/strict_replay_v42.py
```

本地任务根使用 `E:\Anaconda\envs\DL\python.exe` 依次执行 `tools/analyze_replay_v42.py`、`tools/analyze_strict_v42.py`、`tools/verify_v42.py`，均返回0。工具拒绝覆盖已有报告；重现使用新目录。

后续实际类v43采用mode7候选，默认关闭；验证组装时因子/聚合表更新、Graph身份、缓冲增长、完整场景和独立路径安全。固定快照通过不是全仿真修复结论。
