# v41：double MAS 逆集成、PCG 失效诊断与完整场景探针

本轮完成了默认关闭的 double MAS 逆计算/存储，以及 host/Graph 的 PCG 数值失效守卫。悬挂布料100帧完成，但兔子 Graph/host 分别完成44/26帧后出现负rho。**完整轨迹修复失败，精度与性能均未验收，不改变默认逆精度。** 新证据将剩余问题定位到病态局部输入的非对称误差、原无主元求逆与随后对称存储之间的相互作用；仅把逆存为double不足以保持正定性。

## 1. 实现及冻结身份

- 独立源码 `sources/stiff_perf_v41`，从v39复制，未覆盖v37/v39或官方base。
- `GIPC_MAS_INVERSE64=1` 自动启用wide R/Z、局部作用与输出；默认0。新逆缓冲区参与分配/释放、Graph身份和完整失败快照。全double模式不导出未初始化的float逆。
- 求逆沿用v40固定重放的typed原生核；未加入对角正则、预算扩展或物理参数修改。
- host/Graph共同检查负/非有限rho、零rho但非零残差、非正/非有限曲率、非有限alpha/beta。精确零残差正常结束；保留旧rho判停顺序。守卫始终启用，因此关闭double时也会明确报告数值失效，而非继续到预算上限。
- 失效写入 `breakdown`、rho和迭代数，标记 `iteration_limit=false`，保存失败A/b与MAS内部数据，再抛出异常。失败解不作为接受更新。
- Graph初始化增加一次小标量读回；double逆有额外内存/计算成本，本轮未测正式收益。
- AutoDL RTX4090，CUDA12.8，Release，SM89；远端 `/root/autodl-tmp/stiff_toi_cudagraph_20260929/v41_20261003`。
- 562文件摘要：`8e1023549c9ee336e83805b80021c60e6d89cc29006c5ebdd8a301ffb5a7f26d`。
- 可执行文件18121128字节，SHA256：`141cfac9e8b8b3b8627edc32c0033b97025d4fb1226f17c0e7b9d84230caaefa`。

## 2. 构建、边界与实际类固定快照

`bash tools/autodl_v41_build.sh` 成功；`ctest --test-dir builds/autodl-stiff_perf_v41 --output-on-failure`：2/2通过。原融合夹具28组通过；新守卫夹具覆盖15个标量边界（包括零系统和上一轮已判停但下一rho异常），测试共享host分类逻辑与实际Graph标量kernel，并非15次完整PCG求解。组件检查6+7+7+17项全部通过。

实际MAS对象依次执行native/wide/full64/native，10个探针、每个3次Graph重放；检查精度切换、关闭恢复、R/Z/逆增长的Graph身份。三个快照均完成数据导出与增长检查。

| 固定快照 | full64 Graph/direct最大相对差 | CPU按实际GPU逆重建作用量最大相对差 | double逆非正块数 | 1e-10 Graph门槛 |
|---|---:|---:|---:|---|
| v37 smoke | 9.00e-17 | 3.07e-16 | 0 | 通过 |
| v39默认失败 | 4.14e-13 | 5.89e-12 | 0 | 通过 |
| v39严格失败 | 1.0223223e-10 | 5.63e-11 | 0 | **失败** |

strict的失败来自rhs探针第三次重放；没有提高阈值或重复到通过。夹具返回1，原串行命令因此未自动启动smoke。完成CPU调查后，单独启动后续有界场景诊断；这些结果不抵消夹具失败。

数据：`MAS_FIXTURE_V41_CPU.json`，`downloads/fixtures_v41_20261003/runs/fixtures/*/fixture.json`。

## 3. 八次有界场景运行

共同设置：dt=.01，Newton tol=.01，rho tol=1e-4，TOI robust velocity tol=.05，suite=1，原0.3N PCG预算，240秒超时，磁盘保留768MiB。全部启用PCG审计、失败系统导出与quality-only；耗时仅为诊断。完整命令在下载结果的 `reports/V41_*.json` 中。

| 运行 | 完成帧 | PCG求解数（含失败） | 单次最大迭代 | 结果 |
|---|---:|---:|---:|---|
| native smoke / Graph MAS | 2/2 | 4 | 24 | 完成 |
| double smoke / Graph MAS | 2/2 | 4 | 24 | 完成 |
| double smoke / host MAS | 2/2 | 4 | 24 | 完成 |
| diag控制，inverse64标志忽略 | 2/2 | 4 | 13 | 完成 |
| double悬挂 / Graph | 100/100 | 405 | 163 | 完成 |
| float逆+wide守卫 / 兔子Graph | 40/100 | 471 | 1233 | 第41帧、第8轮负rho=-1.2185224 |
| double兔子 / Graph | 44/100 | 516 | 738 | 第45帧、第5轮负rho=-2199.81349 |
| double兔子 / host | 26/100 | 334 | 495 | 第27帧、第2轮负rho=-.22369978 |

八次均零预算上限；三次数值失效均保存。**零上限来自提前识别失效，不等于PCG或模拟成功。** 轨迹不同，不能把完成帧数变化解释为同一系统的稳定改善。host也失败，排除了“只有Graph才发生”的解释。按计划停止扩展repeat和strict整场景矩阵。

成功返回PCG的真相对残差最大值：悬挂.319877、兔子Graph .074841、host .201047；原rho判停不等于真残差门槛，不能宣称高精度收敛。失败迭代的CPU真残差分别为12.3296和1.22966。

## 4. 新失败的CPU复核

对实际存储逆、层级映射与残差进行独立CPU重建。所有审计算子调用前后解向量逐位不变。

| 项目 | double Graph失败块1444 | double host失败块679 |
|---|---:|---:|
| 输入对称部分最小特征值 | 1.0 | 1.26506 |
| 输入条件数 | 2.5966e9 | 1.3153e10 |
| CPU Cholesky | 成功 | 成功 |
| 输入相对非对称性 | 7.39e-15 | 7.57e-17 |
| GPU double逆最小特征值 | -2.9210e-9 | -1.7992e-11 |
| GPU逆乘积误差 `||HB-I||F/sqrt(48)` | 1.79873 | .400983 |
| CPU对称输入LAPACK逆乘积误差 | 3.60e-9 | 2.11e-8 |
| CPU对称输入逆最小特征值 | 3.8512e-10 | 6.0100e-11 |

所有输入块的对称部分均未检出非正特征值（与kernel一致，把零对角填为1）。Graph只有块1444的存储逆为负，host只有块679为负。两次初始rho为正；失败时true-residual探针的CPU MAS二次型分别为约-2199.8134899和-.22369975，直接复现GPU失效符号。Graph坏块单项贡献-17006.786，压过其余正项。

进一步CPU消融使用相同48×48输入：

- 原无主元Gauss–Jordan步骤并镜像上三角：Graph块逆最小特征值-2.9165e-9，乘积误差1.79628；与实际GPU逆相对差2.34e-10。支持局部数值过程即可重现问题。
- 对**原始略非对称输入**使用LAPACK逆，再按原存储方式镜像上三角，Graph/host仍得到负特征值，乘积误差1.78828/.666365。因此不能只归咎于GPU或无主元消元；输入非对称性与强制对称存储也是关键因素。
- 先取输入对称部分再做原无主元步骤，Graph/host乘积误差降为9.65e-6/.130898，仍显著大于CPU对照。单纯对称化未解决全部稳定性问题。
- 新Graph失败算子的CPU/GPU相对差最大6.28e-10，GPU自身重复差5.17e-10，也没有通过严格1e-10固定算子门槛。

这些实验指向“对称输入 + 保持正定性的分解/作用方式”，尚未证明新的GPU实现或完整轨迹通过。CPU求逆是诊断参考，不是本轮模拟器中的回退算法。

证据：`MAS_V41_CPU.json`、`MAS_V41_NEGATIVE_BLOCKS.json`。CPU分析脚本 `tools/audit_negative_blocks_v41.py` 明确分开raw输入、对称输入、原步骤和LAPACK参考。

## 5. 接受路径安全、身份与清理

- 悬挂100帧：202个接受推进+99个静止桥接=301段。
- 兔子Graph：44个完整帧及第45帧部分记录，269个接受推进+44个桥接=313段。
- 共614段CPU BVH+Tight-Inclusion检查，CCD容差1e-9，零保守碰撞标记；每帧子步计数、初始状态及已完成帧末端逐位核对通过。
- 兔子CCD范围为导出表面顶点；`--stable-nh1`允许FEM翻转。兔子记录6060个累计FEM翻转计数、0个ABD翻转；部分路径零碰撞不能替代100帧完成性或完整物理质量。
- host/float守卫运行未导出子步，本轮没有为它们声称全接受路径CCD。
- 本地核验562源文件、8组参数/runner/二进制身份和下载包；远端同时重新核验v37/v39/v41源与二进制。
- 结果包737161043字节，SHA256 `f8091546ea2b0d4ddcc5244a099fd20675fa91dfb2453d056d92b954407d1360`；夹具包55128758字节，SHA256 `8b0e0ddeac5d7d913aadef35d81ece3925a1c598549cf483668fa19f0a4001cd`。
- 下载校验后仅删除远端这两份重复传输包，回收792289801字节（约755.6MiB）。保留本地包与远端解包原始数据；最终可用2796417024字节（约2.60GiB），GPU 0%、空闲24081MiB。

汇总证据：`VERIFICATION_V41_20261003.json`、`CCD_V41_double_hang.json`、`CCD_V41_double_bunny.json`、`FINAL_REMOTE_V41_20261003.json`。

## 6. 复现与下一步

远端构建/测试：

```sh
bash tools/autodl_v41_build.sh
ctest --test-dir builds/autodl-stiff_perf_v41 --output-on-failure
GIPC_VALIDATE_COMPONENTS=$PWD/reports/components.json builds/autodl-stiff_perf_v41/gipc
python3 tools/benchmark_perf_v41.py --phase fixtures
# fixtures实际返回1；保留结果后独立执行下面的诊断阶段。
python3 tools/benchmark_perf_v41.py --phase smoke
python3 tools/benchmark_perf_v41.py --phase hang
python3 tools/benchmark_perf_v41.py --phase guard
python3 tools/benchmark_perf_v41.py --phase bunny
python3 tools/benchmark_perf_v41.py --phase host
```

上述工具拒绝覆盖现有结果，复现应使用新独立目录。CPU精确命令（工作目录为任务根）：

```powershell
& 'E:\Anaconda\envs\DL\python.exe' tools/analyze_fixture_v41.py downloads/fixtures_v41_20261003/runs/fixtures --output reports/MAS_FIXTURE_V41_CPU.json
& 'E:\Anaconda\envs\DL\python.exe' tools/analyze_mas_v41.py downloads/autodl_perf_v41_20261003 --output reports/MAS_V41_CPU.json
& 'E:\Anaconda\envs\DL\python.exe' tools/audit_negative_blocks_v41.py downloads/autodl_perf_v41_20261003 --output reports/MAS_V41_NEGATIVE_BLOCKS.json
& 'builds/validator/Release/validate_path.exe' downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_double_hang/trace reports/CCD_V41_double_hang.json substeps --stable-nh1
& 'builds/validator/Release/diagnose_first_path.exe' downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_double_bunny/trace reports/CCD_V41_double_bunny.json substeps --stable-nh1
& 'E:\Anaconda\envs\DL\python.exe' tools/verify_v41.py
```

后续顺序：保留v41冻结结果；在新版本对五个失败快照（v39两个、v41三个）比较显式输入对称化、double Cholesky分解/三角求解、必要的尺度处理，检查正定性、作用量与固定PCG真残差；通过后再做实际类Graph生命周期和完整兔子host/Graph。优先保持正定的因子作用，避免再次只扩大显式逆精度。局部分解失败需明确诊断，不能静默改变A、物理参数、预算或安全策略。Graph原子累加的严格重复性仍需独立处理，不能通过放宽既定门槛消除失败。
