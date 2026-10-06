# v43：Cholesky实际集成与兔子超时诊断

执行跨2026-10-03/04，目录和证据文件沿用阶段起始日期20261003。

已将v42通过固定系统验证的“对称Cholesky因子作用＋固定顺序聚合”默认关闭地集成进实际MAS类。**六个实际类快照、组件、小场景及悬挂100帧通过；兔子Graph/host完成34/37帧后均在240秒上限退出。完整轨迹与性能仍未验收。** 新日志显示CCD候选大幅增长，需要对未完成帧做状态保存和分项诊断，不能把本轮结果宣布为整体修复。

## 1. 实现、身份与测试入口

- 独立 `sources/stiff_perf_v43`；`GIPC_MAS_CHOLESKY=1` / runner `--mas-cholesky 1` 启用，默认0。开启时自动wide，优先于旧double显式逆路径；旧路径和v41冻结副本保留。
- 直接复用v42的Cholesky、三角求解与固定顺序CSR聚合kernel。对称化只用于局部预条件器输入，未修改全局A/b、物理参数、安全策略或PCG预算。
- 因子、状态及CSR缓冲区加入快照、Graph身份和释放。组装阶段检查pivot并根据当前层级映射重建CSR；apply内无分配或host读回。此版包含额外CPU准备和因子成本，没有性能优化结论。
- v41的负rho等PCG守卫继续启用。
- AutoDL RTX4090 / CUDA12.8 / Release / SM89。564源码文件摘要：`73fde2961a82e3609717cecff47b3b2495accf5018f4e0c8526a665388618bc8`。
- 可执行文件18229424字节，SHA256：`a29209fc525576a46aff3c33dc6a8a8ed4cd6a123bdcc71dcc9fe7958a7a8a41`。
- GPU源码和二进制从构建至运行保持冻结。首次夹具脚本误改历史v41路径，前三项通过、第四项查找失败；正确路径在独立 `fixtures_corrected` 完成余下三项。首次smoke在GPU启动前因启动器旧实现标签检查失败，修正后使用独立 `smoke_corrected` 报告。原脚本/日志/报告保留，未用这些启动错误替代数值结果。

## 2. 固定系统到实际类的核验

构建成功，CTest 2/2、原28组融合夹具、15项PCG守卫边界以及组件6+7+7+17项通过。

实际MAS对象在六个快照上执行native/cholesky/native切换，每个10探针、3次Graph重放；随后强制因子、R/Z和CSR缓冲增长。全部Cholesky重放逐位一致，模式身份变化、关闭恢复、增长身份及增长前后作用量通过。

- 六快照：v37 smoke、v39默认/严格失败、v41 double Graph/host失败及float wide失败。
- 独立CPU按实际导出的GPU因子与层级重建作用量，最大相对差 `3.0423e-11`，在 `1e-10` 门槛内。
- 因子重建对称输入H的最大相对差 `7.6823e-16`，全部因子对角为正。
- 五个失败快照的实际类输出与v42 mode7重放逐位相同。
- 三组仿真初始快照CPU作用量差最大 `1.53e-16`，GPU重复差0；审计前后解向量逐位不变。

这些结果支持集成保持了固定算子行为。v42的70次固定求解、rho=1e-14失败项和rho=1e-16补充通过项详见 `MAS_CHOLESKY_V42_20261003.md`，不能混同为整场景默认参数的准确性结论。

## 3. 七次实际场景运行

本轮所有实际场景保留默认rho=1e-4，dt=.01、Newton tol=.01、robust velocity tol=.05、suite=1、原0.3N PCG预算。每组240秒，保留768MiB磁盘，启用quality-only及审计；所有耗时只作诊断。未运行更严格整场景或重复矩阵。

| 运行 | 已完成帧 | 已落盘PCG数 | 最大已记录迭代 | 结果 |
|---|---:|---:|---:|---|
| native MAS / Graph控制 | 2/2 | 4 | 24 | 完成 |
| Cholesky / Graph控制 | 2/2 | 4 | 24 | 完成 |
| Cholesky / host控制 | 2/2 | 4 | 24 | 完成 |
| diag控制（Cholesky标志忽略） | 2/2 | 4 | 13 | 完成 |
| Cholesky悬挂 / Graph | 100/100 | 413 | 137 | 完成 |
| Cholesky兔子 / Graph | 34/100 | 205 | 530 | 241.03秒，SIGTERM超时 |
| Cholesky兔子 / host | 37/100 | 280 | 665 | 240.14秒，SIGTERM超时 |

已落盘PCG数据均无breakdown或预算上限，日志未出现Cholesky pivot/PCG负rho错误。**SIGTERM没有落盘第35/38帧的完整统计，也没有终端失败A/b快照；因此上述零计数只适用于已记录数据。** 这两次退出不能归为PCG上限，也不能认定完整PCG/非线性轨迹通过。

已完成帧的最大真相对残差：悬挂 `.319877`、兔子Graph `.0415904`、host `.0833005`。默认rho判停仍不等于真残差精度门槛。Graph/host轨迹已经不同，不能把34/37帧差异当成同系统的性能或稳定性对照。

## 4. 当前超时线索与证据边界

| 数据 | Graph | host |
|---|---:|---:|
| 完整落盘帧 | 34 | 37 |
| 已完成帧求解耗时之和 | 18.82秒 | 26.60秒 |
| 最后完整帧接受外迭代数 | 17 | 15 |
| 最后完整帧最小alpha | .07518 | .07413 |
| 最后完整帧最大safe CCD广相对数 | 438009 | 501255 |
| 运行日志最大检测CCD对数 | 43282830 | 21209022 |

Graph扩容日志依次记录1954320、4785983、13619148、43282830对；host记录1138168、2588358、6064080、11310898、21209022对。GPU增量峰值分别约3764/3086MiB，未触发内存或磁盘保护。运行在未完成帧消耗了大部分墙钟时间。

这些是候选数量急增与超时同时出现的证据，尚未量化CCD每个阶段耗时，也不能只凭它断言确切根因。新的预条件器改变了默认精度下的近似方向及后续轨迹，需把试探状态、内层误差和碰撞工作量分开检查；不应直接删减安全碰撞对、提高PCG预算或宣布Graph导致问题。

证据：`TIMEOUT_V43_20261003.json` 和下载目录的两份 `run.log`、`output/stats.json`、`trace/frames.csv`。

## 5. 独立安全与覆盖

- 悬挂100帧：207个接受推进＋99个静止桥接＝306段，CPU独立BVH＋Tight-Inclusion，容差1e-9，零碰撞标记。
- 兔子Graph：115个导出接受推进＋34个桥接＝149段，零碰撞标记。包括完整统计中106段，以及第35帧只有导出状态、未落盘统计的9段。
- 共455段；逐帧子步序号、完整帧接受计数、各帧初始状态及已完成帧末端逐位核对通过。第35帧另记 `unflushed_partial_frame_stats`，不伪造缺失的求解统计。
- 兔子检查限定导出表面顶点，Stable NH1允许FEM翻转；记录1847个累计FEM翻转计数、0个ABD翻转。仅证明这些导出部分路径的检查结果，不能代替100帧或物理质量通过。
- host未导出子步，本轮没有为host声称全接受路径CCD。

证据：`CCD_V43_chol_hang.json`、`CCD_V43_chol_bunny.json`、`VERIFICATION_V43_20261003.json`。

## 6. 结果保存、复现与下一步

564源文件、7组实际运行、参数、runner及二进制身份已本地核验。夹具包SHA256为 `0ce010a469e94e43a7f39dba165d8be0b355be79fe6307b5654ced84c284935f`；结果包302738462字节，SHA256为 `30536c0bc406388e48f64296a6d5d28ce79da22776fa86a327ef515e0c107e0d`。均已完整下载并安全解包至 `downloads/autodl_perf_v43_20261003`。

v42已完成重复包清理及10份相同因子硬链接去重，逻辑回收合计约505.5MiB。v43下载完成后，SSH连续三次连接超时，最终清理脚本未上传或执行；v43远端重复包清理、可选相同因子硬链接及最终远端身份/磁盘/GPU复核仍待连接恢复。最后成功读取磁盘约1.3GiB可用，不能当作当前值。两次仿真子进程已由runner按超时结束，最终显卡状态未在线复核。详细边界见 `REMOTE_FINALIZATION_V43_PENDING.json`；所有实验与失败证据已本地保存。

实际执行（AutoDL v43目录）：

```sh
bash tools/autodl_v43_build.sh
ctest --test-dir builds/autodl-stiff_perf_v43 --output-on-failure
GIPC_VALIDATE_COMPONENTS=$PWD/reports/components.json builds/autodl-stiff_perf_v43/gipc
python3 tools/benchmark_perf_v43.py --phase fixtures
python3 tools/finish_fixtures_v43.py
python3 tools/benchmark_perf_v43.py --phase smoke_corrected
python3 tools/benchmark_perf_v43.py --phase hang
python3 tools/benchmark_perf_v43.py --phase bunny
python3 tools/benchmark_perf_v43.py --phase host
```

`fixtures`原路径查找失败如前所述，后续三项由finish脚本独立保存；首次 `--phase smoke` 标签失败也保留。各场景完整命令在 `reports/V43_*.json`。重现使用新目录，不覆盖已有结果。

本地任务根以 `E:\Anaconda\envs\DL\python.exe` 执行 `tools/analyze_cholesky_v43.py`、`tools/analyze_timeout_v43.py`、`tools/verify_v43.py`，均返回0。独立安全命令：

```powershell
& 'builds/validator/Release/validate_path.exe' downloads/autodl_perf_v43_20261003/runs/autodl/autodl_perf_v43_chol_hang/trace reports/CCD_V43_chol_hang.json substeps --stable-nh1
& 'builds/validator/Release/diagnose_first_path.exe' downloads/autodl_perf_v43_20261003/runs/autodl/autodl_perf_v43_chol_bunny/trace reports/CCD_V43_chol_bunny.json substeps --stable-nh1
```

下一阶段先给第35/38帧前后的组装、PCG、试探更新和CCD查询增加逐阶段持久化记录/状态快照，保证超时仍留存；在同一保存状态下对照默认与更严格内层精度，检查试探几何范围、候选增长和窄相耗时。固定状态能够复现后，再决定修正方向或候选实现优化。当前默认保持关闭，不认证完整质量或性能收益。
