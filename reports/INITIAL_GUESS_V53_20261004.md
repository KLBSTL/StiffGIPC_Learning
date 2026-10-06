# v53 完整目标能量选择 AL 子问题初值

2026-10-04，本机 RTX3070 Laptop 8GiB / CUDA13 / VS2022 Release。**v53 已单次从零完成混合场景 100 帧（497.844 s），641 段接受路径/桥接独立 CCD 无标记；同程序关闭开关只完成 38 帧后超时。两次 450 s 预算组仍未通过。候选默认关闭，尚不认证可靠修复、物理质量或性能。**

## 依据与实现

v49 真正未完成 outer 的快照中，trial 与 safe 最大距离 11.7778 m；trial 的 FEM 最小 J 为 -316.98、8088 个非正四面体，safe 为 -2.4926、38 个。两者 FEM 能量分别 3.69560 / 0.0054363，分别最小化 slack 后的 AL 能量约 .371257 / .0000976。**这些只是两个能量项，且快照来自 inner=63；不能当作 outer 起点的完整能量或根因证明。**

参考实现也保留上一 outer 的自由试探状态，因此不能认定“保留 trial”本身是移植错误。v53 是默认关闭的算法候选：重新线性化接触后，分别刷新 slack，以现有 GPU `computeEnergy` 比较旧 trial 和当前 safe 的完整目标能量，选择较低者作为新子问题的初值。复制顶点和 ABD q；第一轮 Newton 为胜出的初值重新刷新 slack。

随后沿用原 Newton、固定 slack 曲率、线搜索、乘子更新、原始方向判停和完整 CCD。材料、dt、PCG 容差/上限、接触候选上限未改，v50–v52 三候选均关闭。此改动可能改变非凸问题得到的局部解，完成性改善不能替代物理质量验收。

源码：`sources/stiff_perf_v53/StiffGIPC/solver/toi_solver.cu`。开关：`GIPC_TOI_CHOOSE_START=1`，runner 参数 `--choose-start 1`，默认 0。单文件 overlay 复用冻结 v50 的其余源码/资产/对象文件，旧源码和旧程序身份保留。

## 验证及已知结果

- Release 编译通过；43 项 GPU 组件和 PCG 守卫夹具通过。v53 未重复运行 CTest，不将旧版 CTest 结果写成本轮结果。
- 悬挂场景开关各 3 帧完成，最大顶点差 2.4633e-15；不宣称逐位一致。
- 统一参数：dt=.01，rho=1e-4，速度阈值=.05，MAS Cholesky，native inner exit，Graph，CCD 候选上限 10,000,000。保留追踪和 PCG 审计，所有时间均为诊断运行耗时。

| 运行 | 完成 | 墙钟秒 | 已落盘 PCG / 最大迭代 / 触顶 | 初值比较 / 选 safe | 独立 CCD 段 / 标记 |
|---|---:|---:|---:|---:|---:|
| v53_bunny40 | 40/40 | 56.891 | 185 / 675 / 0 | 48 / 14 | 111 / 0 |
| v53_bunny100 | 97/100，预算超时 | 450.188 | 889 / 1560 / 0 | 468 / 310 | 592 / 0 |
| v53_bunny100_repeat | 95/100，预算超时 | 450.516 | 872 / 1005 / 0 | 460 / 312 | 580 / 0 |
| v53_bunny40_off | 38/40，预算超时 | 180.453 | 267 / 626 / 0 | 关闭 | 199 / 0 |
| v53_bunny100_completion | 100/100，独立 600 s 预算组 | 497.844 | 953 / 1340 / 0 | 518 / 345 | 641 / 0 |

第一组长测试在 97 个完整帧中单帧最多 23 次 Newton（第 74 帧），未重现此前一个帧内数百次内层求解。仍未通过 100 帧门槛；第 98 帧未完成的求解不能计为通过。按原 450 秒预算从零重复，保留首组超时证据。中途观察到的最大值 18 不代表最终整组最大值。

全部已审计初值选择都选择了较低的完整能量，方向更新通过原生能量接受及退出语义检查。40 帧最大初值能量下降 7.43331，首组长测试 39.66432，完成组 17.90544。完成组单帧最多 21 次 Newton，95 帧重复组最多 22 次。所有五组已落盘 PCG 均实际执行 conditional_graph，无已记录的触顶或 breakdown。

关闭开关组的前 38 帧累计 solver 60.963 s，剩余约 119 s 仍未完成第 39 帧；该未完成帧没有完整统计，不能推断其 PCG 均通过。开启组多次越过该位置，支持改善推进表现，但装配存在非确定性、这里只完成一次关闭对照，尚不能证明所有停滞都由同一个原因造成。

两次 450 s 试验结束后，根据持续推进的证据，事先在 WORK_PLAN 中声明并向用户说明：新增一次**独立、从零、600 s 预算的完成性试验**。只调整总墙钟超时，不改变 PCG/CCD 或物理/收敛参数；不是延长原来的进程。该组 solver 495.736 s，正常退出，全部 39475 顶点有限，最低采样空闲显存 4160 MiB。它证明本候选至少一次可完成 100 帧，**不将之前两次超时重新计为通过，也不是 450 s 性能门槛通过**。

## 形变与覆盖限制

| 运行 | 帧端点最小 FEM J | 单端点最多非正 FEM tet | 布料最大边长比 | CCD 路径累计 FEM 翻转观察 |
|---|---:|---:|---:|---:|
| 40 帧 | -0.300852 | 1 | 1.089011 | 14 |
| 首组 97 帧 | -0.669901 | 20 | 1.247922 | 1049 |
| 重复 95 帧 | -0.933804 | 17 | 1.302986 | 1109 |
| 完整 100 帧 | -0.536147 | 14 | 1.350458 | 985 |

独立 CPU BVH + Tight-Inclusion 检查接受子步及跨帧桥接，遵循现有 Stable NH1 可翻转材料策略；路径零碰撞标记不等于 FEM 保正或完整物理质量合格。累计翻转观察会重复计数同一个单元，不是不同坏单元的数量。尚未建立共同物理参考或质量容差，也没有质量匹配的性能结论。

完整组覆盖 542 段接受推进和 99 段静止跨帧连接，端点与导出帧状态核对通过，CCD 进程退出码 0；ABD 翻转观察为 0。全部运行已经结束，无本任务遗留 `gipc` 进程。

## 复现和身份

工作目录为本任务根目录，使用 `E:/Anaconda/envs/DL/python.exe`：

```text
python tools/validate_v53.py smoke
python tools/validate_v53.py bunny
python tools/validate_v53.py long
python tools/audit_initial_guess_v53.py v53_bunny40
python tools/audit_initial_guess_v53.py v53_bunny100
python tools/analyze_local_v53.py v53_bunny40 --ccd --output reports/V53_VERIFICATION_40.json
python tools/analyze_local_v53.py v53_bunny100 --ccd --output reports/V53_VERIFICATION_100.json
python tools/validate_v53.py control
python -c "import sys; sys.path.insert(0,'tools'); from validate_v53 import run; assert run('bunny100_repeat','bunny_cloth_bunny_l',100,timeout=450)"
python -c "import sys; sys.path.insert(0,'tools'); from validate_v53 import run; assert run('bunny100_completion','bunny_cloth_bunny_l',100,timeout=600)"
python tools/audit_initial_guess_v53.py v53_bunny100_completion
python tools/analyze_local_v53.py v53_bunny100_completion --ccd --output reports/V53_VERIFICATION_COMPLETION.json
```

工具拒绝覆盖已有运行/报告；再次复现应使用新运行名。审计和验证 JSON、requested.json、run.log、stats.json 和导出轨迹均保留。源码/程序清单 `manifests/perf_v53_local.json`，294 源文件身份核验通过。

- source_digest: `ff4fdf2f6beb10381b2d2319031023e685c8513cb0903039925658e8a4550e32`
- executable SHA256: `774494860fc4578a990a20fe09743f0c378d8fd5e6d22c3fceaeb4a4010ec6fb`

## 下一步边界

保留并冻结 v53，优先对这个已有改善的候选做相同预算的完整 100 帧重复与 host 对照、共同物理参考及形变检查。原 450 s 指标仍未达到，尚无有效的质量匹配性能结论。不要把单次完成当作任务结束，也不要在缺乏新失败证据时继续叠加新的求解器变体；剩余正常推进的成本应通过有限分项计时定位。
