# AutoDL 两场景、两分辨率100帧组件收益计划

本轮只测试当前执行实现，不修改原生算法或默认值。复用上一轮已核对的4090程序，
独立保存新运行与输入身份，不覆盖75次历史结果。

## 预先选场

| 类型 | L总顶点/布料顶点 | M总顶点/布料顶点 | 选择依据 |
|---|---:|---:|---|
| 悬挂 cloth_hang | 1939/1939 | 7593/7593 | 现有L测试host/Graph约1.241×、Graph/组合约1.141×，更可能体现发射和已有组件收益 |
| 布料落球 cloth_sphere7 | 3084/2601 | 10684/10201 | 固定球接触；历史L在dt=.01约f22出现原生barrier接触，覆盖持续障碍物接触 |

选择在本轮结果产生前固定；报告四项全部结果，不在测后仅选最快项。历史其他版本的
落球收益仅支持选场，不能当作本轮预测。两个类型各自L/M的全部effective_scalar_fields
一致，但不同类型的材料不同；不能跨类型把差异全部归给分辨率。
网格实际存在，部分JSON的mesh_generation_pending标签陈旧。

暂不选table：固定桌面会读取baseline独有的cipc_table排序缓存，需额外处理公平性；
stack10的已有收益较小且M为102010顶点，资源成本高。

## 固定矩阵与预算

四个case为cloth_hang_l、cloth_sphere7_l、cloth_hang_m、cloth_sphere7_m。
每个case的四臂分别从零运行100帧，共16次：original_stiff、ipc_host、ipc_graph、combined_graph。
各case交替正/反顺序。每臂单次样本属于诊断，不给置信区间或正式性能认证；上一轮的
七对主比/三对组件比单独保留。

全部dt=.01、legacy累计.01、minimum6、PCG rho1e-4、legacy MAS、Graph K1、pool关闭、
raw查询、完整CCD。combined实际为FullCCD refit、批量能量、能量复用及普通BVH refit周期8；
MAS topology reuse关闭。没有AL-TOI/gated/compensated或停止条件调整。

GPU串行，每次100帧120秒；保留原显存/磁盘/外来compute及启动负载保护。每次先做
CPU分析再进入下一项。资源/数值/配置硬失败不重试，该case其余臂跳过；材料差异保留
但不自动扩展重复次数。没有300帧或参数网格。

## 输入补充与复现

M网格需要新增METIS排序缓存。在正式运行前，用原构建的METIS静态库编译一个CPU包装
入口生成一次，再逐字节复制到baseline。已有缓存不覆盖，已有原始资产/代码不修改。
缓存生成不混入wall或求解时间。保存包装源码/编译命令/程序SHA、原始网格SHA、排序
obj/part SHA及双方一致性。只允许新增被这四个case明确引用的派生缓存。

保留旧构建证据和旧seal；新seal显式记录“复用旧可执行程序＋新增运行输入”。旧源、
对象、链接、编译器及原资产记录全部核验，实际输入身份在四臂的初态、拓扑、质量、
边界、body IDs和场景对象上逐项核对。不能声称程序重编译了新增缓存。

## 收益与质量报告

- CUDA Graph收益 = ipc_host / ipc_graph；其他组件总收益 = ipc_graph / combined_graph。
- 总收益 = original_stiff / combined_graph。另列Stiff/active host，解释活动基础实现差异。
- 每项同时列求解秒、进程wall、core CUDA event、线性方向数、PCG总量，时间节省率为1−1/比值。
- 分开列装配、整个linear、CCD、线搜索、状态更新；不把linear全部称PCG kernel，不把嵌套时间相加。
- 记录首次原生barrier活动接触和接触帧数；这不是独立接受路径CCD认证。
- 逐臂比较最大/p99拉伸、固定漂移、ABD/FEM、分体位置及可取得的实际速度。原版无真实速度和独立CPU接受路径审计，单次基线也不构成质量标定范围。

上一轮与本轮收益汇总到一张表，明确样本数和分母；不同配对中位数不能简单相乘或相加。
2×只按真实结果判断，不通过改材料、停止阈值或GPU负载取得。
