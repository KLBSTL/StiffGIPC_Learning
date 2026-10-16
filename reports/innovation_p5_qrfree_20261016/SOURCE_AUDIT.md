# 逐项实现复核

范围为本轮新增代码的每个控制/数组/代数路径，生产代码只读；不是对整个仓库重新逐行审计的声明。

| 路径 | 检查及证据 |
|---|---|
| qrfree_numeric.py:7–24 | Z维度/有限；W保留原A；只对小E取对称部分；先谱拒绝再Cholesky，无jitter；作用中的Wᵀ和两次粗解与原balanced一致。可逆换基、微小非对称及拒绝测试通过。 |
| qrfree_numeric.py:26–31 | selected_eigenvalues与系数列顺序/数量匹配，严格正有限，按列除sqrt，不选错误的最小谱方向。 |
| run_qrfree_cpu.py | 源只取上轮冻结S/H，目标解不参与basis；先核验父封存；rank8及场景固定；QR强对照次数与旧结果一致；每次共同真残差与Decimal50；所有目标解和完整trace保存。 |
| frozen_runtime.cuh | 上轮cgstats_cost.cu在void run前的文本一致前缀，生成时LF转成CRLF，并非字节一致；Device/CSR/参考MAS/Replay表达式保持原实现。ORIGIN记录完整原文件hash；CUDA/库全部绑定同一stream；FP64参考不冒称native。 |
| qrfree_cost.cu:4–36 | 新目录输出、ABI、n/width/steps、历史写入边界与最终slot下标/标量上下文；各buffer按列主序预分配，allocation总预算；n×8 cuSOLVER与SpMM leading dimensions一致。 |
| qrfree_cost.cu:37–52 | rho为更新前rho，后缀和索引max(t_i,t_j)；CPU特征升序取末端最大8列；候选才缩放。persistent host_U生命周期跨异步H2D，下一轮读回同步保证无悬空/覆盖。 |
| qrfree_cost.cu:53–67 | QR强对照保持geqrf/对角秩检查/orgqr；候选不调用这些算子。候选E读回/sym谱门槛/Cholesky在计时内；负例以唯一指定错误拒绝，未靠库失败蒙混。 |
| qrfree_cost.cu:83–104 | Z/W/E固定后才捕获Graph；两次E解及Wᵀ不变。copy、gemv、MAS和核调用都在同stream；每种坐标Graph对象先销毁再重建/覆盖矩阵；三个独立probe、Graph/host检查与完整窗口保留。 |
| qrfree_cost.cu:105–110 | 每对按pair/order交替QR/free，共5对，无筛值；每个pipeline真做读回/谱/上传/提取/检查。所有矩阵输出在覆写前保存。失败写failure.json并非成功result。 |
| prepare_qrfree_gpu.py | 借用旧payload用绝对路径且hash先核验；不复制/覆写旧数据；负例实际让S秩一但统计仍满秩，只在负例跳过QR对照以检验候选独立拒绝。 |
| run_qrfree_probe.py | 保留同一全局GPU锁/owned Windows Job；电源逐采样、180秒/1GiB保护；仅清理拥有的进程；预期退出必须匹配指定rank错误。第一次沙箱sanitizer超时和清理结果未删除。 |
| validate_qrfree_gpu.py | Python重建MAS，独立原A验证AZ/E/因子、动作和空间；QR仅作验证；检查每对排程和每个作用窗口完整；reference增量和迁移CPU次数仅为条件模型。 |
| qrfree_audit_export.py | 最终重载6解/492行曲线，6次Decimal50独立检查，verify所有GPU输入/结果和构建；5个GPU Job含1次失败记录；memcheck必须有有效0-errors输出。首版提前审计失败保留执行源及日志。 |

CPU没有偷偷把候选重新正交化；小矩阵被拒绝时也没有自动回退/修补。独立验证需要计算QR以测空间相等，它在验证器里，未纳入算法输入或setup。

原生FP32完整统计/history的缺失已核实，不以旧少量MAS动作校准代替整条历史校准。本轮未声称当前谱误差或迭代数可迁移到native。
