# AutoDL 七场景 120 帧测试计划

**目标：**按用户要求执行 21 次从零开始的运行，并同步既有实现到 GitHub。
**架构：**复用公共 Linux runner、资源保护和 CPU 分析；本轮不改求解器。
**环境：**RTX 4090，CUDA 12.8，Python/NumPy，独立干净构建。

- [x] 将既有提交推送到 `KLBSTL/StiffGIPC_Learning/main`（`3d8c15e`）。
- [x] 准备七场景公共输入、干净构建和实际对象／链接身份。
- [x] 冻结 21 个配置及程序身份，再逐项运行、检查和分析。
- [x] 汇总求解时间、core event、进程 wall、五阶段时间和 PCG 工作量。
- [x] 下载报告、轻量原始证据及关键质量差异；复现工具已同步，完成报告随后提交。

七场景：`cloth_hang_l/m`、`cloth_sphere7_l/m`、`cloth_fixed_bunny_l/m`、`bunny_cloth_bunny_l`。
前三类同时覆盖两个分辨率；混合兔子检查 ABD/FEM/布料耦合。选择在运行前固定，不按结果替换场景。

三个配置：

1. `base`：冻结原版 Stiff，host。
2. `graph`：活动 IPC，conditional CUDA Graph，K1，其余执行组件关闭。
3. `all`：graph＋FullCCD refit＋批量能量＋能量复用＋普通 BVH refit（重建周期 8）。

“所有改进”表示当前保留的执行组合。AL-TOI、残差停止实验、退休融合、K4、接触池不属于该组合；未实施的 segmented energy / row-gather SpMV 不计入。
全部保持 legacy MAS、FP64、rho=1e-4、dt=.01、累计阈值 .01、最少六次有效更新及完整碰撞安全。

每配置只运行一次 120 帧；场景间轮换三臂顺序。预声明单次上限 180 秒，不自动重跑、延长或添加预热模拟。
用户随后明确要求自动跑完并允许暂停对话，因此由后台 batch 在每项 CPU 分析结束后自动推进；无需对话保持在线。
配置／资源／输入身份错误停止整个队列；孤立数值失败保留证据，其他从零独立运行继续，但失败场景不输出有效加速比。
保留现有启动磁盘 4 GiB、运行磁盘 1 GiB、显存储备与外来 GPU 进程保护。GPU 串行。
网格排序缓存在运行前用 CPU 准备，并逐项核对两程序的公共输入哈希。

主速度比使用 `frames.csv` 的同步求解包络累计（含图准备、矩阵转换、MAS），加载和导出在进程 wall 中另列。
core event 嵌套于求解包络，不能与 CPU 时间相加；旧／新阶段字段覆盖差异单独说明。
Graph/base 比包含活动代码与原版的其他实现差异，本轮没有活动 host 臂，不能严格隔离纯 Graph 因果收益。

只对完整 120 帧、配置核对、输入一致且硬检查通过的三臂计算比例。保存有限状态、PCG 触顶／breakdown、材料指标和分体位置差异。
原版没有实际速度和完整 breakdown 观测能力；不以位置差分伪造速度。
单次结果没有置信区间、没有新基线标定、没有独立接受路径 CCD 审计，因此不认证同质量加速或统计 2×。

复现入口：`python tools/seven_eval/batch.py --build-pid PID`。后台等待当前构建身份文件，最多 1800 秒；此后自动封存、串行运行索引 1…21，并输出 JSON、CSV、Markdown。
手动复核入口仍为 `python tools/seven_eval/run.py seal` 与 `xvfb-run -a python tools/seven_eval/run.py run --index N`，N=1…21。
批任务总上限 6000 秒，每次模拟仍为 180 秒，无自动模拟重试。后台日志 `BATCH.log`、状态 `STATUS.json` 与逐项摘要随时可读取。

构建准备曾因未写旧审计接口的开始标记而停止，尚未执行任何模拟。恢复只读取真实配置日志的文件创建时间作为保守时间下界；不伪造 supervisor 开始时间，也不重新编译。
原始停止状态／日志另行保存。实际对象、Ninja 命令、链接、源码前后快照仍完整核验；真实 supervisor UTC 明确标为不可取得。
