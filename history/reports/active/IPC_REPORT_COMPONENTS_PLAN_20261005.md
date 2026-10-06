# 两份报告执行组件实施与本机测试计划

## 范围与冻结边界

按用户最新指令，先实现两个报告中的执行候选，再测量质量和速度；此前“成本证据不足所以暂不实现”的启动决定由这次明确指令覆盖。候选仍默认关闭，原停止参数、材料、完整安全 CCD、质量界限和失败决定均保持。

选择报告 §4.2 的 MAS 最终输出/点积融合和 §6.2 的普通离散 BVH refit。第二份报告的 residual gated / compensated 已有实现并完成有界试验，本轮不重新调参。Hessian 符号拆分不同时展开。

参考程序为上轮轻量观测构建 SHA `d41c9553ab50ff95bcdfb293dca638c625ea278449d1fbfa2ae9448a9b3ea710`；原版 Stiff 与旧 Graph 的身份保留。公共入口 `tools/active/config.py` / `run.py`，一个活动 overlay；两个子代理分别负责独立组件，GPU 与编译串行。

## 组件 A：最终 MAS 输出与 r.z 融合

- 独立 `mas_fused_dot=false` / `GIPC_MAS_FUSED_DOT`，不继承 combined。
- FEM 最终 gather 在拥有最终 z 的位置产生 block partial；最终 ABD 等非 FEM 区间单独贡献；统一设备归约。不能遗漏或双算 ABD，不能读取未完成的全局原子结果。
- host 和 conditional Graph 均可选择；不改变停止时序和预条件器。scratch 独立拥有，Graph signature 包含地址、长度及开关。
- `fixed_mas_dot_study` 在同进程保护固定 A/b/M，交错核对旧/新 z、rho、完整准备+PCG成本；恢复主方向和临时状态。实际 effective 状态及不生效原因进入每个 PCG 记录。

## 组件 B：普通离散 BVH refit

- 独立 `discrete_bvh_refit=false`、固定 `discrete_bvh_rebuild_interval=8`；本轮不搜索间隔网格。
- 首次、拓扑身份/数据映射/缓冲改变、swept 模式切换时全重建，其余符合条件调用更新全部当前叶及内部 AABB。
- 普通与 FullCCD 当前共享节点，模式变化必须失效普通拓扑，不能为提升命中率忽略 swept 构建。索引内容在 init/invalidate 之间视为不变；原地重编号需显式失效。
- 不复用接触集合，不删 CCD；保留原叶过滤。栈深度/编号/拓扑在诊断中核对。
- `discrete_bvh_validate` 主动同树 refit 后查询，与同状态全重建查询逐项比较规范化 pair、多重集合、类型及 MatIndex 关系。额外 allocation、同步和查询只在诊断开启时发生；失败抛错，不能静默继续。

## 本机有限验证

1. 配置契约、Release 构建与源/对象/链接身份核对；8 项既有 GPU fixture。
2. 短启动：两布料 old/dot/bvh/both 四臂，混合 old/both、两布料 both host，各 3 帧。逐项核对请求、resolved、实际 PCG 与组件计数。
3. 受保护固定系统：两布料 f2/direction1 的 MAS dot 对照；普通 BVH 三场景各一次 3 帧强制同状态核对。静态审查指出初始布料可能零接触，因此在任何GPU运行前另声明固定兔子59帧非空接触对照，保留原计划/protocol及v2追加身份。失败保留，显存/磁盘保留线不放宽。
4. 两布料 dt=.01、从零 100 帧，原版 Stiff / 旧 combined Graph / +dot / +discrete / +both 五臂，各 3 轮交错，共 30 次。关闭重型诊断和真实速度导出；每次运行后主动检查 PCG cap/breakdown/非有限/ABD 翻转，硬失败停止对应 scene×arm 后续重复。
5. 独立质量诊断：两布料观测原版/旧组合/dot/bvh/both，各一次 100 帧真实速度导出；四个活动臂采用相同接受子步导出方式，避免只对新组合增加观测。优先对 both 两条轨迹作独立 CPU CCD 审计。观测原版没有该子步插桩，仅作单独参考，不能把其差值全归因于组件。CPU 每场景上限 300 秒，诊断计时不得混入性能配对。
6. 混合模型 old/both 各一次 35 帧真实速度兼容检查，两者统一子步导出；没有同轮混合材料冻结认证，报告 FEM/ABD 分项及硬失败，不假装质量通过。

所有 GPU 运行串行，单次上限 120 秒；没有 300 帧/AutoDL/新清理。资源失败与数值失败区分，缺失覆盖明确待验，不临时重试或延长。

## 分析与交付

- 保留同轮 Stiff、旧组合和各单组件，报告配对耗时比、PCG/方向数、实际 build/refit/validation 数量。新实现不等于提速。
- 材料使用原 `ipc_revision_20261005_quality_protocol.json` 整段 max/p99 拉伸与固定漂移界限；原版自身越界仍标记协议无法认证，不增加形变预算。
- 位置和真实速度按 cloth/FEM/ABD 分开报告。只有独立诊断轨迹接受路径 CCD 可宣称对应覆盖，不能继承旧零标记或覆盖全部计时运行。
- 两组件各有独立模块说明、旧路径、开关、生命周期和调试入口；主报告、不可覆盖分析与实验索引追加记录。
- 未通过质量的配置不推广默认；局部或整体没有收益同样保留实测结果，不为了推广改参数或扩大候选。
