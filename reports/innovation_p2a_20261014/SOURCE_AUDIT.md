# P2-A 消费者、依赖及逐项审查

范围：起点 a4b83c0；只处理 host AABB 和 ABD 能量结果传输。默认开关关闭，原融合 DCD、Full CCD、目标函数、材料、PCG 与退出规则不变。`baseline/` 不修改。M0 报告和封存结果不改写。

## 消费及依赖表

| 点 | 生产者、消费者 | 原等待 | 本轮处理 |
|---|---|---|---|
| `calcMaxBV` | CUB 合并叶 AABB，D2D 写根；Morton 编码读 device 根 | 同步 D2H 48 B，host `scene` | device 归约和 D2D 保留；lazy 开关决定是否读回 |
| face/edge `ConstructRebuild` | 归约 → Morton → 排序 → 节点 → 内部包围盒；query 全部消费 device | 两树各一次 host 根读回 | 同一提交顺序；读取 CPU 根时调用 `host_scene()` |
| `discrete_refit` | 当前坐标 → 全部叶 → 原 permutation → 全部内部界 | face/edge 各一次 D2H | 更新后标记 host 缓存过期；off 保留 eager 读回 |
| `ConstructFullCCD` | 当前坐标、方向、alpha → swept 叶和根 → Morton/tree | 与 rebuild 相同 | 只延迟 host 结果；所有 device 计算保留 |
| `RefitFullCCD` | 重算 swept 叶、排序、内部界 | 原来没有 host 根刷新 | 不新增生产读回；标记 host 缓存过期以供显式消费者 |
| `getSceneSize` | 重算叶和 device 根，返回 device 指针 | 旧 D2H 的返回值被丢弃 | off 仍读回；on 只保留 device 路径 |
| `GIPC::init` | 唯一实际 CPU 消费者：根 → `bboxDiagSize2` → `dHat`/`dTol` | 直接读公开 `scene` | 改为有效性 accessor；必须在构建根后读取 |
| 两个注释中的 `bvh_f.scene` | 注释，不执行 | 无 | 不作为消费者或成本证据 |
| `cal_abd_kinetic_energy` | 原 kernel → 原 CUB Sum → DeviceVar 隐式转换 | D2H 8 B | 原 API 保留；可选择只排队原 kernel/Sum，目标地址由调用者提供 |
| `cal_abd_shape_energy` | 原 kernel → 原 CUB Sum → DeviceVar 隐式转换 | D2H 8 B | 同上，保持 kinetic 后 shape 顺序 |
| batch FEM 能量 | 九组原 partials/归约 → D2H 72 B | 然后再执行/读回 ABD 两项 | on 时两项写到结果槽 9/10，一次读 88 B；host 仍从 c0 开始，先加 ABD kinetic、shape，再加原各项 |
| scalar 能量 | FEM kinetic 的标量结果，之后两项 ABD | 两次 ABD D2H | on 时相邻两项一次读回，两个标量依次加回；不会改变 TOI-AL 的原 scalar 调用者 |
| `lineSearch` 入口 `wait_device()` | 方向分发和几何限步之后；后续多处结果读回亦会等待 | `cudaDeviceSynchronize` | 保留，仅新增 CPU cost scope；没有删除错误检查或跨 stream 屏障 |

## 缓存与所有权检查

1. `scene` 不再向生产消费者公开，避免直接读过期字段。host 值、是否当前、根是否已计算，随完整 ordinary/swept storage bundle 一同交换。
2. 初次分配/释放没有有效根；accessor 拒绝读取。重建/归约/实际 refit 完成排队后，根可在同一 owner thread 上通过同步 D2H 消费。输入坐标/拓扑改变后仍须先调用对应构建/refit，接口不承诺在尚未更新的 device 根上推导新几何。
3. swept 私有存储 resize 使 host 根无效。既有 topology/signature/interval 控制逻辑不改。没有把另一棵树的 host 结果借给当前树。
4. ABD enqueue 结果地址在发射前 resize：batch 固定在当前结果缓冲内；scalar pair 使用 ABDSystem 自有缓冲。零 ABD 明确输出两个 0；不读取上次结果。没有临时 stack 的 device 指针或异步 host 生命周期问题。
5. kinetic、shape 的数学 kernel 无修改。两次 Sum 仍串行，不能同时在不同 stream 共享原 CUB scratch。
6. 生产读取为同步 `cudaMemcpy`；不宣称 overlap。CostTrace 的 CPU readback 窗口包含等待前序 GPU 工作，不等于纯 8/48 B 传输耗时。父相位与子项不可重复相加。
7. `GIPC_P2_AUDIT=1` 逐次重算原两项并比较 16 B 位级值，只用于正确性运行；所有配对计时关闭此开关和 CostTrace。
8. 原 `line_search_acceptance.h`、步长循环、预算 9、初始/后置几何检查、失败及 forcing retry 路径未修改。用原 CPU/native 失败测试和真实短场景核对，而不是以很小的总能量相对差取代分支一致性。

## 编译和官方依赖依据

检查生成/实际编译记录中所有相关 CUDA TU 的 `--default-stream=per-thread`，以及 Graph replay 使用 `cudaStreamPerThread`。原 CUB wrapper 默认 stream 0 在这些 TU 上遵循 per-thread 默认流；同 owner thread 的 kernel、归约、结果复制按顺序排队。

- [CUDA 13.0 Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)：per-thread 默认 stream 与编译选项。
- [NVIDIA CUB device-scope 文档](https://nvidia.github.io/cccl/unstable/cub/developer/device_scope.html)：同 stream 顺序复用临时存储的条件。

## 本轮有限验证预登记

- CPU 配置契约 5 项；现有 M0 runtime contracts。
- native host-scene 生命周期，lazy off/on 各一份；ABD 真 kernel/API 输出测试，1/3/513/0/2 body，包含 Free/Fixed/Motor 和边界 offset。
- 原 line-search 接受/失败、PCG forcing、budget/controller、普通 BVH/contact/query 相关 tests。
- Boxes17、Mixed15、Bunny14：新 binary 内重新捕获 M0 所选事件的 pristine prefix，off/on 都从同一 checkpoint 恢复。禁止改旧 binary_id。
- 三状态四臂：control、scene、abd、both；成本诊断每臂 1 次，正确性 control/both 每臂 1 次，轻量计时 3 对。
- 各状态从零短段 control/both 3 对，用于观察整段成本。不是 100 帧正式认证，不复用插桩时间证明收益。
- 复杂 P2-B 的继续门槛仍为完整线搜索约 15% 净减少且整段约 5% 的可信空间。三对仅用于工程筛选，不声称正式置信度或 2 倍加速。

尚未验证的极端 full-path 状态会在结果报告中明确列出；不能把入口 invalid-alpha fixture 说成已覆盖真实几何回退。
