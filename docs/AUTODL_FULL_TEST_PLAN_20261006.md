# AutoDL 完整运行与配对计时计划

2026-10-06，用户指定新端口22209。位置切换由本次请求授权；此前本机计划保留为历史。

## 配置和比较

源码起点为`5e2f9bba4abadf21591aa28d4e6b039447c3f181`。新建独立目录和两套全新Linux
Release/SM89构建，不改冻结Stiff，不清理旧目录。GPU实验串行；保留失败证据，不自动重试。

场景：悬挂布料`cloth_hang_l`、固定兔子布料`cloth_fixed_bunny_l`、混合兔子
`bunny_cloth_bunny_l`。每次从零连续100帧，dt=.01，材料/ABD/FEM/完整CCD保持原值，
legacy累计阈值.01、min6、PCG rho1e-4。K1、raw edge order、接触池关闭。

| 臂 | 程序 | 执行配置 | 用途 |
|---|---|---|---|
| original_stiff | 独立冻结baseline | 原host实现 | 同轮正式分母 |
| ipc_host | 活动源码 | base执行组件、host PCG | 当前IPC执行参考 |
| ipc_graph | 活动源码 | base执行组件、conditional Graph | 隔离Graph贡献 |
| combined_graph | 活动源码 | conditional Graph、现有refit/批量能量/能量复用/MAS拓扑复用、普通BVH refit周期8 | 当前执行组合主测试 |

K4在本机短/长冻结系统未达收益门槛，本轮不将其当作最快配置。AL-TOI、gated、compensated
属于另行算法实验，不混入本次执行优化比；不关闭原累计出口或放松停止规则。

## 有界顺序

1. 核对GPU/资源/工具；上传冻结源码。新目录内清洁构建active/base，保存编译前后源、
   实际对象/链接/二进制和输入身份。独立CPU合同及7项原生CTest，GPU测试串行。
2. 每场景先5次仅Stiff：前三次标定，后两次独立复核。在看候选质量前冻结整段材料范围
   与数值比较容差1e-12；不能根据候选扩大范围。旧失败质量界限不改写。
3. 每场景7组Stiff/combined交错配对，前3组另测host/Graph。共每场景25次、最多75次100帧。
   每次完成后CPU分析再发下一次；前5基线不混入配对速度样本。
4. 每场景仅当全部100帧硬检查通过且冻结整段材料比较通过，再允许1次300帧combined
   稳定性检查；否则写未满足条件的跳过记录。最多3次，不借超时追加预算。

每100帧上限120秒，300帧上限600秒；原GPU显存、磁盘储备和独占锁保护不变。
数值/config/资源失败停止相应场景配置，后续相同配置记录跳过；材料差异记录详细诊断，
不自动把当前组合认证为同质量加速或推广默认。

## 指标和证据

报告全部样本与配对中位：Stiff/combined、当前host/Graph、Graph/combined的独立贡献。
七对主比给出单侧95%配对自助法下界；负载控制和质量有缺口时仅作统计诊断。

实际记录：wall、每帧solver host时间、core CUDA-event总时间，加载/导出尽可能单列；
assembly、整个linear阶段、CCD、line search及未分类成本。linear含矩阵转换/预条件器准备，
不能全称PCG kernel。host同步等待不能与GPU事件相加。保留Newton方向数、PCG总迭代、
每方向迭代、触顶/breakdown/非有限和退出原因实际可用性。

全程最大/p99布料拉伸、混合FEM最小J/非正单元/负体积、ABD翻转/固定漂移，位置及活动
程序实际速度按布料/FEM/ABD分别统计。原版无实际速度或resolved-config时明确缺失，不
通过差分重构冒充实际速度。本轮没有自动继承接受路径CPU CCD，因此完整运行不等同于
完整碰撞/物理质量认证；2×只有证据充分时才能报告实现。

每run保留配置、二进制/输入SHA、GPU负载/时钟/显存、资源收据、原始日志与CPU分析。
磁盘受限时只归档本轮已结束的原始大状态文件：先生成archive、回传本机并对SHA，再移除
新目录中已存档的对应raw文件，留下manifest/trace/analysis和archive_receipt。旧文件不动，
归档保留完整原始证据；不以删除失败数据腾空间。报告和紧凑索引回传仓库。

## 执行后勘误（2026-10-07）

保留上面的原计划文字。实际公共runner的combined preset仅启用FullCCD refit、批量能量、
能量复用，并另启用普通BVH refit周期8；`GIPC_ACCEL_SUITE=0`且未设置
`GIPC_MAS_STATIC_TOPOLOGY`，所以无接触MAS拓扑复用实际关闭。本轮结果不归因于该组件，
没有在计时中追加开关或改变配置。完整执行和质量结果见
[AutoDL报告](../reports/autodl_full_20261006/FULL_TEST_REPORT.md)。
