# M0 逐行检查范围与实施契约

本记录对应a4b83c0 native、两份本机构建和M0工具。逐行检查指下列与本阶段有关的函数/文件及新实现全部行，不声称审查了整个仓库所有第三方源码。

| 检查范围 | 精确核查结果 | 验证/处理 |
|---|---|---|
| root/baseline CMake，35/37 TU映射、实际NVCC/CL/link | 物理宏及Release CUDA设置相同；B0原对象命名会大小写冲突 | 外围对象名hash；从项目目录解析相对源路径；独立verify_m0 |
| app/benchmark_scenes.inl，SimpleSceneImporter::load_geometry，metis_sort缓存入口 | dt/材料从JSON加载后再应用环境override；FEM使用排序mesh/part | 所选两方JSON/原mesh/派生cache字节一致，native初始数组再比较 |
| app/gl_main.cu 1877–2027，frame_recording.h及统计导出 | checkpoint_save在目标物理帧之前保存；trace编号为local；velocity为实际设备速度 | 40+1恢复核对frame_context、位置和实际速度字节 |
| core/frame_checkpoint.cu全文 | schema2绑定binary_id/ABI/physical context；newton_tol也是context；先加载校验文件再transfer；BVH/Graph/MAS冷重建 | 新二进制新checkpoint；收紧outer从零运行，不绕过context守卫 |
| baseline/core/GIPC.cu solve_subIP 10975–11136 | 原版Kmin=6，从第6次更新开始累计beta，同时保留movement先退出 | B0不冒充movement-only，不使用当前程序Host开关当冻结原版 |
| active/core/GIPC.cu solve_subIP 11423–11890 | movement优先；res增加额外退出；forcing需legacy/rho1e-4并排除pool；重求包含失败尝试成本 | 新候选全关；CPU合同测试；Bunny14实际movement残差>.03 |
| solver/ipc_budget.h和ipc_residual_controller.h全文 | 参考在第6次更新之前冻结；正部标量包络；compensated不能早于direct；Kappa epoch记录不等于重置预算 | 40次同一实际前缀谓词核验，history_exit_veto=0；未称完整向量缺陷 |
| solver/ipc_residual.inl全文 | ABD广义梯度scalar max，FEM自由梯度component max；一次double2读回；无质量/J/h归一化 | 精确标注raw残差，不当位置/速度误差；未在M0更换门控 |
| abd_system/setup gradient边界分支及barrier gradient分支 | Fixed ABD梯度清零，barrier映射跳过fixed体 | Bunny观测abd_inf=0，排除固定ABD污染的猜测 |
| core/GIPC.cu::lineSearch 11132–11373，line_search_acceptance.h全文 | alpha已含CCD/CFL；初始相交缩步和energy缩步不同；9次预算；最终几何变化须重算能量；NaN失败 | CPU预算/非有限合同通过；自然M0没有相交回退，native失败fixtures留给M1 |
| pcg_graph_impl.inl控制scalar/终止及pcg_fixed_study.inl试验和恢复路径 | 兼容原版更新x/r后检查previous rho；record rho_stop非当前真实residual；固定probe恢复原x/系统 | 全A/rhs ABI/hash独立加载，CPU/native真残差对照，primary_restored_bitwise |
| femEnergy.cu SNK1能量及分支、core/GIPC.cu geometric_limit | SNK1对负J有限；IPC的InjectiveStepSize调用被注释；FullCCD不保证tet正J | 原版/当前Mixed均第13帧翻转；保留模型，独立rest-J指标 |
| tools/local/windows_owned_job.py全文 | suspended创建→AssignJob→resume，关闭句柄杀拥有进程，无广泛PID终止 | 61次完整运行均拥有Job关闭，失败不吞成成功 |
| tools/local/windows_runner.py execute/monitor/cleanup/validate，config.py expand/environment | 有限timeout/资源/foreign-load；环境清除ambient GIPC；旧from_zero恒True | 6行最小元数据修正；旧runner归档；新10恢复+1前缀验证 |
| 新M0分析/控制/构建/验证代码全部行 | 输出不覆盖；缺失native字段不补零；N/P与frame crosscheck；J用pristine rest；body层Fixed ABD不当自由顶点；源/输出/宏逐项记录 | 14项新增CPU合同+4项原native CPU合同；独立验证收据 |

## 实施中防住的问题

1. 终止组装行的P可为空，不能当作坏PCG记录；没有pcg对象才确认未做solve，字段缺失另记。
2. 仅比较总pair数不证明分类等价；M0不改query，后续P3必须比较typed multiset和MatIndex下游关系。
3. 不从两次不同轨迹时间差直接扣“日志开销”；正式与重型诊断分开，当前所有时间只作诊断。
4. 不把checkpoint变形坐标当rest；新增翻转fixture明确显示同一翻转状态相对自己会产生虚假的J=1。
5. legacy请求标记错不重写旧结果；用native上下文建立derived origin审计，并保留真实运行源码。
6. 原版没有velocity/resolved/checkpoint能力，保持capability=false；共同输入冷恢复只用于B1。
7. 解压逐Newton大CSV采用流式读；原数据SHA前后相同，帧数1..105、逐行index、N/P/phase均强检查。
8. 固定矩阵CPU模型保留原COO方向/对角块，不擅自对称化修正；记录asymmetry和零自由度，B_ref共用原MAS系数。
9. 接触/矩阵数值有效性与Graph地址有效性分开；frame checkpoint仍冷重建，不声称热继续逐位等价。
10. 固定/adaptive forcing规则不能混说：adaptive具有epoch/进展保护，fixed会显式覆写eta；本轮forcing off。
11. point BoundaryType==Free不意味着ABD body自由；最终analysis_v2按scene/body约束剔除fixed ABD，初版保留为superseded结果，不把它作为最终自由顶点误差。

## 精度验收状态

执行路线现在有同状态/分支/完整候选的验证入口，可以继续P2-A。数值路线需要共同显式线性norm和独立外层参考；现有gate不会约束所有movement退出，Bunny内层-only参考在困难状态仍有显著重复分歧，因此没有放行新的数值停止规则或低秩生产预条件器。

全场精度/物理认证继续pending。这是由实测参考分歧、模型翻转及缺失独立路径CCD证据产生的限制，不是为停止工作额外引入的假设性审批流程。
