# 无接触阶段退出语义：待验证假设

**2026-10-05 修正：原文的悬挂观察保留，但该假设不能直接作为 mixed 第 1 帧差异的首要解释。** 新复核显示 mixed 的 TOI AL 集确实为空，而 IPC 屏障半径为 0.00333508 m，是日志几何接触分类距离/TOI delta 的 10 倍；IPC 帧末仍有 2625 个 narrow self pairs，最小距离 0.00215039 m。两后端第一次 PCG 的迭代数与 rho 在任何退出判断前已经不同；TOI 末方向轴速度 0.0053394 m/s 又低于该场景 Stiff 的名义 0.0333508 m/s 阈值。故 AL 活跃集为空不等于 IPC 没有屏障作用，不能把退出差异当作独立主因。先做同状态、接触/非接触分项 A/b 与预测/ABD 状态审计，再判断停止规则是否另有贡献。初始精确 barrier 列表和完整矩阵尚未转储，不把帧末候选数冒充初始计数。详见 [TOI 迭代与第 1 帧复核](TOI_ITERATION_AND_F1_AUDIT_20261005.md)。

2026-10-04。状态：**源码和运行现象已核对；因果假设待验证，尚未实施。** 本轮只记录这一质量支线，不与当前 `B=L^-1` 局部MAS执行候选同时改变算法。恢复更充分的无接触求解可能增加方向次数与耗时，不能预先称为加速优化。

## 已观察到的差异

当前serial TOI本来就存在相对Stiff的形变差异：悬挂100帧最大边长比约1.131726081，Stiff为1.128876204–1.128876222；固定兔子serial为1.040099–1.040358，Stiff为1.035521–1.035715。warp限制不是这些差异的起点；它与serial仍有逐帧分岔，不能据此反向宣称warp严格同质量。

以下来自本轮 `cloth_restrict_hang_toi_serial_r1` 与 `cloth_restrict_hang_stiff_r1` 的 `output/stats.json`。每帧dt=.01；TOI所列帧均在outer=0结束，接受后接触数为0、active_added=0、alpha=1、beta=0，帧退出原因为 `cumulative_toi`。

|物理帧|Stiff方向次数|serial TOI方向次数|TOI末方向最大轴速度 m/s|TOI末方向最大顶点L2长度 m|TOI inner退出|
|---|---:|---:|---:|---:|---|
|1|3|2|0.00506664519|5.29191496e-5|velocity_converged|
|2|3|2|0.01718906954|1.74543019e-4|velocity_converged|
|3|3|2|0.02262420935|2.35676707e-4|velocity_converged|
|18|7|6|0.04965250324|5.31911584e-4|velocity_converged|

悬挂初始bbox对角线为sqrt(2)，IPC movement名义尺度为 `.01 * sqrt(2) * .01 = 1.41421356e-4 m`，对应最大轴方向速度约 `.0141421356 m/s`。TOI绝对速度阈值为 `.05 m/s`。第2/3/18帧的TOI末轴方向超过前一个尺度，但满足TOI速度规则。这是不同停止语义的直接观察，不是证明Stiff会在完全相同中间状态继续同一条轨迹。

## 源码位置及不能混淆的边界

- [TOI方向范数实现](../../sources/stiff_active/StiffGIPC/solver/toi_solver.cu#L275)：`direction_squared_norm`计算顶点L2平方，`direction_axis_max`计算最大坐标绝对值。第1257–1273行分别形成 `direction_norm` 与 `trial_velocity`；后者除以dt。
- [TOI inner退出](../../sources/stiff_active/StiffGIPC/solver/toi_solver.cu#L1554)：速度收敛与满步退出是两条不同路径。第1564行满步条件还要求全帧累计至少6方向及现有safe重启守卫，不能把所有outer改成重新计6次。
- [TOI beta更新](../../sources/stiff_active/StiffGIPC/solver/toi_solver.cu#L1683)：robust路径每outer更新beta。第1709行额外 `movement_contact_free` 判断要求 `k>0`；第1713行随后按剩余量退出。所以所列 `k=0, alpha=1` 帧会在beta=0时返回，没有经过该附加movement门槛。
- [Stiff范数](../../sources/stiff_base/StiffGIPC/core/GIPC.cu#L9936)：`calcMinMovement`最终使用最大坐标绝对值，不是TOI诊断的最大顶点L2。第11010–11032行在本次PCG之前检查现有方向缓冲，并允许 `k && gradVanish` 退出；对照必须同时核对状态时序，不能只替换一个数值阈值。
- [Stiff Kmin边界](../../sources/stiff_base/StiffGIPC/core/GIPC.cu#L10985)：`Kmin=6`；第11110–11122行在达到Kmin后更新beta，并允许累计TOI退出。**Stiff也不是每帧都必须达到movement阈值。** 尤其第18帧不能单凭最后方向超过movement尺度，就断言违反了Stiff全部退出规则。

## 后续唯一有界因果实验

假设：早期无接触阶段的尺度与退出时序差异，可能在首次接触之前就产生可累积的形变/速度状态分岔。先只对“接触集为空、没有新增接触、完整安全CCD接受”的阶段，做默认关闭的Stiff语义配对；复用其真实轴范数、迭代时序与Kmin边界。接触outer、跨帧历史、罚权、乘子、材料、rho、完整CCD和资源上限均保持。不得改成全局 `velocity_only`，不得把本实验与新的MAS执行路径捆绑。

第一步使用同程序从0到3帧，再从0到21帧，重点1–3与16–21；比较Stiff自身重复范围、现有serial/warp TOI与唯一候选的方向数、每方向实际判停量、能量、接受步、布料拉伸及分组位置。没有独立速度导出时不以位置差分代替真实速度验收。若早期差异不能收缩至原冻结范围，就停止该假设，不扩大阈值网格；即使质量改善也不能自动认定性能改善。

通过后才考虑从0推进的固定兔子22–26/40–44、落球20–26/42–49，以及混合兔子24–26/33–35；不使用未经续算等价验证的检查点替代。算法推广仍要求窗口质量不退化且总耗时降低至少10%；若只是质量修复，应明确新增成本，不能借此包装加速。

当前先完成 `B=L^-1` 候选的固定系统数值、含准备成本的局部收益和本机代表场景测试。本假设及后续AutoDL场景均保留为诊断，不代表混合兔子或三种布料已经通过相对Stiff的质量门禁。

证据：[布料结果](CLOTH_RESTRICT_RESULTS.md)、[修正后的完整观测](CLOTH_RESTRICT_QUALITY_TIMING_V2.json)、[执行候选协议](FACTOR_ACTION_PROTOCOL.md)。
