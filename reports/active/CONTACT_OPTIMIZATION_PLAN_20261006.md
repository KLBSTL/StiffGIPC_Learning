# 接触布料：成本分析、有限优化和2×目标

目标仍以从零100帧、dt=.01、相同材料和停止规则的官方Stiff为分母，至少2×为目标；共享本机测量只是诊断，完整质量和受控性能分别验收。旧质量协议保持不变。

首场景cloth_hang_l，从零至少51帧覆盖f41–50完整接触和f51脱离；完整速度100帧。第二场景cloth_fixed_bunny_l，接触f22–100持续，短窗至少59帧，特别检查f45–59。日志geometric_contact比IPC barrier半径更严格，不能用它排除有接触；同时记录native pairs和Newton active_pairs。固定接触窗口在候选运行前冻结，不随结果移动。

先冻结并串行运行4组100帧基线：两场景Stiff/current组合。首次只有一组，不能单次认证加速或质量。保存每阶段及未归属耗时，全程与接触窗口分开；有资源失败保留，不扩大显存/时间预算。当前组合的BVH子树过滤关闭。

复用历史24组100f和既有Nsight给出潜力判断。线性准备+PCG约占46–49%，是最大总成本；CCD约9–11%，不能仅靠小block专化追求2×。不重开已否定的全query剪枝、D2D Async、Graph初始化、MAS输出或SpMV小融合网格。

候选最多一项并最多修订一次。候选须有明确可消除工作，先同状态正确性，再三轮交错off/on接触短窗。达到窗口净提速5%且无硬错误后才进入3轮100帧Stiff/off/on；物理、轨迹和速度独立报告。没有达到门槛即封存，不临时改停止条件。候选选择必须在本轮成本/源语义分析后冻结，禁止一边计时一边改native。

拟核查的算法等价执行路径是“请求时间上界内CCD”：只对已有temp_alpha截断的第二次swept CCD，研究能否在原ACCD已经证明超过该区间后停止。所有candidate、原距离计算、完整安全检测、原收敛规则保留；不能先假定合法，必须检查返回时序、finite/单调条件和逐pair截断值。若证明或收益不足，停止该路径，回到线性阶段工作量证据。

实施选择冻结：bounded_ccd/ bounded_ccd_validate 独立默认false，仅第二次IPC swept query、0<temp_alpha<1、ccd_size=1生效。原gap退出、toc累加和toc>1返回顺序保持，有限非负单调前缀还需实际倒数往返值达到请求上界后才可截断。超出请求区间的原程序浮点行为不作无条件等价保证；以原设备函数为真值，逐pair clipped、完整倒数/Max/倒数归约、生产结果及CFL后最终alpha逐位核对。诊断计数副本与原函数交叉检查，独立scratch，失败即停止。

当前成本证据只支持可消除工作，尚不支持5%收益。因而安排一次有界试验，不将此小分支视作2×主解法。fixture后运行悬挂51/固定59/混合3帧守卫，三轮交错screen(off/on)共12次。完整性能gate预先固定为两场景各3对、全前缀配对中位>=1.05、无硬错误；接触窗口比值另列，未过gate不扩到新100/300帧。停止条件仍legacy/.01/min6/rho1e-4，材料和碰撞对不裁剪。

线性热点补测独立进行：当前原组合 bounded_ccd=false，Nsight node模式从零推进到hang42/fixed50，只捕获hang41/fixed49，cost_events=false关闭观测事件/flush同步。新增MAS numeric_fill/numeric_aggregate/legacy_factor scope仅分开已有准备调用；按实际Graph节点符号补充restrict/local/prolong/spmv/vector/reduce分类，未知保留，不把图内缺scope解释为零成本。fixed49本轮线性153.052ms、8方向/659PCG，高于fixed57的110.970ms、6方向/461PCG，且处于接触峰窗45–59；不改变窗口来寻找收益。

每轮均按COMPONENT_REVIEW_PROTOCOL收尾：程序身份、真实开关、同状态误差、准备成本、PCG与方向量、配对时间、材料/实际速度、覆盖缺口、下一唯一方向。原失败、594项旧索引和旧决定不清理、不追溯改判。本轮不安排AutoDL、清理或默认推广。
