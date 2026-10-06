# SRBK 二次型融合与双容差补偿 TOI 实施计划

用户本轮选择一个新执行组件，以及报告中的 compensated 独立入口与专项验证。两个改动分开实现、测试和报告，不把已有公式重新算成新算法，不预设停止条件会提速。

## 冻结和代码范围

参考为上一轮 SHA `9ae1c09394e9f36aed90499dae578064f28921bd96467f6c5a45206cea2a8df0`。旧两个组件保持默认关闭，保留旧源码、结果、460项实验索引及质量协议。默认 IPC legacy/.01/min6/rho1e-4/atomic、材料和完整 CCD 不变。一个活动 overlay，两个子代理分别实现执行组件、只读审查停止时序；公共入口、控制模块、编译和串行GPU由主代理负责。

## 新执行组件：SRBK SpMV＋pᵀAp

独立 `spmv_fused_quadratic=false`。当前 SRBK 对每个存储(i,j)做 Aij*xj→i，非对角再做 Aijᵀ*xi→j，不假设所有块均在上三角。融合按每个存储块的原对称扩展，形成对角 xiᵀAii xi、非对角2 xiᵀAij xj 的贡献。不能读取未完成原子累加的Ap，不在同普通kernel清零并跨block累加。

host/conditional Graph 接入 pAp，PCGSolver 独立拥有 partial/CUB scratch，图签名包含地址/长度/模式；原PCG停序及守卫保留。逐系统requested/effective/fallback记录，禁用路径不增加GPU工作。

受保护固定系统覆盖实际向量、零向量、signed-tail，各host/组件Graph首次/重放。原SpMV的原子结合顺序本来可变，因此记录Ap逐位差异，验收依据在GPU运行前固定的FP64贡献误差界和独立CPU块扩展参考；不得看到失败后扩大容差。主状态/A/b/M/Graph恢复仍要求逐位。此算子误差界不能当材料形变预算。

## 报告 TOI：compensated 独立入口

新增 `preset=toi_compensated`：IPC接触后端、conditional Graph、compensated/.001/.03/min6。可显式改execution=host；不自动开启已有执行组件、稳定MAS或AL罚参数，不改变legacy默认。累计容差与残差容差不同，速度收益单列。

整理纯CPU `IpcResidualController`：第六次有效接受更新前冻结reference，接受最终alpha后在下一次正常装配延后审计；拥有beta/u/z和pending状态，日志与决策离开主求解循环。movement出口继续优先，附加出口要求animation_fullRate>0.99。非法输入即sticky-invalid，含激活前非法alpha/residual；禁止预算失效后通过补偿出口。原final_beta保留，另记真实budget_beta/pending/reference/audits。

严格解析native整数/布尔参数；公开runner拒绝bool冒充整数。专项CPU测试预算恒等式/上下界、零步/全步、历史独立否决、weight=1兼容、固定reference、遗漏/重复接受、非法输入、激活边界、动画门控、movement优先。

## 有限验证与交付

1. 配置契约及新的纯CPU控制模块测试；Release与源码/对象/链接身份验证。
2. 8项既有GPUfixture，两个布料f2固定系统SpMV study。
3. 两布料legacy/spmv/compensated/both，host/Graph，统一dt=.01有限短窗，重型诊断与正式性能分开。短窗材料及PCG结果只作局部证据，不作100帧认证。
4. 选择已有日志中最早补偿激活窗，从零有限运行一次，以验证真实装配→接受→延后审计链；若当前显存保留线阻止，则保留失败并停止，不调min_updates来伪造默认场景覆盖。
5. 旧100帧资源失败条件未消除时不重复长测，不放宽1536MiB保留线或120秒上限，不清理/终止用户进程。无本轮AutoDL/300帧。

交付独立模块说明、调试入口、配置、守卫报告与不可覆盖实验索引。默认不推广；没有完整配对与质量协议证据，就明确速度/质量待验。
