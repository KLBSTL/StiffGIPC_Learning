# 下一轮：swept barrier候选池复用

本轮bounded CCD已完成并停损。下一轮只实现本候选，WHILE分块保持备用；目标继续相对官方Stiff至少2×，不能把一项5%组件收益当最终目标。以下为待执行方案，不是已实现或已通过的结果。

## 依据与预先预算

从零100f当前组合为hang1.730×／fixed1.486×，线性46.8%／47.1%、线搜索23.5%／28.0%。新profile实际hang41／fixed49普通VF＋EE query为22.855／32.644ms，即诊断帧7.32%／12.57%。相同Newton方向在此前已经生成带sqrt(dHat)半径的FullCCD swept候选，可研究避免再次遍历普通树。只知道两个诊断帧，尚无整场节省保证。

以这两帧作为预算估计，净减5%帧成本至少要减少原query成本约68%／40%，另需抵扣pool元数据、bounds守卫、窄相分类、压缩和传输。不能把所有线搜索或所有CCD时间当作可省量。

## 实现边界

1. 一个default-off开关及独立validate开关；仅用于IPC，不继承combined preset。requested／resolved和逐frame／Newton实际使用次数、回退原因同时记录。
2. pool属于一条Newton方向及其swept请求区间，保存generation、origin、direction、temp_alpha、dHat及几何／BodyID／boundary／mapping身份。跨帧、方向、拓扑、映射、属性或半径变化均失效；不继承上一轮接触集。
3. 首先只替换`buildCP`普通VF／EE query。现有普通buildBVH、FullCCD、安全交叉检测`isIntersected`及地面检测保持。通过后才能另审查是否可以减少树维护；本候选不同时做这件事。
4. 在swept query生成时另保存原vertex／face／edge IDs、方向、类型和重复条目。现有int4不足以重建近平行EE的`-obj_idx-2`编码，禁止只按四顶点去重或排序改变方向。
5. 窄相复用原普通query的距离分类和typed编码，输出原DCD＋必要CCD格式、各类型计数及MatIndex类型内置换。每次trial重新过滤／压缩，不复用旧活动接触或旧数值。
6. 缓冲扩容／映射改变后全部写入有效元数据；capacity、written count、epoch和输出poison守卫分开。完整swept候选保留；维护只在开关生效时执行，不让disabled路径支付pool成本。
7. 诊断结果和scratch由模块独立拥有。对照旧query后恢复原生产候选、计数及任何暂借状态；不让诊断改变接受步、Kappa或材料。Graph签名与原缓冲增长守卫继续保留。

## 必须处理的机器包含问题

实数域：same origin／direction且0<=alpha<=temp_alpha时，瞬时primitive AABB属于swept AABB；两条query均用sqrt(dHat)，且body／fixed／shared-vertex／EE原ID过滤相同。

机器域不能无条件推出。ABD用`J(q-alpha*dq)`，swept端点用`x-alpha*(Jdq)`。double反例 `J=[1,-1], q=[1e16,1e16-2], dq=[1,1], alpha=1` 的预测端点为2而实际为4，已CPU核实。它不是本场景已发生的错误，但说明必须有guard。

初版采用每trial实际primitive bounds包含检查：实际位置、alpha／请求区间与缓存swept bounds都有效且包含才使用pool；否则明示回退旧query。优先复用已有bounds，维护成本计入。只加一个经验epsilon或再膨胀一遍barrier半径不是机器证明；无证明的padding不能替代guard。

## 检错与测试顺序

1. CPU配置及epoch／invalid-alpha边界：legacy/min6/.01/rho1e-4不变；disabled不准备pool；旧配置哈希保留。
2. GPUfixture：零／单／尾块、重复primitive、PT／EE所有typed类、近平行mollification、fixed/free组合、动态属性／mapping变化、扩容、超出请求alpha、ABD包含反例和fallback。逐位检查完整typed DCD＋CCD多重集、类型计数、原始ID、方向及MatIndex。
3. 同进程从零hang51／fixed59 guard及混合活动窗口。每trial用私有旧／新输出比较，特别记录接受步后 `isIntersected` 的覆盖。pool与旧query的分项能量、方向／接受alpha核对；FP累加差不能被normalized tuple比较隐藏。
4. 三轮交错off/on 51／59帧，重型诊断关闭。分开pool准备、bounds guard、分类压缩、ordinary query回退、CCD、线搜索、MAS／PCG工作及全前缀／冻结接触窗口时间。若原query下降但总耗时未下降，不追加参数网格。
5. 只有两场景净收益及质量有依据才进入三轮100帧Stiff／当前off／新on。新组件候选最多一次修订。任何正确性失败先停；维护＋查询＋窄相净收益不达5%或全前缀配对中位不达1.05即封存。

GPU始终串行。100f上限120s、300f600s、既有显存预算不变。候选不通过不能延长预算或放宽材料范围；共享本机只报告诊断。现有质量协议及固定基线历史重复失败保留；独立接受路径CPU CCD／真实速度缺口未补齐时不认证同质量。

## 备用方向及停止

若pool没有净收益，先用冻结A/b/M的未profiled事件重放查WHILE边界，不能直接将16.393／33.398ms间隙视为调度收益。长尾含共享GPU、WDDM及profiler，需先反证。若证据成立才单独实现小范围分块，保留previous-rho停止时序、sticky active/error/limit、准确计数和停止后x/r不变，并计入额外尾轮成本。

不同时展开MAS小kernel、SpMV小融合、Tensor Core、低精度、pipelined CG或AL TOI参数。两条主要候选没有明确工作／成本改善后，应提交未达2×与瓶颈结论，不能无限增长版本。下一轮按COMPONENT_REVIEW_PROTOCOL交付完整因果分析、配置、原始报告及索引，再决定后续。
