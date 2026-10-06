# 安全 edge–triangle 查询顺序候选：实施与检错边界

2026-10-06。候选已在24次BVH消融完成后应用，原raw函数文本保持一致。
代码进入独立fresh build；实际构建、GPU验证和保留决定见
[本轮主报告](ROUND_REVIEW.md)。24次旧结果及原程序身份保留。

## 唯一假设和接口

假设：把查询面从原始face ID顺序改为**当前普通面BVH的叶顺序**，使同warp查询的面更接近，改善对同一完整边BVH的遍历。没有新增空间排序、面过滤、候选池或子树资格过滤。它不是已有负收益eligibility分支，也不改变CCD/材料/停止条件。

- `GIPC_EDGE_QUERY_ORDER=raw|leaf`，缺省`raw`，不继承加速suite；其他值启动报错。
- `GIPC_EDGE_ORDER_PROBE_FRAMES`可选，严格为`1,41,57`的无重复子集；不能含空白、其他帧或尾逗号。
- `GIPC_EDGE_ORDER_PROBE_FILE`须与非空帧列表同时给出。第一次写入发现文件已存在就拒绝覆盖。
- leaf或probe仅允许IPC；TOI请求直接报错。生产mode不会因probe开启而改变。
- resolved新增`edge_query_order`独立对象，记录requested/effective/default/scope、probe帧/路径/7对次数和边界范围。公共runner独立key、环境映射和validator已整合；旧默认配置继续可读，旧收据不回写。

补丁新增`collision/edge_query_order.h/.inl`，修改`core/GIPC.cu`及`solver/toi_options.h`；夹具增量另加`app/gl_main.cu`早退出入口和一个CMake测试声明，不改已有测试源码。原 `_edgeTriIntersectionQuery` 和原host wrapper均保留；默认raw、无probe、至少两条边继续调用原GPU kernel。新增模板副本仅修改face来源并在诊断模板中写私有结果。大场景边遍历栈、左右子树顺序、gap=0、`_overlap`、共享顶点排除、全部顶点`btype>=2`排除及`segTriIntersect`参数/算式不变。

leaf使用`bvh_f._nodes[idx+face_number-1].element_idx`指向原face；先检查当前两树ordinary mode、没有暴露swept storage、节点/边界数量一致。它依赖现有BVH构建的排列合同，不读取旧扫掠缓存。生产不额外排序、下载排列或修改树；探针CPU独立证明选定状态叶ID恰为0…F−1。所有探针GPU输入只读；新输出和事件为函数内私有资源，不覆盖`_scalar_scratch`、接触对、原树或求解状态。

## 有限同状态探针

每个指定**一基物理帧**第一次真正edge-query调用触发一次，`total_Frames+1`，不重新推进物理状态：

1. CPU读回当前face leaves，排序其original IDs，仅用于检查完整排列；不是新的生产空间顺序。
2. 用原raw生产kernel得到私有any-hit真值。raw/leaf两个诊断模板分别输出any-hit以及按原face ID索引的per-face布尔结果，预先poison为−1，要求每个槽均写成0/1、逐面完全一致、any-hit与原raw一致。
3. 对两种**无per-face写入的生产kernel**各预热两次；同一几何、同一树上固定7对，先后顺序raw/leaf、leaf/raw交替。原raw kernel作为raw timing，不用诊断副本冒充生产成本。
4. CUDA event只包住kernel；清零在start之前，host结果读回在end同步之后。记录每对`raw_kernel_ms`/`leaf_kernel_ms`、单独`host_readback_ms`、全部`diagnostic_host_ms`，每次timed any-hit也必须与真值一致。
5. JSONL逐帧保存通过/失败及实际尺寸、比较面数、相交面数、未写槽/差异计数、生产mode。探针C++异常写失败证据后抛出；CUDA致命错误可能直接结束进程，缺失记录必须由runner判失败，不能当通过。

该探针不是冻结完整求解器对照，不认证轨迹或整场提速。需要至少有非平凡排列、非空face/edge、真实contact场景；真实安全状态很可能any-hit都false，不能把这些case说成覆盖相交true。

## 0/1边和缺口

旧kernel会直接读取root左右子节点，不适用于单叶边树。补丁对IPC明确：F=0或E=0返回无相交；E=1以唯一edge0执行同AABB/shared/fixed/segment-triangle谓词，raw/leaf共用该边界实现，不读取root孩子。这是显式定义旧未覆盖边界，不把它写成原非法访问的逐位等价，也不把0/1结果作为排序性能证据。新非空leaf/诊断路径若不能证明当前普通树存储，就抛错，不悄悄改用其他树。

以下为实现时冻结的验收清单；当前已完成解析/构建、12合成case和两场景
私有probe，实际结果见主报告。生产leaf长轨迹与完整质量尚未验收：

- CPU检查strict解析、resolved与实际请求，原raw函数文本不变；fresh clean build和新manifest。已编译并通过相关CPU/CTest，构建身份另行封存。
- GPU小fixture：F/E=0/1、257尾块、非平凡face排列、包含intersect true/false、共享顶点和全固定/部分固定、普通/swept暴露错误应拒绝、非法叶排列probe应拒绝。边界测试单列，不与大场景旧path算速度。
- 同进程选帧probe全部逐面/any-hit通过；7对局部结果只作为继续与否的诊断。若未覆盖true，补一个有交叉的合成fixture，不能拿全false覆盖代替。
- 然后才关闭probe，原同轮对照验证轨迹、actual velocity、固定点、布料/ABD/FEM质量和总工作量；保持原120秒/显存门禁及材料/停止阈值。只有整段收益达到主计划门槛才保留；不得以局部edge kernel提速宣称2×。

当前f57 edge-triangle成本约9.5–9.8ms，仅约单帧墙钟6%；这说明即使本核明显变快，总收益仍受上界限制。默认关闭不变，失败只留证据不追加另一种排序。

## 合成夹具增量（12/12通过）

入口`gipc::edge_query_order_fixture(const char*)`，环境`GIPC_EDGE_ORDER_FIXTURE=<新输出.json>`，在场景/OpenGL初始化之前早退出。要求`GIPC_DISCRETE_BVH_REFIT=0`。新增CTest `edge_query_order_equivalence`，RUN_SERIAL、TIMEOUT=120；场景同状态探针仍最多由主任务安排三批，不增加图计时或参数网格。

固定12项：0/0、0面/3边、3面/0边；1面1边命中/非命中；共享顶点；全部btype2；全部btype3；btype3中一顶点1；负btype；257面交错命中/非命中及反序叶排列；一次真实D0 `ConstructRebuild`普通树。每个非空case比较原raw或共同单边路径与leaf的实际any-hit，并逐面核对诊断副本与解析预期，检查poison全部被写、叶排列完整和暴露swept storage时拒绝查询。

共享顶点case专门使用斜面，使严格AABB重叠和未过滤segTri谓词都成立，避免在更早的AABB判定已经拒绝而虚假宣称验证shared mask。单线程诊断核返回overlap/shared/fixed/segTri四位mask，对照解析预期；all-fixed与partial-fixed也具有实际几何命中。夹具不把手工平衡树当生产排序，只为固定有限输入。最后一次使用真正D0 builder：`discrete_construct`关闭分支进入`ConstructRebuild`，face `mlbvh.cu:2463`和edge`:2579`无条件`discrete_note_rebuild`，因此普通mode检查在D0实际成立。原默认raw>=2绕过新的tree限制。

## 静态检查结果

应用前`git apply --check`通过。源文件部分CRLF混合，应用保留原换行；独立
审查确认原raw kernel和host wrapper不变。修改native后的程序必须重新构建
并seal；旧源码可由Git检查点恢复，旧数据不回写。提交代码与紧凑报告，
生成补丁仅留本机runs归档，避免主树重复保存一份完整实现。
