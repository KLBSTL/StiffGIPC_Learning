# SpMV 与线搜索复查

2026-10-08。用户提供的审查意见逐项核对；附件中的“需确认/可配置警告回退”是审查建议。本轮沿用已授权的纠错范围，保留拒绝非法接受的策略，不加入接受坏步长的开关。

## 结论与逐项处理

| 审查点 | 当前结论 | 本轮处理与证据 |
|---|---|---|
| SpMV 尾 warp 提前 return | 已修复。无效 lane 携零值与独立 head 参与归约，不读取矩阵、不写回向量 | 本轮未再修改算子；既有生产算子测试覆盖171 case、513次Graph重放和三个sanitizer。新构建再次执行同一CTest算子测试 |
| 第9次能量回退后仍不满足接受条件 | 有意改为运行失败，9次预算不变。不会继续接受能量升高或非有限状态 | 保留 helper 的 fail 分支；纯条件测试覆盖预算8/9边界。不是正常路径的停止阈值调整 |
| 异常一路冒泡、是否无兜底 | 批量调用已有 main 的函数级 try/catch，原说法遗漏此兜底；交互回调原先依赖跨FreeGLUT调用栈传播 | 本轮给 display 加局部捕获，两入口共用 controlled_failure，记录失败并以退出码4结束 |
| 失败报告保存失败 | 原 main catch 空吞保存异常；Statistics 的 ofstream 默认也不报告写入错误 | 本轮检查 open/write/close 错误；保存失败向 stderr 报原因，仍以4结束。独立故障fixture检查这一分支 |
| type-1 安全回退后总是重新计算能量 | 正确且必须，最终状态不能继承旧trial的接受结论；可能增加这一异常路径的能量求值与回退 | 保留实现。continue 不重置基准能量或回退计数，下一轮必消耗一次能量半步，预算仍有界 |
| alpha=0/NaN/Inf | 非法推进应失败；正常零方向用alpha=1，不需要零步接受 | 核查ground/self零位移分支与唯一Newton调用；新增实际lineSearch零步长、NaN和非有限基准能量fixture，核对失败JSON与退出码 |
| 半步下溢或无进展 | 已修复，实际下一步必须有限、正且严格小于旧值 | 既有2304项helper测试覆盖subnormal端点及非法CFL，复用CTest |
| armijoParam=0 | 原版既有语义，实际是有限能量非增门 | 保留0。改成正值会改变接受规则，本轮不调整 |
| stopped与isStop死变量 | 返回false且唯一调用者不用返回值 | 删除两个局部死变量，保留bool接口及false返回值，避免无关接口迁移 |
| 新文件未追踪、CMake未入HEAD | 审查时属实，缺少头文件会导致新提交构建失败 | 代码、两个头文件、生产SpMV测试、接受条件测试、故障fixture及根CMake作为一个原子代码提交纳入；报告另提交 |

## 失败策略的含义

当前实现位置：`StiffGIPC/linear_system/utils/spmv.cu:36`（valid lane）、`StiffGIPC/solver/line_search_acceptance.h:19`（纯接受门）、`StiffGIPC/core/GIPC.cu:11114`（失败记录）、`GIPC.cu:11294`（type-1真实能量复检）、`GIPC.cu:11552`（唯一Newton调用）、`StiffGIPC/app/gl_main.cu:1530`（共用失败处理）、`:1617`（display捕获）、`:2035`（main捕获）、`StiffGIPC/gipc/statistics.cpp:7`（检查写入），以及`tests/check_line_search_failure.cmake:1`（原生故障检查）。行号对应本轮最终源文件，不覆盖历史冻结程序。

新策略把原来的非法“成功”改为受控失败：记录reason、当前帧、Newton轮、能量、步长、两种回退次数及是否对应当前trial；关闭接触池trial epoch；通过C++栈展开释放局部CUDA计时事件；程序输出明确错误并返回非零退出码。失败帧不执行后续postLineSearch、速度更新或成功终态导出。它不会回滚后继续跑，也不保证磁盘故障时报告能保存；后者现在有可见错误。

正常零方向：ground核保持value=1，self ACCD在零相对位移时返回1；CFL的maxSpeed=0得到正无穷，alpha仍为1。alpha=0常来自贴地朝地运动、CCD数值停滞或非有限计算。唯一调用者没有零步正常退出分支；零步既不增加有效接受次数，也不减小beta。0×Inf还可能产生NaN，因此不能用零步掩盖非法方向。

type-1路径发生于能量回退后的安全复查，几何状态再改变时必须重建接触并重新计算能量。能量回退计数始终累计，最多9；安全半步严格减小正有限alpha，无法继续推进时失败。该分支正确性已静态推演并用helper检查，但常规真实轨迹尚未触发它，不能宣称完成原生后置安全回退路径的动态覆盖。

## 验证边界与记录

本轮使用新的独立Windows构建与已有GPU串行锁、OwnedJob、120秒测试上限及显存/磁盘保护。构建的实际源/对象/链接/程序身份保存在build manifest；运行收据另存本目录。新CMake故障检查使用string(JSON)，最低CMake版本明确调整为3.19。未修改基线、材料、MAS数学作用、FP64、rho=1e-4、legacy累计阈值.01/min6或完整CCD。

新增实际故障fixture在main初始化网格前调用生产lineSearch入口，以缓存基准能量触发真实接受门；故障在任何网格/ABD访问和trial推进前被拒绝。入口先执行wait_device，因此这些是需要正常CUDA设备的集成测试，GPU异常仍可能被原CUDA_SAFE_CALL直接abort，不属于本次受控线搜索异常的覆盖。fixture只验证入口失败、统计和退出，不把合成输入冒充完整Newton轨迹，也未给生产Newton循环加入注入分支。

初次配置在补齐测试的CMake版本和关闭观测方式时主动停止，仅终止精确识别的本任务配置进程树；未启动GPU，输出保留，不用于程序身份或通过结论。最终版本使用另一空构建目录全量编译。

最终程序SHA256：`0241124167cc9f2f770f4f7b71d2844b71f6878b7be2525e3dde0cc24bb18f29`。代码原子提交`bdedab7`，以下进程监管修复另提交`61d1028`。必要头文件、源文件、测试与CMake均已进入Git树，未混入两个既有未追踪的历史报告目录。

### 当前测试结果

```text
E:/Anaconda/envs/DL/python.exe tools/build_windows.py --kind active --label report56_recheck_final_20261008 --jobs 2
E:/Anaconda/envs/DL/python.exe reports/report5_report6_recheck_20261008/job_observe_test.py
E:/Anaconda/envs/DL/python.exe reports/report5_report6_audit_20261008/verify_fixes.py --build build/report56_recheck_final_20261008 --out runs/report56_recheck_20261008/full_after_runner_fix --stage full
E:/Anaconda/envs/DL/python.exe tools/bench/contracts_test.py
E:/Anaconda/envs/DL/python.exe tools/local/contracts_test.py
E:/Anaconda/envs/DL/python.exe reports/report5_report6_recheck_20261008/smoke_recheck.py
```

- 全量独立构建成功，37个原生编译单元，38个对象记录含额外device-link对象；源/对象/链接映射及pre/post身份检查通过。
- 完整CTest **13/13通过**；新增4项实际程序故障检查均通过：零alpha、NaN alpha、非有限baseline，以及stats.json无法打开写入。前三项得到真实lineSearch失败记录、JSON的failure字段和退出码4；写入故障同时输出保存错误并以4结束。
- 原有接受条件2304项、生产SpMV的171 case/513 Graph重放在新构建再次通过CTest。上轮三个Sanitizer结果对应上轮程序，本轮没有重复运行，不能宣称新程序已重新通过Sanitizer。
- 公共bench合同 **20/20**，本机runner合同 **9/9**；本轮针对进程观测新增的CPU边界 **6/6**通过；diff whitespace检查通过。

首次完整CTest在第3项期间被进程监管中止：子进程可能已退出，QueryFullProcessImageNameW返回WinError5。保存原日志和RESULT.json；该RESULT仍是中途running状态，不用于通过结论。没有求解器失败证据，也没有自动重试模拟。本轮修正OwnedJob.observe：只有同一进程句柄被WaitForSingleObject确认已退出时，保留kernel membership、creation time、退出码和路径查询错误，允许缺少image字段；活进程、未知wait状态或无法确认归属仍失败。六项CPU边界涵盖已退出、活进程错误、未知wait、成功路径、PID复用和进程已消失。修正后使用新结果目录补一轮完整CTest，13项通过；未修改GPU/磁盘/时间保护或按进程名终止进程。

| 当前程序正常路径冒烟 | 帧数 | 方向数 | PCG总量 | 最大能量回退 | 最大相交回退 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 落球L Graph | 30 | 109 | 2323 | 1 | 0 | 完整、有限、配置和数值guard通过 |
| 固定兔子L Graph+四组件 | 60 | 340 | 20582 | 1 | 0 | 同上 |

90帧正常轨迹没有PCG breakdown/上限、Newton上限或line_search_failure。相较上轮同名短测，落球PCG2276→2323；固定兔子方向337→340、PCG20234→20582。未建立这些差异的因果解释，不能据此声称观察改动逐位中性，也不能将差异自动归给某一组件。当前结果仅证明正常场景能完成并满足现有程序guard；未做独立材料标定、全接受路径CPU CCD、轨迹/实际速度等价或配对性能验收。

本轮不测量或宣称新的加速比，不把旧4090性能归给新程序。可交付日志、程序/文件身份、故障JSON及测试范围见[VALIDATION_RECEIPT.json](VALIDATION_RECEIPT.json)。原中止记录在`runs/report56_recheck_20261008/full`，正常运行原始轨迹保留在`runs/report56_recheck_20261008/smoke`，不上传原始轨迹。

## 另外发现但本次未扩展的旧问题

- ground/CCD reduction的部分比较可以掩盖NaN；新能量接受门不能替代完整几何有限性审计。
- 边界动画type-6循环另有直接alpha/=2路径，不调用lineSearch，未获得此helper的下溢保护。
- 尚无原生fixture覆盖第9次拒绝后的真实场景状态及type-1重估分支；当前证据由helper边界、静态控制流和正常场景冒烟组成。

这些缺口单列为待验证，不把本轮“几项审查意见已处理”扩大为整个求解器已无错误。
