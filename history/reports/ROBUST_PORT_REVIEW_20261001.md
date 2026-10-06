# v32 独立审阅闭环

规格审阅 `port_spec_review`、代码审阅 `port_code_review` 均为只读；实现、GPU 测试与构建由主任务串行完成。供体验证 `robust_reference` 在独立目录操作。

| 发现 | 修复与证据 |
|---|---|
| 过期旧身份阻止新候选重新加入 | 只有年龄仍保留的旧身份进入 existing；26 次释放删除、重激活/新候选年龄 0。 |
| 奇异夹具误把错误编号当作 0 | GPU linearize 用 i+1 编码；夹具断言 1，受控失败验证通过。 |
| blocker 审计重算并替换 alpha | 直接读取同批完整 CCD 数据；shared 模式固定安全 ratio 0.2。 |
| 摩擦快照几何/重复 solve 刷新 | 使用 mesh.o_vertexes 与 physical frame cache；实际 IDs/force/UV/basis 字节审计 824 次通过。 |
| contact-free Hessian 重用忽略非空摩擦 | 重用须同时为空 normal 和 friction 集合。 |
| 快照夹具只检查 λ/UV | 补充 PT/EE 下一帧旋转/坐标更新、切向正交基、原始 λ 不乘衰减 gamma、地面力。 |
| cache 复用未验证记录身份/参数 | 比较源、EXE、runner SHA 和完整固定协议；矩阵保存身份。 |
| 新头字段默认初始化没进入旧对象 | 旧 EXE/矩阵归档；clean-first；每个 100 帧 robust 记录均检查真实捕获和帧号。 |
| μ 范数包含固定 ABD/FEM | 供体源码确认排除；加入 movable_diagonal。实际 GPU 第 7 夹具 max=4；真实组装 profile 固定 ABD norm=0，μ=.1*freeMax。 |
| timer/报告范围可能混淆 | 明示 host steady_clock 包围 IPC_Solver+device同步；独立 CCD 只覆盖四组 60 帧，100 帧只声明程序内安全检查。 |

最终验证范围：fixed sphere + free cloth、motion_rate=1、RTX 3070 Laptop，本机 CUDA 13.0/sm86。固定/自由 DOF 布局与生产指针经独立源码审阅；未拓展为所有运动 ABD 配置的验证。多个 solve/同帧的原统计文件编号覆盖仍是已知限制，runner 捕获协议失败，不给该模式完整性资格。

初始 full-scope 分支和未完整更新初始化的记录均保存，最终报告只用 `robust_v32_matched_*`。供体原 CUDA EE 3/9 失败、修正版 9/9 通过；主组件 7/7 与原 AL/warm/volume 通过；最终 18 组各 100 帧完成。物理质量仍未过门槛，未宣称质量匹配的加速。
