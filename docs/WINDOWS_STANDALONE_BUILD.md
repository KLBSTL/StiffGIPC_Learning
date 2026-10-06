# 本机独立构建交接

2026-10-06，只读核查；未执行 configure/build/GPU。新活动源在仓库根
`StiffGIPC/`，不要调用旧 `tools/active/build.py`：它硬编码旧 overlay 和
`builds/active/inherited-v50/gipc.vcxproj`。

## 已核实的本机配置

| 项 | 路径/既有记录 |
|---|---|
| PowerShell7 | `E:/university_class/Tools/PowerShell/7/pwsh.exe` |
| CMake | `D:/computer/cmake/bin/cmake.exe`，旧缓存版本目录4.0.0-rc2 |
| VS2022 | `D:/vs2022` |
| MSBuild x64 | `D:/vs2022/MSBuild/Current/Bin/amd64/MSBuild.exe` |
| MSVC | `D:/vs2022/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe`；旧缓存19.44.35222 |
| CUDA | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe`；旧缓存13.0.88 |
| vcpkg | `E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake`；x64-windows |
| GPU编译架构 | 旧本机配置 `86`；不要把4090的89复制到本机配置 |

工具文件已检查存在。出处：原实验 `tools/active/build.py:99–108`、
`builds/active/CMakeCache.txt`、`CMakeFiles/4.0.0-rc2/*Compiler.cmake`。
这些是既有缓存身份，新构建需重新采集实际工具身份。

## 命令与先决检查

在 PowerShell7 中，使用一个尚不存在的新构建目录；完整日志保留，失败只显示末尾。
以下 configure 沿用旧项目已使用的工具链选项，源码根改为新仓库：

```powershell
$repo = 'E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning'
$build = "$repo/build/windows-stage2"
$cmake = 'D:/computer/cmake/bin/cmake.exe'
$msbuild = 'D:/vs2022/MSBuild/Current/Bin/amd64/MSBuild.exe'
if (Test-Path -LiteralPath $build) { throw 'Use a fresh build directory; preserve existing evidence.' }
New-Item -ItemType Directory -Path $build | Out-Null
& $cmake -S $repo -B $build -G 'Visual Studio 17 2022' -A x64 `
  '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022' `
  '-DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake' `
  '-DVCPKG_TARGET_TRIPLET=x64-windows' '-DCMAKE_CUDA_ARCHITECTURES=86' `
  *> "$build/configure.log"
if ($LASTEXITCODE -ne 0) { Get-Content "$build/configure.log" -Tail 60; throw 'Configure failed' }
```

**configure 后必须先确认对象名不冲突，不能直接凭 37 TU 数量启动 build。**
源中存在 Windows 大小写冲突：`solver/PCG_SOLVER.cu` 与
`linear_system/solver/pcg_solver.cu`。本次只读检查时，flat CMake 尚未包含旧工程
的每源 `CompileOut` 哈希设置；默认 `$(IntDir)%(Filename).obj` 不能用来承诺
这两个 CUDA 对象独立。由主任务决定修复构建配置，本审查没有改 CMake。

旧项目已用的解决方式（原 `sources/stiff_active/CMakeLists.txt:72–81`）：
对每个实际 TU 的路径 SHA256 取16字符；在 `TARGET_DIRECTORY gipc` 的 source
属性里为 `.cu` 设置 `VS_SETTINGS "CompileOut=$(IntDir)${stem}_${hash}.obj"`，
`.cpp` 设置 `ObjectFileName`。新仓库只需自己的 GIPC_SOURCE 循环，无需恢复
overlay 或冻结源 include。修复后单独重新 configure，确认 XML 与目标配置一致。

通过下面身份检查后再构建：

```powershell
& $msbuild "$build/gipc.vcxproj" /p:Configuration=Release /p:Platform=x64 `
  /m:2 /v:normal /t:Build "/bl:$build/build.binlog" *> "$build/build.log"
if ($LASTEXITCODE -ne 0) { Get-Content "$build/build.log" -Tail 60; throw 'Build failed' }
# 如需构建现有测试目标，另以同配置构建 ALL_BUILD，不在该构建命令中启动测试。
& $cmake --build $build --config Release --target ALL_BUILD --parallel 2 `
  -- /v:normal "/bl:$build/tests-build.binlog" *> "$build/tests-build.log"
if ($LASTEXITCODE -ne 0) { Get-Content "$build/tests-build.log" -Tail 60; throw 'Test targets build failed' }
```

产物为 `$build/Release/gipc.exe`。官方参照单独 `-S "$repo/baseline"`
并选另一个全新 build 目录；同样先检查对象名冲突和 source 根，不能混用活动对象。
不自动清理或复制旧 DLL；记录新 Release 目录中的实际依赖，缺失再由主任务处理。

## 37 TU 与实际链接身份

VS 生成器不能依赖 `CMAKE_EXPORT_COMPILE_COMMANDS` 得到 Linux 风格的完整
compile_commands.json。采用已验证的 MSBuild 证据链：

1. 读取新 `gipc.vcxproj` XML 的 **ItemGroup** `CudaCompile`/`ClCompile Include`
   条目（排除 ItemDefinitionGroup 默认）。解析成绝对路径，与新仓库
   `StiffGIPC/**/*.cu,*.cpp` 的37个文件集合完全相等。不能包含历史源、baseline
   或外部实验源；MeshProcess 是独立库，不能混计进37。
2. 对每条生效 Release 配置求实际 `CompileOut/ObjectFileName`；规范化路径并
   casefold 后必须37个唯一值，且全部位于此 build。仅记录源码 SHA 不够。
3. clean build 的 `/v:normal` 日志保留每条真实 nvcc 调用；C++ 使用
   `CL.command.1.tlog`，记录 compiler、flags、源、object和依赖。保存源与
   include 内容哈希的构建前后清单，期间有变化则本次身份无效。
4. 对每个预期 object 检查存在、大小、SHA；`link.command.1.tlog` 与
   `link.read.1.tlog` 证明所有37个对象实际进入新 gipc，且无旧构建目录对象。
   Device-link 中间对象单独记录，不冒充第38个源 TU。
5. 保留 cache、vcxproj、configure/build日志、binlog、工具身份、link输入、
   executable/DLL SHA。之后未改动增量构建检查应不再重编这37个源。

原 `tools/active/build_support.py:67–134` 有读取 native tlog 和真实 nvcc日志
的方法可参考，但它的 effective_sources.tsv/overlay 前提不适用于新仓库，
不能直接调用并把旧 source map 当新证明。CMake 或构建工具修改由主任务负责。
