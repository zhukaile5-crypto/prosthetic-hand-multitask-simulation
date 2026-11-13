# 五指假肢多任务融合控制与仿真

本仓库整理了五指假肢抓握研究中的控制策略、MuJoCo 物理平台、批量数据仿真、交互演示和分析工具。当前主线是一个宏观协调策略：在同一条动作决策链中处理 **抓稳、完成运动、控制受力和放下交接**。

仓库包含代码、模型、运行参数和复现方法，**不包含已经完成的实验结果、曲线、视频或实验日志**。运行示例会在本机生成新的输出。这里的三维画面由刚体接触仿真渲染得到，未接入真实机械臂或假肢硬件。

## 从哪里开始

- 了解整体思路：[架构与项目导航](docs/architecture.md)。
- 查看当前基线与宏观策略：`projects/多任务联合控制研究_2026-11-03/cold_start_take_v1/`、`midterm_core_v1/`。
- 运行可分享版本：使用该项目下新增的 `public_runtime/` 入口。
- 查看原始五指交互画面：`projects/五指假肢抓握仿真/`。
- 查看第三方模型来源与许可：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 当前方案

**基线：力比 PI。** 根据观测到的剪力和法向抓力关系调节抓力，结合滑移反馈维持抓持。这是论文原理在本项目平台上的迁移实现；它并不是“无限持续加力”的对照，也不等同于原论文装置与数值结果的精确复现。

**宏观策略：共享观测与执行条件下的任务协调。** 持握时共同考虑抓力建议、支撑需求、运动目标和力预算；接近接收面时根据剩余间隙减速；接触接收面后，把卸力、低运动和撤离组织为同一个交接过程。运行期间由统一执行入口提交动作。

当前全程控制入口从拿取阶段就分别使用基线与宏观策略，随后完成运输和放下。两种方法共享场景和随机种子，便于配对比较。原先从共同准备状态接管的版本仍保留在源码中，不能与全程控制实验混为一组。

## 安装

Windows 用户建议将仓库解压或克隆到较短的路径（例如 `D:\projects\hand-sim`），历史研究源码中部分文件名较长。

建议使用独立 Python 虚拟环境。在仓库根目录执行：

```powershell
# 确认版本为 Python 3.12
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

建议使用 **Python 3.12**；项目原环境为 Python 3.12.14。根 `requirements.txt` 固定了该环境使用的主要依赖版本。部分历史审查及 V22/V23 求解器还使用 SciPy，详见 `requirements-review.txt`。V22/V23 历史入口另绑定原环境位置与二进制摘要，安装 SciPy 本身不能恢复这些入口。

Linux/macOS 的 Bash 或 Zsh 命令与 PowerShell 不同，请参考下方独立示例；目前完整安装及运行检查仅在 Windows 执行。渲染需要可用的 OpenGL 图形环境；首次安装建议先执行下面的零步检查，再运行一组小实验。中文路径和目录同级关系属于现有模块导入约定，请整体保留 `projects/` 结构。

## 运行当前基线与宏观版

以下命令在仓库根目录的 PowerShell 中执行。先保存仓库内 Python 的完整路径，再进入项目目录：

```powershell
$taskPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location .\projects\多任务联合控制研究_2026-11-03

# 构造两种策略和物理模型，不推进仿真时间
& $taskPython -B -m public_runtime.verify

# 一个正常运输场景、一个随机种子，两种策略同条件运行
& $taskPython -B -m public_runtime.run --out output/demo_nominal --scenes nominal --seeds 12345 --workers 1

# 四场景、两个随机种子的配对运行；会启动新的物理仿真
& $taskPython -B -m public_runtime.run --out output/demo_four_scenes --scenes all --seeds 12345 23456 --workers 2
```

本发布版已通过两种策略的零步构造检查及命令行参数检查；原型平台已通过单帧渲染与 HTTP 服务检查。整理发布包时未重新运行完整抓握试验，未重新生成算法性能结果。

输出目录必须使用新名称。批量运行时间取决于处理器、线程设置和场景复杂度；首次运行建议使用单场景、单进程。

| 场景标识 | 场景含义 |
|---|---|
| `nominal` | 正常拿取、运输和放下 |
| `friction_drop` | 运输途中摩擦条件下降 |
| `fragile_conflict` | 更严格的力限制与持握需求发生冲突 |
| `collision` | 运输过程中出现碰撞扰动 |

随机种子用于保持配对物理条件和观测噪声。新增种子产生的是新实验，不能自动继承此前版本的验证结论。

### Linux/macOS 命令示例

以下使用 Bash 或 Zsh，从仓库根目录执行；该示例尚未在 Linux/macOS 上实测。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
TASK_PYTHON="$(pwd)/.venv/bin/python"
cd projects/多任务联合控制研究_2026-11-03
"$TASK_PYTHON" -B -m public_runtime.verify
"$TASK_PYTHON" -B -m public_runtime.run --out output/demo_nominal --scenes nominal --seeds 12345 --workers 1
cd ../..
.venv/bin/python scripts/run_five_finger.py --check
```

## 交互画面与回放

原始五指平台提供浏览器交互页面，由 Python HTTP 服务驱动 MuJoCo 渲染。它适合检查手部模型、接触和基础抓握流程；该演示平台有自己的原型控制逻辑，不能把其画面自动解释为当前宏观策略的行为。

运行前请回到仓库根目录（如果刚在上面的项目子目录运行过命令，执行 `Set-Location ../..`），然后执行：

```powershell
# 五指原型；在前台运行，浏览器显示本机交互页面
.\.venv\Scripts\python.exe scripts/run_five_finger.py

# 双夹爪原型
.\.venv\Scripts\python.exe scripts/run_two_finger.py

# 只加载五指模型并渲染一帧，随后退出
.\.venv\Scripts\python.exe scripts/run_five_finger.py --check
```

启动器支持 `--no-browser` 和 `--port 8888`；服务运行时使用 `Ctrl+C` 停止。上述命令展示原型平台，当前策略的批量入口见上一节。

完整策略接入版提供观测、执行、策略适配、可视化和记录回放的源码。`tools/archived_visualization/` 保留双策略同步视频和总指力曲线的制作方法；这些历史脚本引用未打包的原记录，使用时需改为新记录路径并核对格式。当前主线的渲染与分析工具也随包保留；其输入应是新生成且格式匹配的记录。历史脚本中固定引用原实验路径的入口，需按[架构说明](docs/architecture.md#历史入口的运行边界)处理。

## 仓库分类

| 目录 | 定位 |
|---|---|
| `projects/多任务联合控制研究_2026-11-03/` | 当前研究主线、宏观版、全程控制、转腕与接触任务扩展、各阶段算法源码 |
| `projects/五指策略接入_2026-11-03/` | 完整五指平台的观测、执行、校准、策略迁移、服务器与回放 |
| `projects/统一人工反射重构/` | 五个柔顺触点与自由物体的简化物理基准、统一协调器及评价代码 |
| `projects/多任务融合对照实验/` | 早期基线、融合控制和 Shadow 五指物理对照 |
| `projects/多任务融合第二轮/` | 第二轮协调策略、消融与分析方法 |
| `projects/五指假肢抓握仿真/` | 原始 Shadow 五指可视化原型和模型资产 |
| `projects/抓握交互仿真/` | 双夹爪交互仿真原型 |
| `projects/仿真调研_2026-10-08/local_probe/` | MuJoCo 与 Allegro 环境探索探针 |
| `tools/archived_visualization/` | 双策略视频、总指力及阶段曲线制作的历史脚本，使用前需适配输入记录 |
| `docs/` | 架构、运行边界及发布说明 |

旧版本用于理解设计演进和复用算法，不代表每个历史入口在移除结果后都可直接运行。目录中的版本号不表示多个独立开源产品。

## 如何比较策略

使用相同的物理场景、初始条件、种子、传感条件和执行权限，分别记录两种方法。评价同时包含：

- **任务完成**：拿取、运输与放下是否完成，是否发生失抓或掉落。
- **受力品质**：总指力和逐指力变化、波动、峰值及接触瞬间首峰。
- **时间代价**：稳定等待、扰动恢复、接触交接和清离时间。
- **力限制**：逐指或总力超限的幅值、持续时间及积分。
- **任务专项指标**：转腕对准误差，或轻压任务的超调与稳态误差。

总指力曲线适合直观比较；逐指指标用于检查载荷分配。失败后失去接触产生的低力不能视为平稳性收益。不同历史版本的控制时间范围、滤波口径和阶段边界可能不同，比较时以相应代码与协议为准。

本包不附性能排名或显著性结论。多个场景的一次运行可以展示现象；稳健性结论需要固定策略、独立种子重复试验及配对统计。

## 安装与运行验证

在 Windows 11、Python 3.12.14 的独立虚拟环境中，已从源码压缩包安装 `requirements.txt`，并运行正常运输场景的基线和宏观策略各一次：两次运行均完成并生成记录，配对条件检查通过。五指平台已检查浏览器抓握与暂停；双夹爪原型已检查渲染和控制接口。测试使用本机已有的 Python 解释器，并非在另一台全新电脑上完成；Linux/macOS 尚未实测。上述检查证明示例可运行，不代表性能结论或独立物理验收通过。

## 许可与贡献

自行编写的程序和文档采用 [MIT License](LICENSE)。Shadow Hand 模型与派生模型保留 Apache-2.0 许可，Allegro 模型保留 BSD-2-Clause 许可；详见 [第三方来源与许可证](THIRD_PARTY_NOTICES.md)。

贡献代码时，请保持原实验输出独立，在新目录生成数据，并在变更中写明修改的控制层、运行条件和检查方式。共享补充结果时，保留场景、种子、参数、源码版本和指标定义，避免只上传筛选后的图。

