# 第三方来源与许可证

本仓库自行编写的策略、仿真程序、运行工具与文档采用根目录的 [MIT License](LICENSE)，版权标识为 `Copyright (c) 2026 project contributors`。第三方模型、网格及其派生模型保留各自的许可证与归属；根目录的 MIT 许可证不替代这些条款。文件或子目录存在独立许可声明时，应同时查看相应声明。

## Shadow Hand 模型与网格

- **来源**：Google DeepMind MuJoCo Menagerie 的 [Shadow Hand E3M5](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/shadow_hand)。原始机器人描述与资产由 Shadow Robot Company 提供；上游 README 记录了 URDF 到 MJCF 的转换过程。
- **固定提交**：`0059d4335f8156206f63a35662313385f7ad6d74`。
- **许可证**：Apache License 2.0。
- **原有版权声明**：`Copyright 2022 Shadow Robot Company Ltd`。
- **仓库内位置**：
  - `projects/五指假肢抓握仿真/upstream_shadow/`
  - `projects/五指策略接入_2026-11-03/assets/shadow/`

上述目录中的 `LICENSE`、`README.md`、`CHANGELOG.md` 和 `download_manifest.json` 随模型保留。下载清单记录了上游提交、文件来源和 SHA256，可用于核对模型及网格。此次固定版本的资产目录没有独立的上游 `NOTICE` 文件；本说明不冒充上游 NOTICE。

### 派生场景

以下场景沿用了 Shadow Hand 的 MJCF 结构及模型资产，按 Apache-2.0 保留其来源和许可边界：

- `projects/五指假肢抓握仿真/model.xml`
- `projects/五指策略接入_2026-11-03/assets/legacy_model.xml`
- 由这些模型和项目场景构建程序生成、包含相同上游结构或资产的派生模型。

项目修改包括运输升降关节与执行器、指垫材料和接触设置、目标物、支撑面与接收场景等。修改的来源可参见 `projects/五指假肢抓握仿真/build_scene.py` 及接入平台中的场景构建代码。发布副本中的两份派生 XML 已附加修改声明；再次修改、生成和分发派生模型时应继续保留上游许可、版权和归属信息，并标明文件修改。

独立编写的控制算法和场景构建程序采用本仓库 MIT 许可证；这不将它们引用或生成的 Shadow 模型资产重新授权为 MIT。

## Allegro Hand 探索模型

- **来源**：Google DeepMind MuJoCo Menagerie 的 [Wonik Allegro Hand V3](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/wonik_allegro)，上游说明其模型由公开的 Allegro Hand ROS 机器人描述派生。
- **固定提交**：`0059d4335f8156206f63a35662313385f7ad6d74`。
- **许可证**：BSD 2-Clause。
- **原有版权声明**：`Copyright (c) 2016, SimLab. All rights reserved.`
- **仓库内位置**：`projects/仿真调研_2026-10-08/local_probe/menagerie_allegro/`。

该目录属于早期仿真路线探索，包含独立的 `LICENSE`、`README.md`、`CHANGELOG.md` 与 `download_manifest.json`。它使用四指 Allegro 模型，与当前五指 Shadow 平台区分。源码分发保留 BSD 版权、条件与免责声明；二进制分发按其附带许可证保留对应说明。

## 冻结的项目内部代码

`projects/五指策略接入_2026-11-03/vendor/` 中的 `frozen_controllers.py`、`frozen_evaluation.py` 等是项目内部实现的冻结副本，目录名 `vendor` 不代表这些文件来自 MuJoCo Menagerie。项目内部代码适用根 MIT 许可证；相关模型资产仍按上文的独立许可证处理。

## 运行依赖和其他材料

MuJoCo、NumPy 等运行依赖通过安装命令取得，各自的许可证由其发行包提供。根 MIT 许可证不替代这些依赖以及操作系统字体的许可。本源码包不包含 Python 虚拟环境、操作系统字体、已有实验输出或第三方论文全文；论文与软件的来源链接用于说明方法和依赖来源。
