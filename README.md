# HimLoco Lab

## 项目概述

本项目是将 **HimLoco** 从 **IsaacGym** 移植到 **Isaac Lab**。

HimLoco 是一个基于强化学习的四足机器人步态控制器，采用双网络架构设计。通过本项目，您可以在 Isaac Lab 环境中训练、导出和部署 HimLoco 策略。

## 项目结构

```
himlocolab/
├── scripts/                          # 脚本文件
│   ├── himloco_rsl_rl/
│   │   ├── train.py                  # 训练脚本
│   │   ├── play.py                   # 批量推理和策略导出
│   │   ├── play_interactive.py       # 交互式控制脚本
│   │   └── cli_args.py               
│   ├── list_envs.py                  # 列出可用环境
│   ├── zero_agent.py                 
│   └── random_agent.py               
│
├── source/himlocolab/               # 主要源代码
│   └── himlocolab/
│       ├── tasks/                    # 任务定义
│       │   └── locomotion/
│       │       ├── mdp/              # 观测、动作、奖励、终止项
│       │       └── robots/go2/       # 训练 配置
│       │
│       ├── rsl_rl/                   # HimLoco 算法实现
│       │   ├── config/               # 算法配置
│       │   ├── modules/              # HIMActorCritic, HIMEstimator 网络
│       │   ├── algorithms/           # HIMOnPolicyRunner 训练逻辑
│       │   ├── wrappers/             # HimlocoVecEnvWrapper 环境适配 Isaaclab 接口
│       │   └── env/                  # VecEnv 接口
│       │
│       ├── utils/                    # 工具函数
│       │   └── export_policy.py      # JIT 和 ONNX 导出工具
│       │
│       └── assets/                   # 资产文件
│           └── unitree/              # Unitree Go2 URDF 和配置
│
├── deploy/                           # 机器人部署代码
│   └── robots/go2/                   # Go2 控制器实现
│
└── README_CN.md                      # 本文件
```

## 安装指南

### 1. 克隆本仓库

本仓库使用git submodule 维护子仓库isaaclab，故如需使用docker构建训练环境，按照如下克隆本仓库：

```bash
git clone --recursive https://github.com/hyyyyyyz/HIMLocoLab.git
```

### 2. 构建训练环境

#### 推荐使用docker构建训练环境 Isaac Lab 以及相应依赖

进入HIMLocoLab项目根目录下，执行如下命令：

```
sudo chmod +x docker/build.sh
./docker/build.sh
```

如果没有构建过容器，执行该脚本会自动构建容器，如果构建过，会自动进入容器。 如果想删除本地镜像以及容器，执行如下命令：

```
./docker/build.sh -c
```

## 快速开始

### 训练策略

在 Unitree Go2 上训练 HIMLoco 策略：

```bash
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity
```

### 推理和播放

运行已训练的策略进行批量推理：

```bash
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play
```

### 交互式控制

使用键盘在isaaclab中实时控制机器人：

```bash
# 启动交互式控制
python scripts/himloco_rsl_rl/play_interactive.py --task Unitree-Go2-Velocity-Play

# 键盘控制说明:
#   Numpad 8 / ↑   : 前进
#   Numpad 2 / ↓   : 后退
#   Numpad 4 / ←   : 向右平移
#   Numpad 6 / →   : 向左平移
#   Numpad 7 / Z   : 逆时针旋转
#   Numpad 9 / X   : 顺时针旋转
```

### 模型导出

将训练好的模型导出用于部署：

```bash
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play 
```

脚本会在log下生成文件，生成的文件包括：
- `policy.pt` - TorchScript JIT 模型（单文件部署）
- `encoder.onnx` & `policy.onnx` - ONNX 格式（分离模型）

## 部署指南

模型训练完成后，需要在 Mujoco 中对训练好的策略进行模拟验证（Sim2Sim），测试模型性能。
然后才能进行真实机器人部署（Sim2Real）。

### 环境配置

```bash
# 安装依赖库
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# 安装 unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # 安装到 /usr/local 目录
sudo make install
# Compile the robot_controller
cd HIMLocoLab/deploy/robots/go2 
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

安装 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to go2
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.


启动 Mujoco 模拟环境
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

启动控制器：

```bash
cd himloco_lab/deploy/robots/go2/build
./go2_ctrl --network lo # lo 的 ip 地址是 127.0.0.1 表示本地环回通信
```

### Sim2Real

可以使用此程序直接控制真实机器人，但需要确保已关闭机器人上的运控程序。

```bash
./go2_ctrl --network eth0 # eth0 is the network interface name.
```



## 🔗 参考资源

- [HimLoco](https://github.com/RoboLoco/HimLoco)
- [himlocolab](https://github.com/IsaacZH/himlocolab)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)

