# Stable-Baselines3 + Gymnasium 从 0 实战教程

本项目包含一套完整的 **Stable-Baselines3 (SB3) + Gymnasium** 实战教程代码，涵盖强化学习四大主流核心算法（**DQN、PPO、TD3、SAC**）在离散与连续动作空间下的完整中文注释可运行脚本。

## 目录结构

| 文件 | 说明 | 适用动作空间 | 算法类型 |
| :--- | :--- | :--- | :--- |
| [01_dqn_cartpole.py](./01_dqn_cartpole.py) | DQN 求解 `CartPole-v1`（经验回放池、目标网络、$\epsilon$-greedy 衰减） | 离散 (`Discrete`) | Off-Policy / Value-Based |
| [02_ppo_cartpole.py](./02_ppo_cartpole.py) | PPO 求解 `CartPole-v1`（`make_vec_env` 多环境并行、GAE、Clip 裁剪） | 离散 (`Discrete`) | On-Policy / Actor-Critic |
| [02b_ppo_pendulum.py](./02b_ppo_pendulum.py) | PPO 求解 `Pendulum-v1`（连续对角高斯策略 `DiagGaussian`、可选 `gSDE` 探索） | 连续 (`Box`) | On-Policy / Actor-Critic |
| [03_td3_pendulum.py](./03_td3_pendulum.py) | TD3 求解 `Pendulum-v1`（双 Q 网络、延迟策略更新、高斯动作噪声） | 连续 (`Box`) | Off-Policy / 确定性策略 |
| [04_sac_pendulum.py](./04_sac_pendulum.py) | SAC 求解 `Pendulum-v1`（最大熵学习、重参数化采样、自动熵温度调节） | 连续 (`Box`) | Off-Policy / 最大熵策略 |
| [wandb_utils.py](./wandb_utils.py) | 公共实验追踪与目录管理封装（单次记录/模型自动清理、TensorBoard 同步、W&B 分组与 Artifact） | 通用 | 实验追踪工具 |
| [requirements.txt](./requirements.txt) | 项目依赖列表 | - | 配置 |

---

## 快速开始

### 1. 激活虚拟环境

```bash
cd stable-baseline3
source .venv/bin/activate
```

### 2. 运行四大算法示例

```bash
# 1. DQN（离散控制：倒立摆车）
python 01_dqn_cartpole.py

# 2. PPO（4 环境并行离散控制：倒立摆车）
python 02_ppo_cartpole.py

# 2b. PPO（4 环境并行连续控制：倒立摆力矩控制，支持 --use-sde 开启状态依赖探索）
python 02b_ppo_pendulum.py
python 02b_ppo_pendulum.py --use-sde

# 3. TD3（连续控制：倒立摆力矩控制）
python 03_td3_pendulum.py

# 4. SAC（连续控制：倒立摆力矩控制）
python 04_sac_pendulum.py
```

如需快速跑通测试流程（减少训练步数），可传入 `--timesteps` 参数，例如：

```bash
python 01_dqn_cartpole.py --timesteps 15000
```

### 3. 查看 TensorBoard 训练曲线

```bash
tensorboard --logdir ./logs
```

> **单次记录与模型策略**：默认每次运行脚本时，都会自动清空当前算法变体（`02b` 标准模式对应 `ppo_pendulum`，`--use-sde` 模式独立对应 `ppo_pendulum_gsde`）旧的 `logs/` 与 `models/` 子目录，**每个变体仅保留最新 1 次训练曲线与模型权重**（不会残留 `_1`、`_2`、`_3` 或旧的 `best_model.zip`）。若需在本地临时叠加对比同一算法的多次实验，可附加 `--keep-history`。

### 4. 使用 Weights & Biases (W&B) 追踪实验

所有 5 个脚本均已通过 [wandb_utils.py](./wandb_utils.py) 统一接入 **W&B**，**默认关闭**（不传 `--wandb` 时无需登录或联网）。

1. **首次登录 W&B 云端**（使用离线模式可跳过此步）：

```bash
source .venv/bin/activate
wandb login
```

2. **开启在线追踪**（自动同步全部 TensorBoard 曲线、完整超参数、训练前后评估汇总，并上传最终模型与最优模型作为 Model Artifact）：

```bash
# 对比 CartPole-v1 上的 DQN 与 PPO（自动归入 sb3-tutorial 项目、CartPole-v1 分组）
python 01_dqn_cartpole.py --wandb
python 02_ppo_cartpole.py --wandb

# 对比 Pendulum-v1 上的 PPO / PPO+gSDE / TD3 / SAC（自动归入 Pendulum-v1 分组）
python 02b_ppo_pendulum.py --wandb
python 02b_ppo_pendulum.py --use-sde --wandb
python 03_td3_pendulum.py --wandb
python 04_sac_pendulum.py --wandb
```

3. **离线模式记录与后续补传**：

```bash
python 01_dqn_cartpole.py --timesteps 15000 --wandb --wandb-mode offline
# 联网登录完成后，将本地离线记录补传至云端
wandb sync wandb/offline-run-*
```

4. **常用自定义参数**：
- `--wandb-project <name>`：自定义 W&B 项目名（默认 `sb3-tutorial`）
- `--wandb-entity <name>`：指定 W&B 账号或团队名
- `--wandb-group <name>`：覆盖默认分组名（默认按 `CartPole-v1` / `Pendulum-v1` 分组）
- `--wandb-run-name <name>`：自定义 Run 名称（默认自动按 `算法_环境_seed` 生成）
- `--wandb-tags tag1 tag2`：附加自定义标签（自动与内置的算法/动作空间/gSDE 标签合并）

---

## 学习路径建议

1. 按 `01_dqn_cartpole.py` $\rightarrow$ `02_ppo_cartpole.py` $\rightarrow$ `02b_ppo_pendulum.py` $\rightarrow$ `03_td3_pendulum.py` $\rightarrow$ `04_sac_pendulum.py` 顺序逐行阅读代码注释并运行对比（先离散控制 `CartPole-v1`，再连续控制 `Pendulum-v1`）。
2. 启动 `tensorboard --logdir ./logs`，对比不同算法的训练曲线与样本效率差异（如 On-Policy 的 PPO 与 Off-Policy 的 TD3 / SAC）。
3. 尝试修改 `policy_kwargs` 网络结构、使用 `--use-sde` 对比连续探索行为、接入 `VecNormalize`，或使用 `check_env` 封装自己的 Gymnasium 环境。
