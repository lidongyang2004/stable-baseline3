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

---

## 学习路径建议

1. 按 `01_dqn_cartpole.py` $\rightarrow$ `02_ppo_cartpole.py` $\rightarrow$ `02b_ppo_pendulum.py` $\rightarrow$ `03_td3_pendulum.py` $\rightarrow$ `04_sac_pendulum.py` 顺序逐行阅读代码注释并运行对比（先离散控制 `CartPole-v1`，再连续控制 `Pendulum-v1`）。
2. 启动 `tensorboard --logdir ./logs`，对比不同算法的训练曲线与样本效率差异（如 On-Policy 的 PPO 与 Off-Policy 的 TD3 / SAC）。
3. 尝试修改 `policy_kwargs` 网络结构、使用 `--use-sde` 对比连续探索行为、接入 `VecNormalize`，或使用 `check_env` 封装自己的 Gymnasium 环境。
