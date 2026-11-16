"""
================================================================================
Stable-Baselines3 实战教程 02b：PPO (Proximal Policy Optimization) 求解 Pendulum-v1
================================================================================

【算法定位】
- 算法名称：PPO (近端策略优化 —— 连续动作空间版)
- 策略类型：On-Policy（同策略）—— 每次使用当前策略并行采样一批轨迹，更新后即清空丢弃
- 学习范式：随机策略 Actor-Critic（Actor 输出连续动作的高斯概率分布，Critic 估计状态价值 V(s)）
- 支持动作空间：本脚本聚焦 **连续动作空间 (Box)**，与 02_ppo_cartpole.py（离散动作空间 Discrete）形成对照

【核心揭秘：同一套 PPO 如何从离散动作切换到连续动作？】
很多初学者好奇：为什么不用改 PPO 的核心损失函数（GAE 优势估计 + Clip 裁剪），
仅仅换个环境，PPO 就能从“左右推车 (Discrete)”变成“连续力矩控制 (Box)”？
秘密全部在于 Actor 策略网络最后一层的 **动作概率分布 (Action Distribution)**：
1. 离散空间 (02_ppo_cartpole.py, Discrete(2))：
   - Actor 输出每个离散动作的 Logits，经 Softmax 构成分类分布 `CategoricalDistribution`。
   - 采样得到整数动作 a ∈ {0, 1}，计算其离散对数概率 log π(a|s)。
2. 连续空间 (02b_ppo_pendulum.py, Box(-2.0, 2.0, (1,)))：
   - 连续区间 [-2.0, 2.0] 内有无穷多个实数，无法穷举概率。
   - SB3 自动将 Actor 切换为 **对角高斯分布 (`DiagGaussianDistribution`)**：
     * 均值网络 μ_θ(s)：由 Actor 神经网络根据当前状态 s 动态输出每个动作维度的均值；
     * 对数标准差 `log_std` (log σ)：默认是一个与状态无关的可学习参数向量（初始值为 0，即标准差 σ = exp(0) = 1.0）。
       训练初期 σ 较大以充分探索，随着训练推进，梯度下降会自动调小 `log_std`，使策略收敛到精准控制！
   - 采样与更新：训练时从高斯分布 N(μ_θ(s), σ^2) 中采样连续动作 a，并用高斯概率密度函数计算对数似然 log π_θ(a|s)。
     只要能算出 log π_θ(a|s) 和策略熵 Entropy，后续的新旧策略概率比 r_t(θ) = exp(log π_new - log π_old)
     以及 Clipped Surrogate Loss 就与离散版本 **完全一模一样**！

【进阶特性：gSDE（广义状态依赖探索，use_sde=True）】
- 标准高斯策略在每个时间步独立采样高斯白噪声，在高频连续控制（如真实机械臂电机控制）中容易导致动作高频抖动。
- 开启 `--use-sde` 后，SB3 将分布切换为 `StateDependentNoiseDistribution` (gSDE)：
  噪声标准差直接依赖于当前状态特征，且在若干步（`sde_sample_freq`）内固定探索噪声矩阵，使探索轨迹更加平滑高效。

【On-Policy (PPO) vs Off-Policy (TD3 / SAC) 连续控制对比】
- TD3 (03) 和 SAC (04) 拥有经验回放池 (ReplayBuffer)，历史数据可反复利用，在 Pendulum-v1 上 20,000 步即可收敛；
- PPO 作为 On-Policy 算法，每次更新后必须丢弃旧数据，样本效率相对较低，因此通常需要更多交互步数（如 100,000 步），
  但得益于 `make_vec_env` 多环境并行采样与极小的单步计算开销，实际训练墙钟时间（Wall-clock time）依然非常快！

运行方式：
    python 02b_ppo_pendulum.py                           # 默认标准对角高斯策略 (100000 步)
    python 02b_ppo_pendulum.py --use-sde                 # 开启 gSDE 广义状态依赖探索
    python 02b_ppo_pendulum.py --timesteps 20000         # 快速测试全流程
================================================================================
"""

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SB3 PPO 求解 Pendulum-v1 连续动作控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="总训练步数（On-Policy 样本效率低于 SAC/TD3，默认 100000 步，4 并行环境下约需 25000 次并行 step）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数量（默认 4）")
    parser.add_argument(
        "--use-sde",
        action="store_true",
        help="是否开启 gSDE（广义状态依赖探索 State-Dependent Exploration）",
    )
    return parser.parse_args()


def format_policy_std(model: PPO) -> str:
    """提取并格式化当前策略网络的高斯动作标准差 σ = exp(log_std)，方便观察探索幅度变化。"""
    with torch.no_grad():
        if model.use_sde:
            # gSDE 模式下 log_std 是状态特征维度的矩阵，使用 action_dist.get_std() 获取标准差矩阵并统计均值与范围
            std_tensor = model.policy.action_dist.get_std(model.policy.log_std)
            return (
                f"均值={std_tensor.mean().item():.4f} "
                f"(范围 [{std_tensor.min().item():.4f}, {std_tensor.max().item():.4f}], shape={tuple(std_tensor.shape)})"
            )
        else:
            # 标准对角高斯模式下 log_std 是形状为 (action_dim,) 的一维可学习参数向量
            log_std = [round(float(x), 4) for x in model.policy.log_std.detach().cpu()]
            std = [round(float(x), 4) for x in torch.exp(model.policy.log_std).detach().cpu()]
            return f"σ = {std} (对应可学习参数 log_std = {log_std})"


def main() -> None:
    args = parse_args()
    env_id = "Pendulum-v1"

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "ppo_pendulum"
    log_dir = base_dir / "logs" / "ppo_pendulum"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 第 1 步：创建矢量化并行连续控制环境与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 {args.n_envs} 个并行的 Gymnasium 连续控制环境: {env_id}")
    train_env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(gym.make(env_id))

    print(f"  并行环境数 (num_envs):        {train_env.num_envs}")
    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")
    print(f"  连续动作维度 (n_actions):     {train_env.action_space.shape[-1]}")
    print(f"  连续力矩范围:                 [{train_env.action_space.low[0]}, {train_env.action_space.high[0]}]")

    # ==========================================================================
    # 第 2 步：模型创建、连续动作概率分布剖析与超参数配置
    # ==========================================================================
    print("\n[第 2 步] 初始化连续动作 PPO 模型与核心超参数...")

    # Actor (pi) 负责输出连续动作高斯分布的均值 μ(s)，Critic (vf) 负责估计状态价值 V(s)
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    n_steps = 1024
    batch_size = 64
    rollout_buffer_size = n_steps * args.n_envs
    print(f"  每次采样迭代 RolloutBuffer 大小 = n_steps({n_steps}) * n_envs({args.n_envs}) = {rollout_buffer_size}")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,

        # --- 针对 Pendulum-v1 调优的学习率与采样批次参数（参考 RL Baselines3 Zoo）---
        learning_rate=1e-3,       # 在 Pendulum 简单连续控制上，1e-3 比默认的 3e-4 收敛快得多
        n_steps=n_steps,          # 每个并行环境采集 1024 步（4 个环境共 4096 步）后触发一次更新
        batch_size=batch_size,    # 小批次大小 64
        n_epochs=10,              # 每次采集的 4096 条数据重复训练 10 轮

        # --- 折扣因子与广义优势估计 (GAE) ---
        gamma=0.9,                # Pendulum 关注近期力矩平衡，较小的折扣因子 γ=0.9 能显著降低价值估计方差、加速收敛
        gae_lambda=0.95,          # GAE 平滑系数 λ

        # --- PPO 核心裁剪与联合 Loss 系数 ---
        clip_range=0.2,           # 新旧策略概率比裁剪区间 [0.8, 1.2]
        ent_coef=0.0,             # 连续高斯分布初期自带较大标准差 (σ=1.0)，无需额外加熵奖励即可充分探索
        vf_coef=0.5,              # Critic 价值函数 Loss 权重
        max_grad_norm=0.5,        # 梯度范数裁剪上限

        # --- 连续动作空间专属：探索机制选择（标准对角高斯 vs gSDE）---
        use_sde=args.use_sde,     # False: 标准 DiagGaussianDistribution | True: 广义状态依赖探索 gSDE
        sde_sample_freq=4 if args.use_sde else -1,  # 开启 gSDE 时，每 4 步重新采样一次状态依赖噪声权重

        # --- 网络结构与日志 ---
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

    # 直观打印底层动作分布类型与初始高斯标准差，对比离散版 CategoricalDistribution
    dist_cls_name = type(model.policy.action_dist).__name__
    print(f"  底层动作分布类型 (action_dist): {dist_cls_name} (use_sde={model.use_sde})")
    print(f"  训练前初始探索标准差:           {format_policy_std(model)}")

    # ==========================================================================
    # 第 3 步：训练前基线评估
    # ==========================================================================
    print("\n[第 3 步] 训练前基线评估...")
    mean_reward_before, std_reward_before = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    # ==========================================================================
    # 第 4 步：配置 EvalCallback 并启动训练
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练连续动作 PPO（总步数 total_timesteps={args.timesteps}）...")
    eval_freq = max(1_000, (args.timesteps // 5) // args.n_envs)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    tb_log_name = "PPO_Pendulum_gSDE" if args.use_sde else "PPO_Pendulum"
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=tb_log_name,
        progress_bar=False,
    )

    # ==========================================================================
    # 第 5 步：训练后定量评估与高斯标准差收敛观察
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励:       {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励:       {mean_reward_after:.2f} +/- {std_reward_after:.2f}")
    print(f"  训练后收敛探索标准差: {format_policy_std(model)}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载
    # ==========================================================================
    final_model_path = model_dir / "ppo_pendulum_final.zip"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    model.save(final_model_path)

    del model
    best_model_path = model_dir / "best_model.zip"
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = PPO.load(load_path, env=eval_env)

    # ==========================================================================
    # 第 7 步：Gymnasium 标准五元组推理闭环演示
    # ==========================================================================
    print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合...")
    demo_env = gym.make(env_id)
    obs, info = demo_env.reset(seed=args.seed)

    episode_reward = 0.0
    step_count = 0

    while True:
        # 在连续动作 PPO 中：
        # - 训练采样时 deterministic=False：从高斯分布 N(μ(s), σ^2) 中随机采样以保证探索；
        # - 评估/部署时 deterministic=True：直接输出高斯均值 μ(s) 并裁剪到 [-2.0, 2.0]，消除随机抖动！
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)

        episode_reward += float(reward)
        step_count += 1

        if terminated or truncated:
            # Pendulum-v1 没有失败终止状态，总是在 200 步时由时间上限触发 truncated=True
            reason = "达到 200 步最大步数截断 (truncated)" if truncated else "任务终止 (terminated)"
            print(f"  推理回合结束 | 总步数: {step_count} | 累计奖励: {episode_reward:.2f} | 结束原因: {reason}")
            break

    train_env.close()
    eval_env.close()
    demo_env.close()
    print("=" * 70)


if __name__ == "__main__":
    main()
