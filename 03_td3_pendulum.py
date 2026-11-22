"""
================================================================================
Stable-Baselines3 实战教程 03：TD3 (Twin Delayed DDPG) 求解 Pendulum-v1
================================================================================

【算法定位】
- 算法名称：TD3 (双延迟深度确定性策略梯度)
- 策略类型：Off-Policy（异策略）—— 使用 ReplayBuffer 存储历史连续控制经验
- 学习范式：确定性策略 Actor-Critic（Actor 直接输出一个确定的连续动作值 a = μ(s)，而非概率分布）
- 支持动作空间：**仅支持连续动作空间 (Box)**，如机械臂关节力矩、油门开度、舵机角度等

【从 DDPG 到 TD3 的三大核心改进】
经典的 DDPG 算法在连续控制中经常出现“Q 值高估 (Overestimation Bias)”，导致策略学歪。
TD3 提出了三项极具工程威力的改进（也是名字 Twin Delayed 的由来）：
1. 截断双 Q 网络 (Clipped Double Q-Learning —— "Twin")：
   同时维护两个独立的 Critic 网络 Q1 和 Q2 及其目标网络 Q1' 和 Q2'。
   计算目标价值时取二者的较小值：y = r + γ * min(Q1'(s', a'), Q2'(s', a'))，
   用保守估计有效遏制 Q 值虚高！
2. 延迟策略更新 (Delayed Policy Updates —— "Delayed", policy_delay=2)：
   Critic 每次都会更新，但 Actor 和目标网络每隔 `policy_delay` 次（默认 2 次）才更新 1 次。
   先让 Critic 把 Q 值评估准确了，再去指导 Actor 更新，避免 Actor 被充满噪声的 Q 值带偏。
3. 目标策略平滑 (Target Policy Smoothing)：
   在计算目标动作 a' 时，给目标 Actor 输出的动作加上截断的高斯噪声：
   a' = clip(μ'(s') + clip(ε, -c, c), a_low, a_high)。
   这相当于对 Q 函数在动作维度做局部平滑，防止 Critic 在个别动作点产生尖峰误差被 Actor 钻空子。

【确定性策略如何做探索？】
由于 TD3 的 Actor 是确定性的（同一状态永远输出相同动作），如果不额外加噪声就永远不会去探索新动作！
因此在训练阶段，必须手动构造一个 ActionNoise 对象（例如 NormalActionNoise 高斯噪声）
叠加到动作上：a_exec = μ(s) + noise。评估/推理时 SB3 会自动关闭此噪声。

【实验环境：Pendulum-v1（倒立摆旋转控制）】
- 观测空间: Box(3,) —— [cos(θ), sin(θ), 角速度 θ_dot]
- 动作空间: Box(-2.0, 2.0, (1,), float32) —— 施加在摆杆上的力矩，连续实数 [-2, 2]
- 奖励机制: r = -(θ^2 + 0.1 * θ_dot^2 + 0.001 * torque^2)，每回合固定 200 步。
  奖励恒为负数，理想最优策略是把摆杆甩上去并稳稳倒立在正上方（θ=0），每回合总分通常在 -150 ~ -250 左右；
  随机策略总分通常在 -1200 ~ -1600 左右。

运行方式：
    python 03_td3_pendulum.py                  # 使用默认步数 (20000 步) 完整训练
    python 03_td3_pendulum.py --timesteps 5000 # 快速测试全流程
================================================================================
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SB3 TD3 求解 Pendulum-v1 连续控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="总训练步数（Pendulum-v1 在 20000 步左右即可取得优异表现）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "Pendulum-v1"

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "td3_pendulum"
    log_dir = base_dir / "logs" / "td3_pendulum"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 第 1 步：环境初始化与连续动作空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 连续控制环境: {env_id}")
    train_env = Monitor(gym.make(env_id))
    eval_env = Monitor(gym.make(env_id))

    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")
    print(f"  动作维度 (n_actions):         {train_env.action_space.shape[-1]}")
    print(f"  动作下界 (Low):               {train_env.action_space.low}")
    print(f"  动作上界 (High):              {train_env.action_space.high}")

    # ==========================================================================
    # 第 2 步：配置探索噪声、网络结构与 TD3 模型
    # ==========================================================================
    print("\n[第 2 步] 配置动作探索噪声与 TD3 核心超参数...")

    # 获取连续动作的维度（Pendulum-v1 为 1 维）
    n_actions = train_env.action_space.shape[-1]
    # 构造独立同分布的高斯动作噪声：均值为 0，标准差为 0.1（相对于动作范围 [-2, 2] 来说探索幅度适中）
    # SB3 在 model.learn() 期间会自动把该噪声加到 Actor 输出的动作上（并裁剪到合法动作范围）
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    # 对 Off-Policy Actor-Critic 算法（TD3 / SAC），net_arch 字典中：
    # - pi 代表 Actor（策略网络）的隐藏层结构
    # - qf 代表 Critic（Q 价值网络）的隐藏层结构（注意不是 vf，因为 TD3 学的是 Q(s,a)）
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    model = TD3(
        policy="MlpPolicy",
        env=train_env,

        # --- 学习率、折扣与经验回放池 ---
        learning_rate=1e-3,          # 学习率（Pendulum 常用 1e-3，复杂 MuJoCo 环境常用 3e-4）
        gamma=0.99,                  # 折扣因子 γ
        buffer_size=100_000,         # 经验回放池大小
        learning_starts=1_000,       # 预热步数：前 1000 步纯粹随机采样填充经验池，之后开始梯度更新
        batch_size=100,              # 每次更新采样的批大小（TD3 论文默认 100 或 256）

        # --- 训练频率与软更新 ---
        train_freq=(1, "episode"),   # 每结束 1 个完整回合（或设为 1 表示每 1 步）执行一次更新
        gradient_steps=-1,           # -1 表示本次收集了多少步环境数据，就做多少次梯度更新（与 train_freq=(1,"episode") 配合极佳）
        tau=0.005,                   # 目标网络软更新系数（Polyak Averaging）：θ_target = τ*θ + (1-τ)*θ_target

        # --- TD3 专属核心超参数 ---
        action_noise=action_noise,   # 探索噪声对象（极其重要！不传则确定性策略几乎不探索）
        policy_delay=2,              # 延迟策略更新：Critic 更新 2 次，Actor 与目标网络才更新 1 次
        target_policy_noise=0.2,     # 目标策略平滑噪声标准差：计算目标 Q 值时给目标动作加的高斯噪声
        target_noise_clip=0.5,       # 目标策略平滑噪声裁剪边界 [-0.5, 0.5]

        # --- 网络结构与日志 ---
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

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
    print(f"\n[第 4 步] 开始训练 TD3（总步数 total_timesteps={args.timesteps}）...")
    eval_freq = max(1_000, args.timesteps // 5)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name="TD3_Pendulum",
        progress_bar=False,
    )

    # ==========================================================================
    # 第 5 步：训练后定量评估
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载
    # ==========================================================================
    final_model_path = model_dir / "td3_pendulum_final.zip"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    model.save(final_model_path)

    del model
    best_model_path = model_dir / "best_model.zip"
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = TD3.load(load_path, env=eval_env)

    # ==========================================================================
    # 第 7 步：Gymnasium 标准五元组推理闭环演示
    # ==========================================================================
    print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合...")
    demo_env = gym.make(env_id)
    obs, info = demo_env.reset(seed=args.seed)

    episode_reward = 0.0
    step_count = 0

    while True:
        # 推理时 deterministic=True，直接使用 Actor 输出的确定性连续动作（不加探索噪声）
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
