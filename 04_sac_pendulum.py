"""
================================================================================
Stable-Baselines3 实战教程 04：SAC (Soft Actor-Critic) 求解 Pendulum-v1
================================================================================

【算法定位】
- 算法名称：SAC (柔性行动者-评论家 / 软演员-评论家)
- 策略类型：Off-Policy（异策略）—— 样本利用率极高，特别适合仿真耗时或真实机器人场景
- 学习范式：最大熵强化学习 (Maximum Entropy RL) + 随机策略 Actor-Critic
- 支持动作空间：**仅支持连续动作空间 (Box)**（连续控制任务的首选基准算法！）

【什么是“最大熵强化学习”？】
传统强化学习只追求最大化累积奖励：
    max Σ E[r(s_t, a_t)]
而 SAC 追求的是“既要拿高分，又要动作足够随机/多样化”：
    max Σ E[r(s_t, a_t) + α * H(π(·|s_t))]
其中：
- H(π(·|s_t)) 是策略分布的信息熵（熵越大代表分布越分散、探索越充分）；
- α（在 SB3 中对应 `ent_coef`）是温度系数，控制“探索（熵）”与“利用（奖励）”的相对权重。

【为什么 SAC 不需要像 TD3 那样手动加 ActionNoise？】
- TD3 的 Actor 是确定性网络，必须外部手动添加 NormalActionNoise 才能探索；
- SAC 的 Actor 是一个**随机策略网络**，它同时输出高斯分布的均值 μ(s) 和标准差 σ(s)，
  并通过重参数化技巧 (Reparameterization Trick) + tanh 压缩采样得到有界连续动作。
  因此 SAC **天生自带自适应探索能力**！不仅如此，当设置 `ent_coef="auto"` 时，
  SB3 还会通过梯度下降自动调节温度系数 α：前期探索不足时调大 α，后期策略成熟时调小 α。

【实验环境：Pendulum-v1（倒立摆旋转控制）】
- 观测空间: Box(3,)
- 动作空间: Box(-2.0, 2.0, (1,), float32)
- 奖励机制: 每回合 200 步，奖励恒为负数，越接近 0 越好（优秀策略约 -150 ~ -250）。

运行方式：
    python 04_sac_pendulum.py                                  # 使用默认步数 (20000 步) 完整训练
    python 04_sac_pendulum.py --timesteps 5000                 # 快速测试全流程
    python 04_sac_pendulum.py --wandb                          # 开启 W&B 在线实验追踪
    python 04_sac_pendulum.py --wandb --wandb-mode offline     # 开启 W&B 离线本地记录
================================================================================
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from wandb_utils import (
    add_wandb_args,
    build_callbacks,
    finish_wandb,
    get_tb_log_name,
    has_updated_best_model,
    init_wandb,
    log_eval_summary,
    log_model_artifact,
    prepare_run_dirs,
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SB3 SAC 求解 Pendulum-v1 连续控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="总训练步数（SAC 样本效率极高，20000 步内通常可达到非常优秀的控制效果）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="保留历史训练日志与旧模型（默认每次开训前清空当前算法目录，仅保留最新一次）",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "Pendulum-v1"

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "sac_pendulum"
    log_dir = base_dir / "logs" / "sac_pendulum"

    train_env = None
    eval_env = None
    demo_env = None
    wandb_run = None

    try:
        # ======================================================================
        # 第 1 步：环境初始化与空间剖析
        # ======================================================================
        print("=" * 70)
        print(f"[第 1 步] 创建 Gymnasium 连续控制环境: {env_id}")
        train_env = Monitor(gym.make(env_id))
        eval_env = Monitor(gym.make(env_id))

        print(f"  观测空间 (Observation Space): {train_env.observation_space}")
        print(f"  动作空间 (Action Space):      {train_env.action_space}")
        print(f"  动作范围:                     [{train_env.action_space.low[0]}, {train_env.action_space.high[0]}]")

        # ======================================================================
        # 第 2 步：模型创建与超参数详解配置（及可选 W&B 初始化）
        # ======================================================================
        print("\n[第 2 步] 初始化 SAC 模型与核心超参数...")

        # 同样支持分别为 Actor (pi) 和 Twin Critic (qf) 指定隐藏层结构
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

        hyperparams = dict(
            # --- 优化器与折扣因子 ---
            learning_rate=3e-4,       # SAC 最经典的默认学习率（对大多数连续控制任务都很稳健）
            gamma=0.99,               # 折扣因子 γ

            # --- 经验回放池 (Replay Buffer) ---
            buffer_size=100_000,      # 经验池容量
            learning_starts=1_000,    # 预热步数：先随机探索 1000 步积累基础样本再开始更新
            batch_size=256,           # SAC 推荐稍大的 batch_size（如 256）以降低梯度方差

            # --- 训练节奏与双 Q 目标网络软更新 ---
            train_freq=1,             # 每与环境交互 1 步，就做一次训练（高频更新带来极高的样本效率）
            gradient_steps=1,         # 每次训练执行 1 步梯度下降
            tau=0.005,                # 软更新系数：目标 Critic 网络以 0.5% 的步幅缓慢追踪主 Critic 网络

            # --- SAC 核心：最大熵温度系数自动调节 ---
            # ent_coef="auto" 是 SAC 最省心的特性！算法会自动优化温度系数 α，
            # 使策略熵维持在目标熵 target_entropy="auto"（默认为 -dim(A)，此处为 -1.0）附近。
            ent_coef="auto",
            target_entropy="auto",
        )

        wandb_run = init_wandb(
            args,
            env_id=env_id,
            algo_name="SAC",
            default_run_name=f"SAC_Pendulum_seed{args.seed}",
            default_tags=["Pendulum-v1", "SAC", "Continuous", "Off-Policy"],
            config={
                "policy": "MlpPolicy",
                "total_timesteps": args.timesteps,
                "seed": args.seed,
                "net_arch": policy_kwargs["net_arch"],
                **hyperparams,
            },
            base_dir=base_dir,
        )
        prepare_run_dirs(model_dir, log_dir, base_dir=base_dir, keep_history=args.keep_history)

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            **hyperparams,
            # 注意：SAC 无需设置 action_noise，因为策略本身就是随机高斯分布（重参数化采样）！
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            verbose=1,
        )

        # ======================================================================
        # 第 3 步：训练前基线评估
        # ======================================================================
        print("\n[第 3 步] 训练前基线评估...")
        mean_reward_before, std_reward_before = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

        # ======================================================================
        # 第 4 步：配置 EvalCallback 并启动训练
        # ======================================================================
        print(f"\n[第 4 步] 开始训练 SAC（总步数 total_timesteps={args.timesteps}）...")
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
        callbacks = build_callbacks(eval_callback, wandb_run)
        tb_log_name = get_tb_log_name("SAC_Pendulum", wandb_run)

        run_start_time = time.time()
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=tb_log_name,
            progress_bar=False,
        )

        # ======================================================================
        # 第 5 步：训练后定量评估
        # ======================================================================
        print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
        mean_reward_after, std_reward_after = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
        print(f"  训练后平均奖励: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

        # ======================================================================
        # 第 6 步：模型保存与重新加载
        # ======================================================================
        final_model_path = model_dir / "sac_pendulum_final.zip"
        print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
        model.save(final_model_path)

        del model
        best_model_path = model_dir / "best_model.zip"
        best_model_updated = has_updated_best_model(eval_callback, best_model_path, run_start_time)
        load_path = best_model_path if best_model_updated else final_model_path
        print(f"  从磁盘重新加载模型: {load_path}")
        loaded_model = SAC.load(load_path, env=eval_env)

        # ======================================================================
        # 第 7 步：Gymnasium 标准五元组推理闭环演示
        # ======================================================================
        print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合...")
        demo_env = gym.make(env_id)
        obs, info = demo_env.reset(seed=args.seed)

        episode_reward = 0.0
        step_count = 0

        while True:
            # 注意 deterministic 参数的含义：
            # - 训练采样时内部为 deterministic=False：从高斯分布中采样随机动作，保证探索；
            # - 评估/上线推理时设 deterministic=True：直接输出 tanh(μ(s)) 均值动作，输出稳定无抖动！
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)

            episode_reward += float(reward)
            step_count += 1

            if terminated or truncated:
                reason = "达到 200 步最大步数截断 (truncated)" if truncated else "任务终止 (terminated)"
                print(f"  推理回合结束 | 总步数: {step_count} | 累计奖励: {episode_reward:.2f} | 结束原因: {reason}")
                break

        log_eval_summary(
            wandb_run,
            {
                "eval/before_mean_reward": float(mean_reward_before),
                "eval/before_std_reward": float(std_reward_before),
                "eval/after_mean_reward": float(mean_reward_after),
                "eval/after_std_reward": float(std_reward_after),
                "eval/reward_gain": float(mean_reward_after - mean_reward_before),
                "eval/best_model_updated": best_model_updated,
                "demo/episode_reward": float(episode_reward),
                "demo/episode_steps": int(step_count),
            },
        )
        log_model_artifact(
            wandb_run,
            artifact_name="sac-pendulum-model",
            final_model_path=final_model_path,
            best_model_path=best_model_path,
            best_model_updated=best_model_updated,
            metadata={
                "algorithm": "SAC",
                "env_id": env_id,
                "seed": args.seed,
                "total_timesteps": args.timesteps,
                "eval_before_mean_reward": float(mean_reward_before),
                "eval_after_mean_reward": float(mean_reward_after),
                "demo_episode_reward": float(episode_reward),
            },
            aliases=["latest", f"seed-{args.seed}"],
        )
    finally:
        if demo_env is not None:
            demo_env.close()
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()
        finish_wandb(wandb_run)
        print("=" * 70)


if __name__ == "__main__":
    main()
