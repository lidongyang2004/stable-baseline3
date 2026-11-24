"""
================================================================================
Stable-Baselines3 实战教程 02：PPO (Proximal Policy Optimization) 求解 CartPole-v1
================================================================================

【算法定位】
- 算法名称：PPO (近端策略优化)
- 策略类型：On-Policy（同策略）—— 只能用“当前最新策略”采样的数据进行更新，更新完后数据即丢弃
- 学习范式：Actor-Critic（行动者-评论家）
  - Actor (Policy Network π): 负责输出动作概率分布（决定怎么做）
  - Critic (Value Network V):  负责估计当前状态的价值 V(s)（评价当前状态有多好，用来计算优势函数）
- 支持动作空间：**通用全能王！同时支持离散 (Discrete) 与连续 (Box) 动作空间**

【On-Policy 为什么需要多环境并行 (make_vec_env)？】
因为 PPO 是同策略算法，没有庞大的 Replay Buffer 存老数据。
为了在每次迭代中快速收集到足够多样化、低相关性的训练样本，工程上通常会同时开多个并行环境
（例如 n_envs=4 个子环境同时运行）。
每次采样迭代收集的总样本量计算公式为：
    RolloutBuffer 总样本数 = n_steps * n_envs
例如 n_steps=512, n_envs=4，则每次采样共收集 512 * 4 = 2048 步数据，
随后将这 2048 条数据打乱切分为若干个 batch_size=64 的小批次，反复训练 n_epochs=10 轮。

【PPO 的核心稳定机制：Clipped Surrogate Objective】
策略梯度算法最怕“步子迈太大扯着蛋”（一次更新过大导致策略崩溃，之后再也采不到好样本）。
PPO 通过限制新旧策略概率比值 r_t(θ) = π_new(a|s) / π_old(a|s) 只能在 [1 - clip_range, 1 + clip_range]
（通常为 [0.8, 1.2]）范围内获得收益，从而实现稳定、单调的策略提升。

运行方式：
    python 02_ppo_cartpole.py                                  # 使用默认步数 (50000 步) 完整训练
    python 02_ppo_cartpole.py --timesteps 15000                # 快速测试全流程
    python 02_ppo_cartpole.py --wandb                          # 开启 W&B 在线实验追踪
    python 02_ppo_cartpole.py --wandb --wandb-mode offline     # 开启 W&B 离线本地记录
================================================================================
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
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
    parser = argparse.ArgumentParser(description="SB3 PPO 并行环境训练 CartPole-v1 示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="总训练步数（注意：在 4 个并行环境下，50000 步实际只需约 12500 次环境并行 step）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数量（默认 4）")
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="保留历史训练日志与旧模型（默认每次开训前清空当前算法目录，仅保留最新一次）",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "CartPole-v1"

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "ppo_cartpole"
    log_dir = base_dir / "logs" / "ppo_cartpole"

    train_env = None
    eval_env = None
    demo_env = None
    wandb_run = None

    try:
        # ======================================================================
        # 第 1 步：创建矢量化并行环境与空间剖析
        # ======================================================================
        print("=" * 70)
        print(f"[第 1 步] 创建 {args.n_envs} 个并行的 Gymnasium 环境: {env_id}")
        # make_vec_env 是 SB3 的神器：它会自动为每个环境套上 Monitor 包装器，并打包成 VecEnv
        # 默认使用 DummyVecEnv（单进程顺序执行多个环境，对轻量级环境如 CartPole 开销最小）
        # 若环境物理仿真很重（如 MuJoCo/PyBullet），可传入 vec_env_cls=SubprocVecEnv 开启真正的多进程并行
        train_env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed)

        # 验证环境保持单个环境即可
        eval_env = Monitor(gym.make(env_id))

        print(f"  并行环境数 (num_envs):        {train_env.num_envs}")
        print(f"  观测空间 (Observation Space): {train_env.observation_space}")
        print(f"  动作空间 (Action Space):      {train_env.action_space}")

        # ======================================================================
        # 第 2 步：模型创建与超参数详解配置（及可选 W&B 初始化）
        # ======================================================================
        print("\n[第 2 步] 初始化 PPO 模型与核心超参数...")

        # 对于 Actor-Critic 架构（PPO / A2C），net_arch 可以用字典分别指定策略网络(pi)和价值网络(vf)的隐藏层
        # 这样 Actor 和 Critic 不共享特征层，在许多控制任务上训练更稳定
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

        # 攒够这 2048 条数据后，PPO 会把它们打乱；
        # 按照 batch_size = 64 切分成 2048 / 64 = 32 个小批次，依次做 32 次梯度更新（这算 1 个 epoch）；
        # 重复这个过程 n_epochs = 10 轮，也就是这批数据总共会触发 32 × 10 = 320 次网络参数更新；
        # 更新完后清空 RolloutBuffer（因为 PPO 是 On-Policy 算法，旧策略的数据不能留用），再重新采样下一批 2048 条数据。
        n_steps = 512
        batch_size = 64
        rollout_buffer_size = n_steps * args.n_envs
        print(f"  每次采样迭代 RolloutBuffer 大小 = n_steps({n_steps}) * n_envs({args.n_envs}) = {rollout_buffer_size}")

        hyperparams = dict(
            # --- 学习率与采样批次控制 ---
            learning_rate=3e-4,       # PPO 最经典的默认学习率（3e-4）
            n_steps=n_steps,          # 每个并行环境每次采集多少步数据后触发一次网络更新
            batch_size=batch_size,    # 小批次大小：必须尽量能整除 (n_steps * n_envs)
            n_epochs=10,              # 每次采集完一批 Rollout 数据后，重复遍历训练 10 轮 (epochs)

            # --- 折扣因子与广义优势估计 (GAE) ---
            gamma=0.99,               # 折扣因子 γ
            gae_lambda=0.95,          # GAE 平滑参数 λ ∈ [0, 1]：用于平衡优势估计的方差与偏差（0.9~0.98 常用）

            # --- PPO 核心裁剪、联合 Loss 系数与工程稳定项 ---
            # 注：教材基础伪代码常把 Actor/Critic 拆成两个独立优化器，故只会出现 ε (clip_range)；
            # SB3 工程实现使用单个 Adam 优化器优化联合总损失：
            #   L_total = L_clip + vf_coef * L_vf - ent_coef * Entropy
            clip_range=0.2,           # 对应伪代码裁剪超参数 ε：限制新旧策略概率比在 [1-ε, 1+ε] = [0.8, 1.2]，防止策略剧烈震荡
            ent_coef=0.01,            # 熵正则化系数（教材极简版常省略）：总 Loss 中减去 ent_coef*Entropy，鼓励探索、防止策略过早塌缩到次优动作
            vf_coef=0.5,              # 价值函数 (Critic) Loss 权重：单优化器共用 learning_rate 时，用于平衡 Critic 梯度与 Actor 梯度的尺度
            max_grad_norm=0.5,        # 工程防爆技巧（梯度裁剪）：反向传播后若梯度 L2 范数超过 0.5 则等比缩放，防止异常样本冲坏权重
        )

        wandb_run = init_wandb(
            args,
            env_id=env_id,
            algo_name="PPO",
            default_run_name=f"PPO_CartPole_seed{args.seed}",
            default_tags=["CartPole-v1", "PPO", "Discrete", "On-Policy"],
            config={
                "policy": "MlpPolicy",
                "total_timesteps": args.timesteps,
                "seed": args.seed,
                "n_envs": args.n_envs,
                "rollout_buffer_size": rollout_buffer_size,
                "net_arch": policy_kwargs["net_arch"],
                **hyperparams,
            },
            base_dir=base_dir,
        )
        prepare_run_dirs(model_dir, log_dir, base_dir=base_dir, keep_history=args.keep_history)

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            **hyperparams,
            # --- 网络架构与日志 ---
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
        print(f"\n[第 4 步] 开始训练 PPO（总步数 total_timesteps={args.timesteps}）...")
        # 注意：在 VecEnv 下，EvalCallback 的 eval_freq 是按“每次调用 env.step()（即 n_envs 步一起算 1 次）”计数的！
        # 因此若希望大约每 total_timesteps / 5 个总步数评估一次，需要除以 n_envs
        eval_freq = max(500, (args.timesteps // 5) // args.n_envs)
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
        tb_log_name = get_tb_log_name("PPO_CartPole", wandb_run)

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
        final_model_path = model_dir / "ppo_cartpole_final.zip"
        print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
        model.save(final_model_path)

        del model
        best_model_path = model_dir / "best_model.zip"
        best_model_updated = has_updated_best_model(eval_callback, best_model_path, run_start_time)
        load_path = best_model_path if best_model_updated else final_model_path
        print(f"  从磁盘重新加载模型: {load_path}")
        loaded_model = PPO.load(load_path, env=eval_env)

        # ======================================================================
        # 第 7 步：Gymnasium 标准五元组推理闭环演示
        # ======================================================================
        print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合...")
        demo_env = gym.make(env_id)
        obs, info = demo_env.reset(seed=args.seed)

        episode_reward = 0.0
        step_count = 0

        while True:
            # deterministic=True：取策略分布中概率最大的动作（或连续动作的均值），评估/部署时通常设为 True
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)

            episode_reward += float(reward)
            step_count += 1

            if terminated or truncated:
                reason = "达到最大步数截断 (truncated)" if truncated else "摆杆倾倒/出界终止 (terminated)"
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
            artifact_name="ppo-cartpole-model",
            final_model_path=final_model_path,
            best_model_path=best_model_path,
            best_model_updated=best_model_updated,
            metadata={
                "algorithm": "PPO",
                "env_id": env_id,
                "seed": args.seed,
                "n_envs": args.n_envs,
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
