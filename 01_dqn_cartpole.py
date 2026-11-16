"""
================================================================================
Stable-Baselines3 实战教程 01：DQN (Deep Q-Network) 求解 CartPole-v1
================================================================================

【算法定位】
- 算法名称：DQN (深度 Q 网络)
- 策略类型：Off-Policy（异策略）—— 可以把历史交互经验存入“经验回放池”并重复利用
- 学习范式：Value-Based（基于价值）—— 不直接输出动作概率，而是估计每个动作的 Q 值
- 支持动作空间：**仅支持离散动作空间 (Discrete)**，例如“向左/向右”、“上/下/左/右”

【为什么 DQN 只能处理离散动作空间？】
DQN 更新网络时的目标值（TD Target）计算公式为：
    y = r + gamma * max_{a'} Q_target(s', a')
注意其中的 `max_{a'}` 操作：它需要把下一状态 s' 下所有可能动作的 Q 值全部算出来再取最大值。
如果是连续动作空间（比如力矩在 [-2.0, 2.0] 之间有无穷多个实数），根本无法穷举求 max。
因此连续动作空间需要使用 PPO、TD3 或 SAC 等 Actor-Critic 算法。

【DQN 的两大核心稳定机制】
1. 经验回放池 (Replay Buffer)：打破连续时间步样本之间的强相关性，让数据近似独立同分布（i.i.d.），并大幅提升样本利用率。
2. 目标网络 (Target Network)：使用一个参数更新较慢的目标 Q 网络计算 TD Target，避免“自己追自己”引起的训练震荡与发散。

【实验环境：CartPole-v1（倒立摆车）】
- 观测空间 (Observation Space): Box(4,) —— [小车位置, 小车速度, 摆杆角度, 摆杆角速度]
- 动作空间 (Action Space): Discrete(2) —— 0 代表向左推小车，1 代表向右推小车
- 奖励机制 (Reward): 摆杆每保持直立 1 步，获得 +1 奖励；每回合最多 500 步（满分 500 分）

运行方式：
    python 01_dqn_cartpole.py                  # 使用默认步数 (50000 步) 完整训练
    python 01_dqn_cartpole.py --timesteps 15000 # 快速测试全流程
================================================================================
"""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便指定训练总步数与随机种子。"""
    parser = argparse.ArgumentParser(description="SB3 DQN 训练 CartPole-v1 示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="总训练环境交互步数（默认 50000 步，CartPole-v1 通常在 2~5 万步即可接近满分）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证实验可复现")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_id = "CartPole-v1"

    # 定义模型与日志保存目录（当前脚本所在目录下的 models/ 和 logs/）
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "dqn_cartpole"
    log_dir = base_dir / "logs" / "dqn_cartpole"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 第 1 步：环境初始化与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 环境: {env_id}")
    # Monitor 包装器用于自动记录每个回合 (Episode) 的总奖励和回合长度，供日志和回调函数使用
    train_env = Monitor(gym.make(env_id))
    # 独立创建验证环境，避免评估过程污染训练环境状态
    eval_env = Monitor(gym.make(env_id))

    # 查看观测空间与动作空间（新手写强化学习代码前务必先打印这两个属性！）
    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  观测维度 (Shape):             {train_env.observation_space.shape}")
    print(f"  观测下界 (Low):               {train_env.observation_space.low}")
    print(f"  观测上界 (High):              {train_env.observation_space.high}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}  (离散动作数量 n={train_env.action_space.n})")

    # ==========================================================================
    # 第 2 步：模型创建与超参数详解配置
    # ==========================================================================
    print("\n[第 2 步] 初始化 DQN 模型与核心超参数...")

    # policy_kwargs 用于自定义神经网络结构
    # 对于 DQN，net_arch=[128, 128] 表示特征输入后接两层全连接网络，每层 128 个神经元（默认是 [64, 64]）
    policy_kwargs = dict(net_arch=[128, 128])

    model = DQN(
        # 策略类型："MlpPolicy" 表示多层感知机（全连接网络），适用于向量观测；
        # 若输入是图像像素，则需改为 "CnnPolicy"；若观测是字典类型，改为 "MultiInputPolicy"。
        policy="MlpPolicy",
        env=train_env,

        # --- 优化器与折扣因子 ---
        learning_rate=1e-3,             # 学习率：DQN 常用 1e-4 ~ 1e-3
        gamma=0.99,                     # 折扣因子 γ ∈ [0, 1)：越接近 1 越重视长远未来奖励

        # --- 经验回放池 (Replay Buffer) 相关参数 ---
        buffer_size=50_000,             # 经验池最大容量：最多存储 5 万条 (s, a, r, s', done) 转移样本
        learning_starts=1_000,          # 预热步数：先随机探索收集满 1000 条样本后再开始梯度更新，避免前期数据过少导致过拟合
        batch_size=64,                  # 每次更新从经验池中随机采样的批量大小

        # --- 训练频率与目标网络更新 ---
        train_freq=4,                   # 每与环境交互 4 步，执行一次训练更新
        gradient_steps=1,               # 每次触发训练时，做 1 次梯度下降更新
        target_update_interval=500,     # 目标网络同步间隔：每 500 个环境步将主 Q 网络权重同步到目标 Q 网络
        tau=1.0,                        # 软更新系数：tau=1.0 表示硬更新（直接覆盖复制权重）；若 <1.0 则做指数滑动平均

        # --- ε-greedy（epsilon-贪婪）探索策略衰减 ---
        # 智能体在训练初期以高概率随机探索，后期逐渐减小随机性并依赖学到的 Q 网络决策
        exploration_initial_eps=1.0,    # 初始探索率 ε=1.0（100% 随机动作）
        exploration_fraction=0.2,       # 在前 20% 的总训练步数内，将 ε 从 initial 线性衰减到 final
        exploration_final_eps=0.05,     # 最终探索率 ε=0.05（保留 5% 随机探索防止陷入局部最优）

        # --- 网络结构、日志与随机种子 ---
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),   # TensorBoard 日志保存路径
        seed=args.seed,
        verbose=1,                      # 日志详尽程度：0=静默，1=输出训练指标表，2=输出调试信息
    )

    # ==========================================================================
    # 第 3 步：训练前基线评估 (Baseline Evaluation)
    # ==========================================================================
    print("\n[第 3 步] 训练前基线评估（未训练模型的随机/初始表现）...")
    # evaluate_policy 会在 eval_env 上运行 n_eval_episodes 个完整回合，返回平均奖励与标准差
    mean_reward_before, std_reward_before = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    # ==========================================================================
    # 第 4 步：配置回调函数 (Callback) 与启动训练
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练 DQN（总步数 total_timesteps={args.timesteps}）...")
    # EvalCallback 会每隔 eval_freq 步在 eval_env 上评估一次当前策略，
    # 如果平均奖励刷新了历史最高记录，就自动把最优权重保存为 best_model.zip。
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
        tb_log_name="DQN_CartPole",
        progress_bar=False,
    )

    # ==========================================================================
    # 第 5 步：训练后定量评估 (Post-Training Evaluation)
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载 (Save & Load)
    # ==========================================================================
    final_model_path = model_dir / "dqn_cartpole_final.zip"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    model.save(final_model_path)

    # 删除内存中的旧模型对象，演示从磁盘文件全新加载模型
    del model
    # 如果训练中保存了表现更好的 best_model.zip，优先加载它，否则加载 final 模型
    best_model_path = model_dir / "best_model.zip"
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = DQN.load(load_path, env=eval_env)

    # ==========================================================================
    # 第 7 步：Gymnasium 标准五元组推理闭环演示
    # ==========================================================================
    print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合（Gymnasium 标准循环）...")
    demo_env = gym.make(env_id)
    # Gymnasium reset() 返回二元组: (初始观测 obs, 辅助信息字典 info)
    obs, info = demo_env.reset(seed=args.seed)

    episode_reward = 0.0
    step_count = 0

    while True:
        # deterministic=True 表示推理阶段关闭 ε-greedy 随机探索，总是选择 Q 值最大的最优动作
        action, _states = loaded_model.predict(obs, deterministic=True)

        # Gymnasium step() 返回五元组:
        # - obs:        下一时刻状态观测
        # - reward:     本步获得的标量奖励
        # - terminated: 任务本质终止（例如在 CartPole 中摆杆倾倒超过阈值角度，或小车驶出边界）
        # - truncated:  时间步截断（例如达到环境设定的最大步数 500 步，并非失败）
        # - info:       调试/诊断附加信息字典
        obs, reward, terminated, truncated, info = demo_env.step(action)

        episode_reward += float(reward)
        step_count += 1

        # 只要 terminated 或 truncated 任一为 True，说明当前回合结束，必须退出循环并 reset()
        if terminated or truncated:
            reason = "达到最大步数截断 (truncated)" if truncated else "摆杆倾倒/出界终止 (terminated)"
            print(f"  推理回合结束 | 总步数: {step_count} | 累计奖励: {episode_reward:.2f} | 结束原因: {reason}")
            break

    # 释放环境资源
    train_env.close()
    eval_env.close()
    demo_env.close()
    print("=" * 70)


if __name__ == "__main__":
    main()
