"""
================================================================================
Stable-Baselines3 教程公共模块：Weights & Biases (W&B) 实验追踪封装
================================================================================

本模块为 5 个训练脚本提供统一的 W&B 接入能力，核心设计原则如下：
1. 默认关闭、零侵入：不传 `--wandb` 时不会导入 `wandb`，原有命令行为完全不变。
2. 自动同步 TensorBoard：通过 `sync_tensorboard=True` + `WandbCallback` 自动同步
   SB3 全部训练与评估曲线（rollout/*、train/*、eval/*、time/fps）。
3. 统一实验组织：默认归入 `sb3-tutorial` 项目，按环境自动分组（CartPole-v1 /
   Pendulum-v1），并附带算法、动作空间、gSDE 等标签，便于跨算法对比。
4. 完整资产归档：将训练前后评估汇总写入 `run.summary`（不干扰 TensorBoard 的
   global_step 时间轴），并将最终模型与本次更新的最优模型打包为 Model Artifact。
================================================================================
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from stable_baselines3.common.callbacks import BaseCallback, CallbackList


def add_wandb_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """为命令行解析器统一注册 Weights & Biases (W&B) 相关参数。"""
    group = parser.add_argument_group("Weights & Biases (W&B) 实验追踪配置")
    group.add_argument(
        "--wandb",
        action="store_true",
        help="开启 Weights & Biases 实验追踪（默认关闭，不开启时无需登录或联网）",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="sb3-tutorial",
        help="W&B 项目名称（默认 sb3-tutorial）",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B 账号或团队名称（默认使用当前已登录的账号）",
    )
    group.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B 实验分组名（默认自动使用环境名 env_id，如 CartPole-v1 / Pendulum-v1）",
    )
    group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="自定义本次 Run 名称（默认自动按 算法_环境_seed 生成）",
    )
    group.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="附加自定义标签（空格分隔，自动与内置算法/环境标签合并）",
    )
    group.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline"],
        default="online",
        help="W&B 运行模式：online 实时上传云端，offline 仅写本地 wandb/ 目录（默认 online）",
    )
    return parser


def _import_wandb() -> Any:
    """按需延迟导入 wandb；若未安装则给出明确的安装提示。"""
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "已开启 --wandb，但当前环境未安装 wandb。请先执行：\n"
            "  uv pip install --python .venv/bin/python -r requirements.txt\n"
            "或：\n"
            "  pip install wandb"
        ) from exc
    return wandb


def _merge_tags(
    default_tags: Sequence[str],
    extra_tags: Optional[Sequence[str]] = None,
) -> list[str]:
    """合并默认标签与用户自定义标签，保持顺序并去重。"""
    merged: list[str] = []
    for tag in list(default_tags) + list(extra_tags or []):
        cleaned = str(tag).strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged


def init_wandb(
    args: argparse.Namespace,
    *,
    env_id: str,
    algo_name: str,
    default_run_name: str,
    default_tags: Sequence[str],
    config: Mapping[str, Any],
    base_dir: Path,
) -> Optional[Any]:
    """
    若开启了 `--wandb`，初始化一个 W&B Run 并开启 TensorBoard 自动同步；
    否则返回 None。

    注意：必须在创建 SB3 模型与调用 `model.learn()` 之前调用本函数，
    以便 `sync_tensorboard=True` 能够自动挂钩 TensorBoard SummaryWriter。
    """
    if not getattr(args, "wandb", False):
        return None

    wandb = _import_wandb()
    group_name = args.wandb_group or env_id
    run_name = args.wandb_run_name or default_run_name
    tags = _merge_tags(default_tags, args.wandb_tags)

    full_config = {
        "algorithm": algo_name,
        "env_id": env_id,
        **dict(config),
    }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group_name,
        name=run_name,
        tags=tags,
        config=full_config,
        mode=args.wandb_mode,
        dir=str(base_dir),
        sync_tensorboard=True,
        save_code=True,
    )
    print(
        f"[W&B] 已初始化 Run: name={run.name} | id={run.id} | "
        f"project={args.wandb_project} | group={group_name} | mode={args.wandb_mode}"
    )
    return run


def get_tb_log_name(base_tb_log_name: str, wandb_run: Optional[Any]) -> str:
    """
    生成 TensorBoard 运行子目录名。
    开启 W&B 时附加 Run ID，避免与 logs/ 目录下的历史运行日志串扰。
    """
    if wandb_run is None:
        return base_tb_log_name
    return f"{base_tb_log_name}_{wandb_run.id}"


def build_callbacks(
    eval_callback: BaseCallback,
    wandb_run: Optional[Any],
    verbose: int = 2,
) -> BaseCallback:
    """
    组合训练回调：
    - 未开启 W&B：直接返回原始 EvalCallback；
    - 开启 W&B：使用 CallbackList 同时挂载 EvalCallback 与官方 WandbCallback。
    """
    if wandb_run is None:
        return eval_callback

    from wandb.integration.sb3 import WandbCallback

    wandb_callback = WandbCallback(verbose=verbose)
    return CallbackList([eval_callback, wandb_callback])


def has_updated_best_model(
    eval_callback: Any,
    best_model_path: Path,
    run_start_time: float,
) -> bool:
    """
    判断本次训练期间是否确实产出/更新了 best_model.zip。
    用于防止在短步数测试或未触发评估时，把上次历史运行遗留的旧模型误上传到当前 Artifact。
    """
    if not best_model_path.exists():
        return False
    best_reward = getattr(eval_callback, "best_mean_reward", -float("inf"))
    if best_reward <= -float("inf"):
        return False
    return best_model_path.stat().st_mtime >= run_start_time - 1e-3


def log_eval_summary(
    wandb_run: Optional[Any],
    summary_metrics: Mapping[str, Any],
) -> None:
    """
    将训练前后评估指标、推理回合结果等汇总数据写入 `run.summary`。
    相比 `wandb.log()`，写入 `summary` 不会与 `sync_tensorboard=True` 维护的
    global_step 时间轴产生步数冲突，同时能在 W&B Runs 表格中作为独立列横向对比。
    """
    if wandb_run is None:
        return
    wandb_run.summary.update(dict(summary_metrics))


def log_model_artifact(
    wandb_run: Optional[Any],
    *,
    artifact_name: str,
    final_model_path: Path,
    best_model_path: Optional[Path] = None,
    best_model_updated: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    aliases: Optional[Sequence[str]] = None,
) -> None:
    """
    将训练得到的最终模型（及本次运行更新的最优模型）打包上传为 W&B Model Artifact。
    """
    if wandb_run is None:
        return

    wandb = _import_wandb()
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=dict(metadata or {}),
    )

    added_files: list[str] = []
    if final_model_path.exists():
        artifact.add_file(str(final_model_path), name=final_model_path.name)
        added_files.append(final_model_path.name)

    if best_model_updated and best_model_path is not None and best_model_path.exists():
        artifact.add_file(str(best_model_path), name=best_model_path.name)
        added_files.append(best_model_path.name)

    if not added_files:
        print("[W&B] 未找到可上传的模型文件，跳过 Artifact 记录。")
        return

    artifact_aliases = list(aliases) if aliases else ["latest"]
    wandb_run.log_artifact(artifact, aliases=artifact_aliases)
    print(f"[W&B] 已记录模型 Artifact: {artifact_name} (包含文件: {', '.join(added_files)})")


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """安全结束当前 W&B Run，确保日志与 Artifact 完整刷盘。"""
    if wandb_run is None:
        return
    wandb_run.finish()


def prepare_run_dirs(
    model_dir: Path,
    log_dir: Path,
    *,
    base_dir: Path,
    keep_history: bool = False,
) -> None:
    """
    准备本次训练的模型与日志目录：
    - 默认（keep_history=False）：清空该算法变体旧的 models/ 与 logs/ 子目录，确保仅保留最新一次记录与模型；
    - 开启 --keep-history：保留历史运行文件，在原目录基础上追加。
    """
    models_root = (base_dir / "models").resolve()
    logs_root = (base_dir / "logs").resolve()
    resolved_model_dir = model_dir.resolve()
    resolved_log_dir = log_dir.resolve()

    if resolved_model_dir.parent != models_root:
        raise ValueError(
            f"安全校验失败：model_dir ({resolved_model_dir}) 必须位于 {models_root} 下一级子目录"
        )
    if resolved_log_dir.parent != logs_root:
        raise ValueError(
            f"安全校验失败：log_dir ({resolved_log_dir}) 必须位于 {logs_root} 下一级子目录"
        )

    if not keep_history:
        if resolved_model_dir.exists():
            shutil.rmtree(resolved_model_dir)
        if resolved_log_dir.exists():
            shutil.rmtree(resolved_log_dir)

    resolved_model_dir.mkdir(parents=True, exist_ok=True)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
