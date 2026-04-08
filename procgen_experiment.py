"""

we designed this for Apple Silicon Macs running an x86_64 / Rosetta Python 3.10 environment,
because Procgen's published macOS wheels are x86_64-only.

Run:
  python procgen_generalization_macos_local.py --run_mode quick
  python procgen_generalization_macos_local.py --run_mode full
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional


def check_runtime_or_raise() -> None:
    py = sys.version_info
    if (py.major, py.minor) != (3, 10):
        raise SystemExit(
            "This script expects Python 3.10. "
            f"You are using {py.major}.{py.minor}. "
            "Create a Python 3.10 virtual environment first."
        )

    if sys.platform == "darwin" and platform.machine() != "x86_64":
        raise SystemExit(
            "On Apple Silicon, run this script under a Rosetta x86_64 Python 3.10 environment. "
            "Your current interpreter is native arm64, which will usually fail to install Procgen."
        )


def safe_imports():
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import gym
        import gymnasium as gymnasium
        import imageio.v2 as imageio
        import torch as th
        import torch.nn as nn
        from packaging.version import Version
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecFrameStack,
            VecMonitor,
            VecTransposeImage,
        )
        import procgen  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing or incompatible dependencies. Install them in a fresh x86_64 Python 3.10 env with:\n\n"
            'pip install "numpy<2" "gym==0.23.1" "procgen==0.10.7" '
            '"gymnasium>=0.29,<1.1" "shimmy>=1.3,<2" '
            '"stable-baselines3>=2.3,<2.5" pandas matplotlib imageio imageio-ffmpeg\n\n'
            f"Original import error: {exc}"
        )

    if Version(np.__version__) >= Version("2.0.0"):
        raise SystemExit(
            f"NumPy {np.__version__} is too new for this stack. Use numpy<2."
        )

    return {
        "np": np,
        "pd": pd,
        "plt": plt,
        "gym": gym,
        "gymnasium": gymnasium,
        "imageio": imageio,
        "th": th,
        "nn": nn,
        "GymV21CompatibilityV0": GymV21CompatibilityV0,
        "PPO": PPO,
        "BaseCallback": BaseCallback,
        "evaluate_policy": evaluate_policy,
        "BaseFeaturesExtractor": BaseFeaturesExtractor,
        "DummyVecEnv": DummyVecEnv,
        "VecFrameStack": VecFrameStack,
        "VecMonitor": VecMonitor,
        "VecTransposeImage": VecTransposeImage,
    }


check_runtime_or_raise()
mods = safe_imports()

np = mods["np"]
pd = mods["pd"]
plt = mods["plt"]
gym = mods["gym"]
gymnasium = mods["gymnasium"]
imageio = mods["imageio"]
th = mods["th"]
nn = mods["nn"]
GymV21CompatibilityV0 = mods["GymV21CompatibilityV0"]
PPO = mods["PPO"]
BaseCallback = mods["BaseCallback"]
evaluate_policy = mods["evaluate_policy"]
BaseFeaturesExtractor = mods["BaseFeaturesExtractor"]
DummyVecEnv = mods["DummyVecEnv"]
VecFrameStack = mods["VecFrameStack"]
VecMonitor = mods["VecMonitor"]
VecTransposeImage = mods["VecTransposeImage"]

GAMES = ["coinrun", "dodgeball", "starpilot"]
VARIANTS = ["ppo", "ppo_frame_stack", "ppo_frame_stack_large_cnn"]
DISTRIBUTION_MODE = "easy"


@dataclass
class ExperimentConfig:
    run_mode: str
    total_timesteps: int
    eval_freq: int
    n_eval_episodes: int
    n_envs: int
    train_num_levels: int
    test_num_levels: int
    train_start_level: int
    test_start_level: int
    seeds: List[int]
    frame_stack: int
    video_max_steps: int
    out_dir: str
    learning_rate: float = 2.5e-4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.output_dir or f"runs/procgen_generalization_{timestamp}"
    ).resolve()

    if args.run_mode == "quick":
        cfg = ExperimentConfig(
            run_mode="quick",
            total_timesteps=args.total_timesteps or 20_000,
            eval_freq=args.eval_freq or 5_000,
            n_eval_episodes=args.n_eval_episodes or 5,
            n_envs=args.n_envs or 4,
            train_num_levels=args.train_num_levels or 50,
            test_num_levels=args.test_num_levels or 50,
            train_start_level=0,
            test_start_level=10_000,
            seeds=[0],
            frame_stack=4,
            video_max_steps=400,
            out_dir=str(out_dir),
        )
    else:
        cfg = ExperimentConfig(
            run_mode="full",
            total_timesteps=args.total_timesteps or 1_000_000,
            eval_freq=args.eval_freq or 100_000,
            n_eval_episodes=args.n_eval_episodes or 20,
            n_envs=args.n_envs or 8,
            train_num_levels=args.train_num_levels or 200,
            test_num_levels=args.test_num_levels or 200,
            train_start_level=0,
            test_start_level=10_000,
            seeds=[0, 1, 2, 3, 4],
            frame_stack=4,
            video_max_steps=800,
            out_dir=str(out_dir),
        )

    return cfg


class LargerCNN(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gymnasium.spaces.Box, features_dim: int = 512
    ):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def variant_settings(variant: str, cfg: ExperimentConfig) -> Dict:
    settings = {
        "frame_stack": 1,
        "policy_kwargs": {},
        "variant": variant,
    }
    if variant == "ppo_frame_stack":
        settings["frame_stack"] = cfg.frame_stack
    elif variant == "ppo_frame_stack_large_cnn":
        settings["frame_stack"] = cfg.frame_stack
        settings["policy_kwargs"] = {
            "features_extractor_class": LargerCNN,
            "features_extractor_kwargs": {"features_dim": 512},
        }
    return settings


def make_procgen_env(
    game: str,
    start_level: int,
    num_levels: int,
    seed: int,
    render_mode: Optional[str] = None,
) -> gymnasium.Env:
    legacy_env = gym.make(
        f"procgen-{game}-v0",
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode=DISTRIBUTION_MODE,
        use_sequential_levels=False,
        render_mode=render_mode,
    )
    # Patch seed() on the entire gym wrapper chain — procgen envs are
    # seeded at creation time via start_level, but shimmy/SB3 will call
    # env.seed(seed) during reset() and the innermost ToGymEnv.seed()
    # does not accept arguments.
    e = legacy_env
    while True:
        e.seed = lambda *_a, **_kw: None
        if not hasattr(e, "env"):
            break
        e = e.env
    env = GymV21CompatibilityV0(env=legacy_env, render_mode=render_mode)
    env.reset()
    return env


def make_vec_env(
    game: str,
    start_level: int,
    num_levels: int,
    seed: int,
    n_envs: int,
    frame_stack: int,
    render_mode: Optional[str] = None,
):
    def thunk(rank: int) -> Callable[[], gymnasium.Env]:
        def _make() -> gymnasium.Env:
            return make_procgen_env(
                game=game,
                start_level=start_level,
                num_levels=num_levels,
                seed=seed + rank,
                render_mode=render_mode,
            )

        return _make

    env = DummyVecEnv([thunk(i) for i in range(n_envs)])
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env


class DualEvalCallback(BaseCallback):
    def __init__(
        self,
        seen_eval_env,
        unseen_eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        game: str,
        variant: str,
        seed: int,
    ):
        super().__init__()
        self.seen_eval_env = seen_eval_env
        self.unseen_eval_env = unseen_eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.game = game
        self.variant = variant
        self.seed = seed
        self.records: List[Dict] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            seen_mean, seen_std = evaluate_policy(
                self.model,
                self.seen_eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                warn=False,
            )
            unseen_mean, unseen_std = evaluate_policy(
                self.model,
                self.unseen_eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                warn=False,
            )
            self.records.append(
                {
                    "game": self.game,
                    "variant": self.variant,
                    "seed": self.seed,
                    "timesteps": int(self.num_timesteps),
                    "seen_return_mean": float(seen_mean),
                    "seen_return_std": float(seen_std),
                    "unseen_return_mean": float(unseen_mean),
                    "unseen_return_std": float(unseen_std),
                    "generalization_gap": float(seen_mean - unseen_mean),
                }
            )
        return True


def train_one_run(
    cfg: ExperimentConfig,
    game: str,
    variant: str,
    seed: int,
    output_root: Path,
) -> Dict:
    settings = variant_settings(variant, cfg)
    frame_stack = settings["frame_stack"]
    policy_kwargs = settings["policy_kwargs"]

    run_dir = output_root / game / variant / f"seed_{seed}"
    model_dir = run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        game=game,
        start_level=cfg.train_start_level,
        num_levels=cfg.train_num_levels,
        seed=seed,
        n_envs=cfg.n_envs,
        frame_stack=frame_stack,
    )
    seen_eval_env = make_vec_env(
        game=game,
        start_level=cfg.train_start_level,
        num_levels=cfg.train_num_levels,
        seed=seed + 1_000,
        n_envs=1,
        frame_stack=frame_stack,
    )
    unseen_eval_env = make_vec_env(
        game=game,
        start_level=cfg.test_start_level,
        num_levels=cfg.test_num_levels,
        seed=seed + 2_000,
        n_envs=1,
        frame_stack=frame_stack,
    )

    callback = DualEvalCallback(
        seen_eval_env=seen_eval_env,
        unseen_eval_env=unseen_eval_env,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        game=game,
        variant=variant,
        seed=seed,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        device="cpu",
    )

    model.learn(
        total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=True
    )
    model_path = model_dir / "ppo_model.zip"
    model.save(str(model_path))

    if not callback.records or callback.records[-1]["timesteps"] != cfg.total_timesteps:
        seen_mean, seen_std = evaluate_policy(
            model,
            seen_eval_env,
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        unseen_mean, unseen_std = evaluate_policy(
            model,
            unseen_eval_env,
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        callback.records.append(
            {
                "game": game,
                "variant": variant,
                "seed": seed,
                "timesteps": int(cfg.total_timesteps),
                "seen_return_mean": float(seen_mean),
                "seen_return_std": float(seen_std),
                "unseen_return_mean": float(unseen_mean),
                "unseen_return_std": float(unseen_std),
                "generalization_gap": float(seen_mean - unseen_mean),
            }
        )

    eval_df = pd.DataFrame(callback.records)
    eval_df.to_csv(run_dir / "eval_log.csv", index=False)

    final_metrics = dict(eval_df.iloc[-1].to_dict())
    final_metrics["model_path"] = str(model_path)

    model = PPO.load(str(model_path), device="cpu")
    record_rollouts(
        model=model,
        cfg=cfg,
        game=game,
        variant=variant,
        seed=seed,
        output_dir=run_dir / "videos",
        frame_stack=frame_stack,
    )

    train_env.close()
    seen_eval_env.close()
    unseen_eval_env.close()

    return final_metrics


def record_rollouts(
    model,
    cfg: ExperimentConfig,
    game: str,
    variant: str,
    seed: int,
    output_dir: Path,
    frame_stack: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("seen", cfg.train_start_level, cfg.train_num_levels),
        ("unseen", cfg.test_start_level, cfg.test_num_levels),
    ]

    for split_name, start_level, num_levels in specs:
        raw_env = make_procgen_env(
            game=game,
            start_level=start_level,
            num_levels=num_levels,
            seed=seed + (0 if split_name == "seen" else 50_000),
            render_mode="rgb_array",
        )
        obs, _ = raw_env.reset()

        frame_buffer: List[np.ndarray] = []
        if frame_stack > 1:
            for _ in range(frame_stack):
                frame_buffer.append(obs)

        frames: List[np.ndarray] = []
        for _ in range(cfg.video_max_steps):
            if frame_stack > 1:
                stacked = np.concatenate(
                    [f.transpose(2, 0, 1) for f in frame_buffer], axis=0
                )
                action, _ = model.predict(stacked, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _, terminated, truncated, _ = raw_env.step(int(action))
            frame = raw_env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)
            if frame_stack > 1:
                frame_buffer.pop(0)
                frame_buffer.append(obs)
            if terminated or truncated:
                break

        raw_env.close()
        video_path = output_dir / f"{game}_{variant}_seed{seed}_{split_name}.mp4"
        gif_path = output_dir / f"{game}_{variant}_seed{seed}_{split_name}.gif"

        if not frames:
            continue

        try:
            with imageio.get_writer(video_path, fps=15) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception:
            imageio.mimsave(gif_path, frames, fps=15)


def aggregate_results(output_root: Path) -> None:
    eval_logs = sorted(output_root.glob("*/*/seed_*/eval_log.csv"))
    if not eval_logs:
        return

    all_eval = pd.concat([pd.read_csv(p) for p in eval_logs], ignore_index=True)
    all_eval.to_csv(output_root / "all_eval_logs.csv", index=False)

    final_df = (
        all_eval.sort_values(["game", "variant", "seed", "timesteps"])
        .groupby(["game", "variant", "seed"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    final_df.to_csv(output_root / "final_metrics_per_seed.csv", index=False)

    summary = (
        final_df.groupby(["game", "variant"], as_index=False)
        .agg(
            seen_return_mean=("seen_return_mean", "mean"),
            seen_return_std=("seen_return_mean", "std"),
            unseen_return_mean=("unseen_return_mean", "mean"),
            unseen_return_std=("unseen_return_mean", "std"),
            gap_mean=("generalization_gap", "mean"),
            gap_std=("generalization_gap", "std"),
        )
        .fillna(0.0)
    )
    summary.to_csv(output_root / "aggregate_summary.csv", index=False)

    plot_learning_curves(all_eval, output_root / "plots")
    plot_final_bars(summary, output_root / "plots")


def plot_learning_curves(all_eval: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for game in sorted(all_eval["game"].unique()):
        game_df = all_eval[all_eval["game"] == game].copy()
        agg = (
            game_df.groupby(["variant", "timesteps"], as_index=False)
            .agg(
                seen_mean=("seen_return_mean", "mean"),
                seen_std=("seen_return_mean", "std"),
                unseen_mean=("unseen_return_mean", "mean"),
                unseen_std=("unseen_return_mean", "std"),
            )
            .fillna(0.0)
        )

        plt.figure(figsize=(10, 6))
        for variant in VARIANTS:
            v = agg[agg["variant"] == variant]
            if v.empty:
                continue
            x = v["timesteps"].to_numpy()
            y_seen = v["seen_mean"].to_numpy()
            y_unseen = v["unseen_mean"].to_numpy()
            s_seen = v["seen_std"].to_numpy()
            s_unseen = v["unseen_std"].to_numpy()
            plt.plot(x, y_seen, linestyle="--", label=f"{variant} seen")
            plt.plot(x, y_unseen, linestyle="-", label=f"{variant} unseen")
            plt.fill_between(x, y_seen - s_seen, y_seen + s_seen, alpha=0.15)
            plt.fill_between(x, y_unseen - s_unseen, y_unseen + s_unseen, alpha=0.15)
        plt.title(f"Learning curves on {game}")
        plt.xlabel("Timesteps")
        plt.ylabel("Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"learning_curves_{game}.png", dpi=160)
        plt.close()


def plot_final_bars(summary: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for game in sorted(summary["game"].unique()):
        game_df = summary[summary["game"] == game].copy()
        if game_df.empty:
            continue
        order = [v for v in VARIANTS if v in game_df["variant"].tolist()]
        game_df = game_df.set_index("variant").loc[order].reset_index()

        x = np.arange(len(game_df))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(
            x - width / 2,
            game_df["unseen_return_mean"],
            width,
            yerr=game_df["unseen_return_std"],
            capsize=4,
            label="Unseen return",
        )
        plt.bar(
            x + width / 2,
            game_df["gap_mean"],
            width,
            yerr=game_df["gap_std"],
            capsize=4,
            label="Generalization gap",
        )
        plt.xticks(x, game_df["variant"], rotation=15)
        plt.title(f"Final metrics on {game}")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"final_bars_{game}.png", dpi=160)
        plt.close()


def save_run_summary(
    cfg: ExperimentConfig, output_root: Path, final_metrics: List[Dict]
) -> None:
    payload = {
        "config": asdict(cfg),
        "games": GAMES,
        "variants": VARIANTS,
        "n_runs": len(final_metrics),
        "final_metrics": final_metrics,
    }
    with open(output_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--n_eval_episodes", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--train_num_levels", type=int, default=None)
    parser.add_argument("--test_num_levels", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    output_root = Path(cfg.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    final_metrics: List[Dict] = []
    print(f"Saving outputs to: {output_root}")
    print(f"Run mode: {cfg.run_mode}")
    print(f"Games: {GAMES}")
    print(f"Variants: {VARIANTS}")
    print(f"Seeds: {cfg.seeds}")

    for game in GAMES:
        for variant in VARIANTS:
            for seed in cfg.seeds:
                print("=" * 80)
                print(f"Training game={game} variant={variant} seed={seed}")
                result = train_one_run(
                    cfg=cfg,
                    game=game,
                    variant=variant,
                    seed=seed,
                    output_root=output_root,
                )
                final_metrics.append(result)
                print(json.dumps(result, indent=2))

    save_run_summary(cfg, output_root, final_metrics)
    aggregate_results(output_root)
    print("Done.")
    print(f"Artifacts written to: {output_root}")


if __name__ == "__main__":
    main()
