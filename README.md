# Procgen Generalization Study (COMP 579 Project)

A study of generalization in deep reinforcement learning using the [Procgen benchmark](https://github.com/openai/procgen). We train PPO agents on a fixed set of procedurally generated levels and evaluate them on held-out levels to measure the train/test generalization gap. We compare three architectural variants across three visually distinct games to see whether temporal context (frame stacking) and larger visual encoders help agents learn transferable skills rather than overfitting to specific training layouts.

---

## Research Question

Do PPO policies trained on a small set of Procgen levels learn transferable skills, or do they overfit to those specific layouts? And if they overfit, do architectural changes (frame stacking, larger CNN encoders) narrow the generalization gap?

We measure:

- **Seen-level return:** mean episode reward on training levels (levels 0-199)
- **Unseen-level return:** mean episode reward on held-out levels (levels 10000-10199)
- **Generalization gap:** `seen_return - unseen_return`. Positive = overfitting.

## Games

Three Procgen environments chosen for visual and mechanical diversity:

| Game | Description | Reward | Difficulty |
|---|---|---|---|
| **CoinRun** | Side-scrolling platformer. Navigate right, avoid obstacles, collect the coin. | Binary (0 or 10) | Easy |
| **Dodgeball** | Team projectile game. Dodge incoming balls while hitting opponents. | Sparse, incremental | Medium |
| **StarPilot** | Scrolling space shooter. Destroy enemies, dodge obstacles. | Continuous, fractional | Medium |

Previously we included Heist, but it requires ~25M timesteps to show non-zero returns, far beyond our compute budget. Dodgeball provides a better mid-difficulty data point.

## Variants

| Variant | Input | Encoder |
|---|---|---|
| `ppo` | Single RGB frame (3 channels) | Default SB3 `NatureCNN` |
| `ppo_frame_stack` | 4 stacked RGB frames (12 channels) | Default SB3 `NatureCNN` |
| `ppo_frame_stack_large_cnn` | 4 stacked RGB frames (12 channels) | Custom deeper CNN (see [procgen_experiment.py:192-215](procgen_experiment.py#L192-L215)) |

The custom `LargerCNN` uses a 64-128-128-128-channel architecture with 4 conv layers, roughly 2x deeper than the default SB3 encoder.

---

## Repository Layout

```
.
├── README.md                  # this file
├── procgen_experiment.py      # single-file training/evaluation pipeline
├── run_analysis.md            # written analysis of experimental results
├── .gitignore                 # excludes venvs, caches, old runs, large binaries
└── runs/                      # experiment outputs (only latest run tracked in git)
    └── procgen_generalization_20260403_170752/   # example run: 1M steps, 5 seeds
        ├── run_summary.json               # config + per-seed final metrics
        ├── aggregate_summary.csv          # mean/std across seeds per game/variant
        ├── final_metrics_per_seed.csv     # per-seed final metrics flat table
        ├── all_eval_logs.csv              # full eval trajectory (every checkpoint)
        ├── plots/                         # learning curves + final-metric bars
        │   ├── learning_curves_<game>.png
        │   └── final_bars_<game>.png
        └── <game>/<variant>/seed_<n>/     # per-run artifacts
            ├── eval_log.csv
            ├── model/ppo_model.zip        # not tracked in git (large)
            └── videos/*.gif               # not tracked in git (large)
```

---

## Environment Setup

Procgen's published macOS wheels are **x86_64-only**. On Apple Silicon you must run under Rosetta with an x86_64 Python 3.10 interpreter. On Linux/Windows x86_64 no Rosetta step is needed.

### Apple Silicon (macOS)

```bash
# Create an x86_64 Python 3.10 venv under Rosetta
arch -x86_64 /usr/local/bin/python3.10 -m venv .venv-procgen
source .venv-procgen/bin/activate

# Install pinned dependencies (Procgen needs gym 0.23 and numpy<2)
python -m pip install --upgrade pip
pip install "numpy<2" "gym==0.23.1" "procgen==0.10.7" \
    "gymnasium>=0.29,<1.1" "shimmy>=1.3,<2" \
    "stable-baselines3>=2.3,<2.5" \
    pandas matplotlib imageio imageio-ffmpeg tqdm rich
```

### Linux / x86_64

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "numpy<2" "gym==0.23.1" "procgen==0.10.7" \
    "gymnasium>=0.29,<1.1" "shimmy>=1.3,<2" \
    "stable-baselines3>=2.3,<2.5" \
    pandas matplotlib imageio imageio-ffmpeg tqdm rich
```

The script will refuse to run if it detects Python != 3.10 or native arm64 on macOS. It also refuses NumPy >= 2.0 because gym 0.23 is incompatible.

---

## Running Experiments

### Quick run (smoke test)

Single seed, 20K timesteps per run, 9 total runs (~a few minutes on CPU):

```bash
python procgen_experiment.py --run_mode quick
```

Quick-mode defaults:

| Parameter | Value |
|---|---|
| Total timesteps | 20,000 |
| Seeds | 1 (seed 0) |
| Training levels | 50 |
| Test levels | 50 |
| Eval frequency | every 5,000 steps |
| Eval episodes | 5 |
| Parallel envs | 4 |

### Full run (for real results)

Five seeds, 1M timesteps per run, 45 total runs (several hours on CPU):

```bash
python procgen_experiment.py --run_mode full
```

Full-mode defaults:

| Parameter | Value |
|---|---|
| Total timesteps | 1,000,000 |
| Seeds | 5 (seeds 0-4) |
| Training levels | 200 |
| Test levels | 200 |
| Eval frequency | every 100,000 steps |
| Eval episodes | 20 |
| Parallel envs | 8 |

### CLI overrides

Any of the defaults can be overridden:

```bash
python procgen_experiment.py \
    --run_mode full \
    --total_timesteps 500000 \
    --n_eval_episodes 10 \
    --output_dir runs/my_custom_run
```

Available flags: `--run_mode`, `--output_dir`, `--total_timesteps`, `--eval_freq`, `--n_eval_episodes`, `--n_envs`, `--train_num_levels`, `--test_num_levels`.

### Train/test split

- **Training levels:** start_level=0, num_levels=200 (i.e. levels 0-199)
- **Test levels:** start_level=10000, num_levels=200 (i.e. levels 10000-10199)

Procgen uses these parameters to deterministically generate levels, so this gives a reproducible seen/unseen split with no overlap.

---

## Outputs

Each run produces a timestamped directory under `runs/` containing:

- **`run_summary.json`** — complete experiment configuration plus final metrics for every (game, variant, seed) triple. The canonical record of the run.
- **`aggregate_summary.csv`** — mean and std of seen/unseen return and generalization gap, aggregated across seeds. This is the table used in the report.
- **`final_metrics_per_seed.csv`** — flat per-seed final metrics for spreadsheet analysis.
- **`all_eval_logs.csv`** — metrics at every evaluation checkpoint (not just the final one). Used to draw learning curves.
- **`plots/learning_curves_<game>.png`** — seen (dashed) and unseen (solid) returns over training for all three variants on one plot.
- **`plots/final_bars_<game>.png`** — bar chart comparing final unseen return and generalization gap across variants.
- **`<game>/<variant>/seed_<n>/model/ppo_model.zip`** — trained SB3 model weights (not tracked in git).
- **`<game>/<variant>/seed_<n>/videos/<name>_seen.gif`** and `<name>_unseen.gif` — rollout videos on a seen and unseen level (not tracked in git).

The `runs/procgen_generalization_20260403_170752/` directory in this repo is a full 1M-step, 5-seed example run included so you can inspect the output format without having to re-run training. Model weights and GIFs are excluded from git to keep the repo small.

---

## Implementation Notes

The entire pipeline is in a single file, [procgen_experiment.py](procgen_experiment.py), which:

1. **Guards the runtime** ([check_runtime_or_raise](procgen_experiment.py#L23)) — fails fast if Python version or architecture is wrong.
2. **Builds the config** ([build_config](procgen_experiment.py#L150)) — `quick` vs `full` mode + CLI overrides.
3. **Creates Procgen environments** ([make_procgen_env](procgen_experiment.py#L235)) — uses `gym.make` to create the legacy env, then wraps with `shimmy.GymV21CompatibilityV0` for Gymnasium compatibility. Includes a patch for the [`ToGymEnv.seed()` signature mismatch](procgen_experiment.py#L250-L259) between procgen's gym env and shimmy's reset-time seeding logic.
4. **Trains** ([train_one_run](procgen_experiment.py)) — instantiates PPO with the variant-specific settings, trains for `total_timesteps`, evaluates on both seen and unseen levels every `eval_freq` steps.
5. **Aggregates and plots** — writes all the CSVs/JSON/plots listed above.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Learning rate | 2.5e-4 |
| n_steps | 256 |
| batch_size | 256 |
| n_epochs | 4 |
| gamma | 0.999 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

These follow the hyperparameters from Cobbe et al. (2020) "Leveraging Procedural Generation to Benchmark Reinforcement Learning". The `distribution_mode` is `"easy"` for all games.

---

## Results Summary

See [run_analysis.md](run_analysis.md) for the full written analysis of the 1M-step run. Key findings:

- **Generalization gaps exist but are small** at 1M steps with 200 training levels. Most clearly visible on CoinRun with baseline PPO (+0.90 +/- 0.89). At this training scale PPO has not yet overfit strongly enough for the gap to be large relative to seed variance.
- **Architectural effects are game-dependent.** Frame stacking helps CoinRun (platforming benefits from velocity cues) but hurts Dodgeball (overwhelms the small CNN). The large CNN helps Dodgeball (complex visual scene) but is too slow to learn on StarPilot at our budget. There is no universal best variant.
- **Learning instability dominates the CoinRun results.** PPO on CoinRun shows dramatic non-monotonic learning curves, with agents reaching peak performance mid-training and then collapsing. Final-checkpoint metrics underestimate peak capability.
- **For Procgen generalization studies to reach a clear verdict, larger compute budgets are needed** (5M+ steps, ideally the canonical 25M). At 1M steps the project demonstrates the methodology and captures early-training dynamics but cannot definitively rank architectures.

---

## References

- Cobbe et al. (2020), "Leveraging Procedural Generation to Benchmark Reinforcement Learning" — the Procgen paper. [arXiv:1912.01588](https://arxiv.org/abs/1912.01588)
- Schulman et al. (2017), "Proximal Policy Optimization Algorithms" — PPO. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- [OpenAI Procgen repo](https://github.com/openai/procgen)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and [Shimmy](https://github.com/Farama-Foundation/Shimmy)
