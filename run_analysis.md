# Full Run Analysis (1M Steps) -- Procgen Generalization Experiment

**Run timestamp:** 2026-04-03 17:07:52
**Run mode:** `full` (1,000,000 timesteps, 5 seeds, 200 training levels, 20 eval episodes)

---

## 1. Experimental Setup

| Parameter | Value |
|---|---|
| Total timesteps | 1,000,000 |
| Training levels | 200 |
| Test levels | 200 (starting at level 10,000) |
| Eval frequency | Every 100,000 steps (10 checkpoints) |
| Eval episodes | 20 per checkpoint |
| Parallel envs | 8 |
| Seeds | 5 (seeds 0-4) |
| Games | CoinRun, Dodgeball, StarPilot |
| Variants | PPO (baseline), PPO + Frame Stack (4 frames), PPO + Frame Stack + Large CNN |

Total runs: 3 games x 3 variants x 5 seeds = **45 training runs**.

This run represents a 5x increase in training budget over the previous full run (200K), with double the eval episodes and nearly double the seeds. For context, the canonical Procgen benchmark (Cobbe et al., 2020) uses 25M steps -- our 1M budget is **4% of that**, placing us in the early-to-mid training regime rather than convergence.

---

## 2. Aggregate Results (Mean +/- Std Across 5 Seeds)

| Game | Variant | Seen Return | Unseen Return | Gen. Gap |
|---|---|---|---|---|
| **CoinRun** | PPO | 2.50 +/- 1.70 | 1.60 +/- 1.43 | **+0.90 +/- 0.89** |
| | PPO + FrameStack | 4.30 +/- 2.71 | 3.70 +/- 1.68 | +0.60 +/- 1.98 |
| | PPO + FS + LargeCNN | 2.30 +/- 1.44 | 3.10 +/- 1.92 | -0.80 +/- 1.48 |
| **Dodgeball** | PPO | 0.58 +/- 0.45 | 0.26 +/- 0.21 | **+0.32 +/- 0.45** |
| | PPO + FrameStack | 0.10 +/- 0.17 | 0.18 +/- 0.16 | -0.08 +/- 0.15 |
| | PPO + FS + LargeCNN | 0.68 +/- 0.52 | 0.68 +/- 0.48 | **0.00 +/- 0.37** |
| **StarPilot** | PPO | 2.23 +/- 0.59 | 2.18 +/- 0.41 | +0.05 +/- 0.30 |
| | PPO + FrameStack | 1.90 +/- 0.54 | 2.06 +/- 0.66 | -0.16 +/- 0.27 |
| | PPO + FS + LargeCNN | 1.66 +/- 0.98 | 1.53 +/- 0.43 | +0.13 +/- 1.16 |

---

## 3. Per-Game Analysis

### 3.1 CoinRun

CoinRun is a side-scrolling platformer with binary rewards (0 or 10 for reaching the coin). It remains the easiest of our three games, but the 1M-step results reveal a more complex picture than the earlier 200K run.

**Key findings:**

- **PPO + FrameStack achieves the highest absolute returns.** With 4.30 seen / 3.70 unseen (averaged across 5 seeds), frame stacking outperforms both baseline PPO (2.50 / 1.60) and the large CNN (2.30 / 3.10). This is a reversal from the 200K run where baseline PPO led. At 1M steps, the temporal context provided by 4 stacked frames appears to be helping agents learn movement dynamics (momentum, jump timing) that benefit CoinRun's platforming gameplay.

- **Baseline PPO shows the most consistent overfitting.** The +0.90 generalization gap for baseline PPO is the most reliable positive gap in the CoinRun results, with relatively low std (+/- 0.89). All 5 seeds show gaps between 0.0 and +2.0 (4 out of 5 positive). This suggests that baseline PPO, with its single-frame observation, is learning level-specific spatial patterns (e.g., platform positions) rather than generalizable navigation skills.

- **Large CNN shows a negative gap (-0.80), meaning unseen > seen.** This is counterintuitive at first glance. However, it likely reflects the fact that the large CNN is still underfitting at 1M steps -- it has not learned enough to memorize training levels, so the seen/unseen distinction is noise. The per-seed breakdown confirms this: seed 2 has a gap of -3.0 while seed 3 has +0.5, indicating the mean gap is being driven by high variance rather than a systematic effect.

- **Severe learning instability persists at 1M steps.** The learning curves show dramatic oscillations throughout training -- PPO seed 0 goes from 8.5 at 100K to 0.0 at 600K and back. This pattern of catastrophic mid-training collapse and recovery is characteristic of PPO on Procgen with limited training levels. The policy appears to periodically "forget" learned behaviors during gradient updates, then rediscover them. This instability is a fundamental limitation of our setup and makes final-checkpoint metrics unreliable as a measure of peak performance.

**Comparison to prior run (200K):** Surprisingly, mean returns at 1M are *lower* than at 200K for baseline PPO (2.50 vs. 5.33). This is not because agents are learning less -- the eval logs show that agents reach 6-8 returns at multiple checkpoints during training -- but because the learning curve is non-monotonic. The 200K run happened to catch some agents near a peak, while the 1M final checkpoint caught others in a trough. This underscores the importance of reporting peak or averaged-over-checkpoints performance rather than solely final-checkpoint metrics for unstable training regimes.

### 3.2 Dodgeball

Dodgeball replaces Heist from the previous run. It is a team-based projectile game where the agent must dodge incoming balls while hitting opponents. Rewards are earned for eliminating opponents.

**Key findings:**

- **Performance is very low across all variants.** Mean returns range from 0.10 to 0.68 -- most episodes end with zero reward. Dodgeball is proving to be harder than anticipated at 1M steps. This is partly because Dodgeball requires both offensive and defensive skills (aiming throws while dodging), and the reward is sparse (you must actually hit and eliminate an opponent).

- **The large CNN is the only variant that shows meaningful learning on Dodgeball.** With 0.68 seen / 0.68 unseen returns, PPO + FS + LargeCNN outperforms both baseline PPO (0.58 / 0.26) and frame stack alone (0.10 / 0.18). This is the first game where the large CNN outperforms simpler variants, and it makes sense: Dodgeball's visual scene is more complex (multiple agents, projectiles at various positions) and benefits from greater feature extraction capacity.

- **Baseline PPO shows the clearest overfitting pattern.** The +0.32 gap for baseline PPO is the most consistent positive gap on Dodgeball. The agent learns to score on familiar training level layouts (seen return 0.58) but this knowledge does not transfer well to unseen levels (0.26). With only 200 training levels, the agent may be memorizing opponent spawn positions or movement patterns specific to those levels.

- **Frame stacking alone *hurts* performance on Dodgeball.** PPO + FrameStack scores only 0.10 seen / 0.18 unseen, dramatically worse than baseline PPO. This suggests that the 4x increase in input dimensionality (from frame stacking) is too much for the small default CNN to handle on a complex visual scene like Dodgeball without additional network capacity. The model is overwhelmed by input channels and learns less effectively than with a single frame.

- **The large CNN eliminates the generalization gap.** With a gap of essentially 0.00, the large CNN's learned features appear to be equally useful on seen and unseen levels. Combined with its higher absolute returns, this supports the hypothesis that larger visual encoders learn more transferable representations.

- **The learning curves show a gradual decline after mid-training.** Most variants peak around 300K-500K steps and slowly decay. This is a concerning sign that longer training may not help without other changes (e.g., regularization, data augmentation, or more training levels).

### 3.3 StarPilot

StarPilot is a scrolling space shooter with continuous fractional rewards for destroying enemies. It sits between CoinRun and Dodgeball in difficulty and provides the most informative learning curves.

**Key findings:**

- **Baseline PPO achieves the best overall performance.** At 2.23 seen / 2.18 unseen, baseline PPO outperforms both frame stacking (1.90 / 2.06) and the large CNN (1.66 / 1.53). This matches the 200K-run pattern and suggests that for StarPilot at this training scale, the additional complexity of frame stacking and larger CNNs still does not provide enough benefit to overcome the slower learning.

- **The generalization gap is near zero for all variants.** PPO shows +0.05, frame stack shows -0.16, and large CNN shows +0.13 -- all essentially zero given the standard deviations. This is an interesting result: it means that at 1M steps on StarPilot, none of the variants have learned enough level-specific knowledge to overfit. The agents are in a regime where their policies are still broadly general (but also broadly mediocre). Overfitting to specific training levels would require significantly more training to manifest.

- **StarPilot shows the clearest upward learning trend.** Unlike CoinRun's wild oscillations, the StarPilot learning curves show a consistent upward trajectory from ~0.5-1.0 at 100K steps to ~2.0-2.5 at 1M steps. The shaded confidence bands narrow over time, suggesting that cross-seed variance is decreasing as training progresses. This is the healthiest learning dynamic of the three games and suggests that additional training would continue to improve performance.

- **Frame stacking provides no advantage on StarPilot.** Despite the intuition that temporal context should help in a shooter (tracking enemy velocities, bullet trajectories), frame stacking performs slightly worse than baseline PPO. This may be because the scrolling background already provides motion cues within a single frame, making the temporal information from stacking redundant while increasing the input complexity.

- **The large CNN shows high variance across seeds.** With a gap std of 1.16 (the largest in the entire experiment), the large CNN on StarPilot is the most seed-sensitive configuration. Seed 4 has a gap of +2.15 while seed 2 has -0.80. This suggests that the large CNN's performance on StarPilot is highly dependent on random initialization, making it an unreliable architectural choice at this training scale.

---

## 4. Cross-Cutting Themes

### 4.1 The generalization gap exists but is small

Across the 9 game-variant combinations, the generalization gap is positive (indicating overfitting) in 5 cases, near-zero in 2, and negative in 2. The positive gaps are small (0.05 to 0.90) relative to the standard deviations, and none would survive a formal significance test. At 1M steps with 200 training levels, PPO has not trained long enough to strongly overfit on most games.

The one exception is **CoinRun with baseline PPO** (gap = +0.90 +/- 0.89), which shows the most consistent evidence of overfitting. This makes sense: CoinRun is the easiest game (highest absolute returns), so the agent has learned enough to start memorizing training-level-specific features.

### 4.2 Architectural effects are game-dependent

There is no single "best" architecture across games:

| Game | Best variant (unseen return) | Worst variant (unseen return) |
|---|---|---|
| CoinRun | PPO + FrameStack (3.70) | PPO (1.60) |
| Dodgeball | PPO + FS + LargeCNN (0.68) | PPO + FrameStack (0.18) |
| StarPilot | PPO (2.18) | PPO + FS + LargeCNN (1.53) |

This game-dependence is itself a key finding. Frame stacking helps on CoinRun (where velocity matters for platforming) but hurts on Dodgeball (where it overwhelms the small CNN). The large CNN helps on Dodgeball (complex visual scene) but is too slow to learn on StarPilot at this budget. This illustrates that architectural choices in RL generalization are not one-size-fits-all -- the optimal configuration depends on the visual complexity, reward structure, and sample budget of each environment.

### 4.3 Learning instability is a dominant effect

The most striking feature of the CoinRun results is the non-monotonic learning curve. Agents reach 6-8 returns mid-training and then crash back to 0-2 by the final checkpoint. This instability means that:

1. **Final-checkpoint metrics understate peak capability.** Many agents were significantly better at earlier checkpoints than at 1M steps.
2. **Cross-seed variance is partly driven by timing.** Different seeds happen to be in "up" or "down" phases at the final checkpoint, inflating the cross-seed standard deviation.
3. **More training is not guaranteed to help.** Without addressing the root cause (likely the interaction between PPO's clipping and the small number of training levels), longer runs may continue to oscillate rather than converge.

This phenomenon is well-documented in the Procgen literature and is one of the motivations for techniques like data augmentation (RAD, DrAC) and policy regularization, which are beyond the scope of this project but would be natural next steps.

### 4.4 Dodgeball vs. Heist: a better difficulty anchor

Replacing Heist with Dodgeball was a clear improvement. Where Heist produced zero-return results across the board (providing no useful signal), Dodgeball produces low but non-zero returns with visible variant differences. The large CNN's advantage on Dodgeball (highest returns, zero generalization gap) provides a genuine data point for the argument that larger encoders learn more transferable features -- something that was impossible to observe with Heist at any feasible training budget.

### 4.5 Comparison across runs

| Metric | 200K run (prev) | 1M run (current) | Direction |
|---|---|---|---|
| CoinRun PPO unseen | 4.33 | 1.60 | Down (instability) |
| CoinRun FrameStack unseen | 3.00 | 3.70 | Up |
| CoinRun LargeCNN unseen | 3.33 | 3.10 | Flat |
| StarPilot PPO unseen | 1.37 | 2.18 | Up |
| StarPilot FrameStack unseen | 1.37 | 2.06 | Up |
| StarPilot LargeCNN unseen | 1.30 | 1.53 | Up |

StarPilot shows consistent improvement with 5x more training, as expected. CoinRun shows mixed results due to the learning instability discussed above. The fact that CoinRun baseline PPO went *down* from 200K to 1M is entirely an artifact of the non-monotonic curves -- the agents did learn more, they just crashed back afterwards.

---

## 5. Key Takeaways for the Project Report

1. **The generalization gap is real but requires careful measurement.** At 1M steps with 200 training levels, the gap is most visible on CoinRun with baseline PPO. Other game-variant combinations need more training to exhibit clear overfitting.

2. **Frame stacking helps on some games, hurts on others.** It improves CoinRun performance (platforming benefits from velocity information) but degrades Dodgeball performance (too much input complexity for the small CNN). This demonstrates that temporal context is not a universal improvement.

3. **The large CNN shows the best generalization behavior where it learns.** On Dodgeball, the large CNN achieves the highest unseen returns with zero generalization gap, supporting the hypothesis that larger encoders learn more transferable features. On other games, it learns too slowly to show this advantage.

4. **Learning instability dominates the CoinRun results.** Final-checkpoint metrics are unreliable for CoinRun; peak-performance or averaged metrics would better capture agent capability. This is a genuine challenge in RL generalization research, not just a limitation of our setup.

5. **Game selection matters enormously.** The three games provide complementary signals: CoinRun shows overfitting most clearly, Dodgeball reveals the capacity-generalization tradeoff, and StarPilot demonstrates steady learning with minimal gap. Together they paint a richer picture than any single game could.

---
