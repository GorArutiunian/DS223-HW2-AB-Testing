# Bonus: Better Implementation Plan (10 points)

The current homework solution is a correct, assignment-faithful single-file design. A production-quality redesign would improve correctness checks, reuse, and evaluation as follows.

## 1. Separate simulation from learning

Keep a thin `Environment` (true means, reward sampler, optimal arm) distinct from `Policy` objects (`EpsilonGreedyPolicy`, `ThompsonSamplingPolicy`). Policies should only see rewards, never true means. That prevents accidental leakage (e.g. using true means inside selection) and makes it easy to plug in Bernoulli, Gaussian, or contextual rewards later.

## 2. Shared experiment harness

Run both algorithms through one `run_experiment(env, policy, T, seed)` loop that returns a typed `Result` dataclass: choices, rewards, instantaneous regret, cumulative reward/regret, and final parameter estimates. Today each algorithm duplicates that bookkeeping; a harness removes drift between implementations and guarantees identical metrics.

## 3. Proper evaluation protocol

- Average over many seeds (e.g. 50–100) and plot mean ± confidence bands for cumulative regret.
- Report simple regret (final recommended arm) as well as cumulative regret.
- Add a fixed A/B split baseline and UCB1 so Thompson / ε-greedy gains are interpretable.
- Unit-test conjugate posterior updates against closed-form Normal-Normal formulas.

## 4. Hyperparameter and schedule design

Replace ad-hoc constants with a small config (YAML/CLI): initial ε, precision τ, prior strength, horizon T. Sweep ε schedules (`1/t`, `c/t`, ε-first) and τ under misspecification. Log every run with seed + config hash for reproducibility.

## 5. Software structure

```
src/bandits/
  env.py
  policies/{epsilon_greedy.py,thompson.py}
  metrics.py
  plots.py
tests/
  test_posterior_updates.py
  test_regret_identity.py
```

Use type hints, dataclasses, and structured logging (as with loguru). Persist results as parquet/CSV with a schema contract (`Bandit`, `Reward`, `Algorithm`, `Trial`, `Seed`). Generate figures in CI so Moodle/GitHub reviewers can inspect artifacts without running locally.

## 6. Why this scores the bonus

It keeps the pedagogical algorithms intact while addressing the main weaknesses of the starter: duplicated experiment logic, single-seed conclusions, and tight coupling between the true environment and the learner. That is the natural next step beyond a correct homework submission.
