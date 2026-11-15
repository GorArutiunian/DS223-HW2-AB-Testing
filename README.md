# DS 223 Homework 2 — A/B Testing (Multi-Armed Bandits)

Epsilon-Greedy and Thompson Sampling for four advertisement options.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python Bandit.py
```

Outputs:

- `outputs/csv/epsilon_greedy_rewards.csv`
- `outputs/csv/thompson_sampling_rewards.csv`
- `outputs/csv/all_rewards.csv` — columns `{Bandit, Reward, Algorithm}`
- `outputs/figures/plot1_epsilon_greedy.png` — learning process (linear + log)
- `outputs/figures/plot1_thompson_sampling.png`
- `outputs/figures/plot2_comparison.png` — cumulative reward & regret

## Design choices

| Setting | Value | Notes |
|--------|--------|--------|
| `Bandit_Reward` | `[1, 2, 3, 4]` | True Gaussian means |
| `NumberOfTrials` | `20000` | Assignment horizon |
| Epsilon schedule | `ε_t = 1/t` | Decays by `1/t` as required |
| Thompson precision | `τ = 1` | Known observation precision |
| Reward model | Gaussian | Required for “known precision” TS |

Regret at trial `t` is `μ* − μ_{a_t}` (expected regret using true means).

## Bonus

See [BONUS.md](BONUS.md) for a stronger implementation plan.
