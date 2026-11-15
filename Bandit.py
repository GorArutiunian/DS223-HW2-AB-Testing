"""
DS 223 Homework 2 — A/B Testing with multi-armed bandits.

Implements Epsilon-Greedy (epsilon decay 1/t) and Thompson Sampling
(Gaussian rewards, known precision) for four advertisement options.

Run:
    python Bandit.py

Instead of bare print(), messages use loguru log levels as required.
"""

from __future__ import annotations

############################### LOGGER
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# Experiment constants from the assignment (exact names + aliases)
Bandit_Reward: List[float] = [1, 2, 3, 4]
NumberOfTrials: int = 20000
BANDIT_REWARDS: List[float] = list(Bandit_Reward)
NUMBER_OF_TRIALS: int = NumberOfTrials

# Algorithm hyperparameters (assignment: values are up to us)
INITIAL_EPSILON: float = 1.0  # epsilon_t = INITIAL_EPSILON / t  (decay by 1/t)
KNOWN_PRECISION: float = 1.0  # tau: reward ~ N(mu, 1/tau)
# Weak prior: means are [1,2,3,4], so N(0,1) is too strong and over-pulls suboptimal arms.
PRIOR_PRECISION: float = 0.0001
RANDOM_SEED: int = 42

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
FIG_DIR = OUTPUT_DIR / "figures"


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


#--------------------------------------#


class Visualization:
    """Create the learning-process and comparison plots required by the report."""

    def __init__(self) -> None:
        """Initialize empty plot buffers filled by ``comparison`` / ``plot1`` / ``plot2``."""
        self.eg_rewards: Optional[np.ndarray] = None
        self.ts_rewards: Optional[np.ndarray] = None
        self.eg_regrets: Optional[np.ndarray] = None
        self.ts_regrets: Optional[np.ndarray] = None
        self.eg_estimates: Optional[np.ndarray] = None
        self.ts_estimates: Optional[np.ndarray] = None

    def plot1(
        self,
        algorithm_name: Optional[str] = None,
        rewards: Optional[np.ndarray] = None,
        true_means: Optional[Sequence[float]] = None,
        estimates_over_time: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Visualize the learning process for each algorithm (linear and log).

        With no arguments, plots both algorithms using data stored on this
        instance (set by ``comparison()``).

        Parameters
        ----------
        algorithm_name : str, optional
            Label used in titles (e.g. ``EpsilonGreedy``).
        rewards : np.ndarray, optional
            Observed reward at each trial, shape ``(T,)``.
        true_means : Sequence[float], optional
            True mean reward of each advertisement / arm.
        estimates_over_time : np.ndarray, optional
            Estimated means after each trial, shape ``(T, K)``.
        save_path : Path, optional
            If given, figure is written here instead of only shown.
        """
        if true_means is None:
            true_means = Bandit_Reward

        # Zero-arg form from the starter: plot both algorithms if data is ready
        if algorithm_name is None and rewards is None:
            if self.eg_rewards is None or self.ts_rewards is None:
                raise RuntimeError("No experiment data yet. Run comparison() first.")
            self.plot1(
                "EpsilonGreedy",
                self.eg_rewards,
                true_means,
                estimates_over_time=self.eg_estimates,
                save_path=FIG_DIR / "plot1_epsilon_greedy.png",
            )
            self.plot1(
                "ThompsonSampling",
                self.ts_rewards,
                true_means,
                estimates_over_time=self.ts_estimates,
                save_path=FIG_DIR / "plot1_thompson_sampling.png",
            )
            return

        if rewards is None or algorithm_name is None:
            raise ValueError("plot1 requires algorithm_name and rewards, or stored data.")

        t = np.arange(1, len(rewards) + 1)
        cumulative_avg = np.cumsum(rewards) / t
        optimal = float(np.max(true_means))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{algorithm_name}: learning process", fontsize=14)

        for ax, scale in zip(axes, ("linear", "log")):
            ax.plot(t, cumulative_avg, label="Cumulative average reward", linewidth=1.5)
            ax.axhline(optimal, color="black", linestyle="--", label=f"Optimal mean = {optimal}")
            if estimates_over_time is not None:
                for k, mean in enumerate(true_means):
                    ax.plot(
                        t,
                        estimates_over_time[:, k],
                        alpha=0.7,
                        linewidth=1.0,
                        label=f"Est. mean arm {k} (true={mean})",
                    )
            ax.set_xlabel("Trial")
            ax.set_ylabel("Reward / estimate")
            ax.set_title(f"{scale} scale")
            ax.set_xscale(scale)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot1 for {algorithm_name} to {save_path}")
        plt.close(fig)

    def plot2(
        self,
        eg_rewards: Optional[np.ndarray] = None,
        ts_rewards: Optional[np.ndarray] = None,
        eg_regrets: Optional[np.ndarray] = None,
        ts_regrets: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Compare Epsilon-Greedy and Thompson Sampling cumulative rewards and regrets.

        With no arguments, uses data stored on this instance.

        Parameters
        ----------
        eg_rewards, ts_rewards : np.ndarray, optional
            Per-trial rewards for each algorithm.
        eg_regrets, ts_regrets : np.ndarray, optional
            Per-trial instantaneous regrets for each algorithm.
        save_path : Path, optional
            If given, figure is written here.
        """
        if eg_rewards is None:
            eg_rewards = self.eg_rewards
        if ts_rewards is None:
            ts_rewards = self.ts_rewards
        if eg_regrets is None:
            eg_regrets = self.eg_regrets
        if ts_regrets is None:
            ts_regrets = self.ts_regrets
        if eg_rewards is None or ts_rewards is None or eg_regrets is None or ts_regrets is None:
            raise RuntimeError("No experiment data yet. Run comparison() first.")
        if save_path is None:
            save_path = FIG_DIR / "plot2_comparison.png"

        t = np.arange(1, len(eg_rewards) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Epsilon-Greedy vs Thompson Sampling", fontsize=14)

        axes[0].plot(t, np.cumsum(eg_rewards), label="Epsilon-Greedy", linewidth=1.5)
        axes[0].plot(t, np.cumsum(ts_rewards), label="Thompson Sampling", linewidth=1.5)
        axes[0].set_title("Cumulative reward")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Cumulative reward")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, np.cumsum(eg_regrets), label="Epsilon-Greedy", linewidth=1.5)
        axes[1].plot(t, np.cumsum(ts_regrets), label="Thompson Sampling", linewidth=1.5)
        axes[1].set_title("Cumulative regret")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Cumulative regret")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot2 comparison to {save_path}")
        plt.close(fig)


#--------------------------------------#


class EpsilonGreedy(Bandit):
    """Epsilon-Greedy bandit arm with Gaussian rewards and epsilon decay 1/t.

    Each instance models one advertisement with true mean ``p``. The classmethod-
    style ``experiment`` builds one arm per mean in ``BANDIT_REWARDS`` and runs
    the full adaptive experiment.

    Parameters
    ----------
    p : float
        True mean reward of this advertisement.
    """

    algorithm_name = "EpsilonGreedy"

    def __init__(self, p: float) -> None:
        """Initialize one Gaussian advertisement arm.

        Parameters
        ----------
        p : float
            True mean reward.
        """
        self.p = float(p)
        self.p_estimate = 0.0
        self.N = 0
        # Populated by experiment()
        self.rewards: Optional[np.ndarray] = None
        self.regrets: Optional[np.ndarray] = None
        self.choices: Optional[np.ndarray] = None
        self.estimates_over_time: Optional[np.ndarray] = None
        self.bandits: Optional[List["EpsilonGreedy"]] = None

    def __repr__(self) -> str:
        """Return a readable summary of the arm."""
        return (
            f"EpsilonGreedy(true_mean={self.p}, "
            f"estimate={self.p_estimate:.4f}, pulls={self.N})"
        )

    def pull(self) -> float:
        """Draw one Gaussian reward: ``N(true_mean, 1)``.

        Returns
        -------
        float
            Observed reward.
        """
        return float(np.random.randn() + self.p)

    def update(self, x: float = None) -> None:
        """Update the empirical mean estimate after observing reward ``x``.

        Parameters
        ----------
        x : float
            Observed reward from ``pull``.
        """
        if x is None:
            raise ValueError("update() requires the observed reward x")
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

    def experiment(
        self,
        bandit_rewards: Sequence[float] = None,
        n_trials: int = None,
        initial_epsilon: float = INITIAL_EPSILON,
        seed: Optional[int] = RANDOM_SEED,
    ) -> Tuple[List["EpsilonGreedy"], np.ndarray, np.ndarray, np.ndarray]:
        """Run Epsilon-Greedy for ``n_trials`` with epsilon_t = initial_epsilon / t.

        Parameters
        ----------
        bandit_rewards : Sequence[float], optional
            True means of the advertisements. Defaults to ``BANDIT_REWARDS``.
        n_trials : int, optional
            Number of pulls. Defaults to ``NUMBER_OF_TRIALS``.
        initial_epsilon : float, optional
            Numerator of the 1/t schedule (epsilon_t = initial_epsilon / t).
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        bandits : list of EpsilonGreedy
            One updated arm object per advertisement.
        rewards : np.ndarray
            Observed rewards, shape ``(n_trials,)``.
        regrets : np.ndarray
            Instantaneous regrets vs the best true mean.
        choices : np.ndarray
            Index of the arm pulled at each trial.
        """
        if bandit_rewards is None:
            bandit_rewards = BANDIT_REWARDS
        if n_trials is None:
            n_trials = NUMBER_OF_TRIALS
        if seed is not None:
            np.random.seed(seed)

        bandits = [EpsilonGreedy(mean) for mean in bandit_rewards]
        optimal_mean = float(np.max(bandit_rewards))
        k = len(bandits)

        rewards = np.empty(n_trials, dtype=float)
        regrets = np.empty(n_trials, dtype=float)
        choices = np.empty(n_trials, dtype=int)
        estimates = np.empty((n_trials, k), dtype=float)

        logger.info(
            f"Starting Epsilon-Greedy: trials={n_trials}, "
            f"means={list(bandit_rewards)}, epsilon_t={initial_epsilon}/t"
        )

        for t in range(1, n_trials + 1):
            epsilon = initial_epsilon / t
            if np.random.random() < epsilon:
                j = int(np.random.randint(0, k))
            else:
                estimates_now = np.array([b.p_estimate for b in bandits])
                # Break ties at random among current greedy arms
                j = int(np.random.choice(np.flatnonzero(estimates_now == estimates_now.max())))

            x = bandits[j].pull()
            bandits[j].update(x)

            rewards[t - 1] = x
            regrets[t - 1] = optimal_mean - bandits[j].p
            choices[t - 1] = j
            estimates[t - 1] = [b.p_estimate for b in bandits]

        self.bandits = bandits
        self.rewards = rewards
        self.regrets = regrets
        self.choices = choices
        self.estimates_over_time = estimates

        logger.success(
            f"Epsilon-Greedy finished | "
            f"cumulative reward={rewards.sum():.2f} | "
            f"cumulative regret={regrets.sum():.2f}"
        )
        return bandits, rewards, regrets, choices

    def report(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """Store trial-level rewards in CSV and log reward / regret summaries.

        Parameters
        ----------
        csv_path : Path, optional
            Destination CSV. Defaults to ``outputs/csv/epsilon_greedy_rewards.csv``.

        Returns
        -------
        pd.DataFrame
            Columns: Bandit, Reward, Algorithm.
        """
        if self.rewards is None or self.choices is None or self.regrets is None:
            raise RuntimeError("Call experiment() before report().")

        if csv_path is None:
            csv_path = CSV_DIR / "epsilon_greedy_rewards.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "Bandit": self.choices.astype(int),
                "Reward": self.rewards.astype(float),
                "Algorithm": self.algorithm_name,
            }
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved Epsilon-Greedy rewards to {csv_path}")

        cumulative_reward = float(self.rewards.sum())
        cumulative_regret = float(self.regrets.sum())
        average_reward = float(self.rewards.mean())
        average_regret = float(self.regrets.mean())

        # PDF: cumulative; starter comments: average — report both via loguru
        logger.info(f"[EpsilonGreedy] Cumulative reward = {cumulative_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Cumulative regret = {cumulative_regret:.4f}")
        logger.info(f"[EpsilonGreedy] Average reward = {average_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Average regret = {average_regret:.4f}")

        if self.bandits is not None:
            for i, b in enumerate(self.bandits):
                logger.debug(
                    f"[EpsilonGreedy] Arm {i}: true={b.p}, "
                    f"estimate={b.p_estimate:.4f}, pulls={b.N}"
                )

        return df


#--------------------------------------#


class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with known precision.

    Prior on the unknown mean is Normal with mean 0 and precision
    ``PRIOR_PRECISION``. Likelihood precision ``tau`` is known
    (``KNOWN_PRECISION``). Posterior updates are Normal-Normal conjugate.

    Parameters
    ----------
    p : float
        True mean reward of this advertisement (used only to simulate pulls).
    precision : float, optional
        Known reward precision tau. Defaults to ``KNOWN_PRECISION``.
    """

    algorithm_name = "ThompsonSampling"

    def __init__(self, p: float, precision: float = KNOWN_PRECISION) -> None:
        """Initialize one Gaussian advertisement arm for Thompson Sampling.

        Parameters
        ----------
        p : float
            True mean reward.
        precision : float, optional
            Known observation precision tau = 1 / variance.
        """
        self.p = float(p)
        self.tau = float(precision)
        # Posterior parameters for the mean: N(m, 1/lambda_)
        self.m = 0.0
        self.lambda_ = float(PRIOR_PRECISION)
        self.N = 0
        self.sum_x = 0.0
        # Populated by experiment()
        self.rewards: Optional[np.ndarray] = None
        self.regrets: Optional[np.ndarray] = None
        self.choices: Optional[np.ndarray] = None
        self.estimates_over_time: Optional[np.ndarray] = None
        self.bandits: Optional[List["ThompsonSampling"]] = None

    def __repr__(self) -> str:
        """Return a readable summary of the arm and its posterior."""
        return (
            f"ThompsonSampling(true_mean={self.p}, "
            f"posterior_mean={self.m:.4f}, precision={self.lambda_:.4f}, "
            f"pulls={self.N})"
        )

    def pull(self) -> float:
        """Draw one reward from ``N(true_mean, 1/tau)``.

        Returns
        -------
        float
            Observed reward.
        """
        return float(np.random.randn() / np.sqrt(self.tau) + self.p)

    def sample(self) -> float:
        """Draw one candidate mean from the current Normal posterior.

        Returns
        -------
        float
            Sampled mean used for Thompson selection.
        """
        return float(np.random.randn() / np.sqrt(self.lambda_) + self.m)

    def update(self, x: float = None) -> None:
        """Bayes-update the Normal posterior after observing reward ``x``.

        Parameters
        ----------
        x : float
            Observed reward from ``pull``.
        """
        if x is None:
            raise ValueError("update() requires the observed reward x")
        # Conjugate Normal-Normal update with known precision tau
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.sum_x += x

    def experiment(
        self,
        bandit_rewards: Sequence[float] = None,
        n_trials: int = None,
        precision: float = KNOWN_PRECISION,
        seed: Optional[int] = RANDOM_SEED + 1,
    ) -> Tuple[List["ThompsonSampling"], np.ndarray, np.ndarray, np.ndarray]:
        """Run Thompson Sampling for ``n_trials`` with known precision.

        Parameters
        ----------
        bandit_rewards : Sequence[float], optional
            True means of the advertisements. Defaults to ``BANDIT_REWARDS``.
        n_trials : int, optional
            Number of pulls. Defaults to ``NUMBER_OF_TRIALS``.
        precision : float, optional
            Known reward precision shared by all arms.
        seed : int, optional
            RNG seed for reproducibility (offset from EG so paths differ).

        Returns
        -------
        bandits : list of ThompsonSampling
            One updated arm object per advertisement.
        rewards : np.ndarray
            Observed rewards, shape ``(n_trials,)``.
        regrets : np.ndarray
            Instantaneous regrets vs the best true mean.
        choices : np.ndarray
            Index of the arm pulled at each trial.
        """
        if bandit_rewards is None:
            bandit_rewards = BANDIT_REWARDS
        if n_trials is None:
            n_trials = NUMBER_OF_TRIALS
        if seed is not None:
            np.random.seed(seed)

        bandits = [ThompsonSampling(mean, precision=precision) for mean in bandit_rewards]
        optimal_mean = float(np.max(bandit_rewards))
        k = len(bandits)

        rewards = np.empty(n_trials, dtype=float)
        regrets = np.empty(n_trials, dtype=float)
        choices = np.empty(n_trials, dtype=int)
        estimates = np.empty((n_trials, k), dtype=float)

        logger.info(
            f"Starting Thompson Sampling: trials={n_trials}, "
            f"means={list(bandit_rewards)}, known precision tau={precision}"
        )

        for t in range(n_trials):
            samples = np.array([b.sample() for b in bandits])
            j = int(np.random.choice(np.flatnonzero(samples == samples.max())))

            x = bandits[j].pull()
            bandits[j].update(x)

            rewards[t] = x
            regrets[t] = optimal_mean - bandits[j].p
            choices[t] = j
            estimates[t] = [b.m for b in bandits]

        self.bandits = bandits
        self.rewards = rewards
        self.regrets = regrets
        self.choices = choices
        self.estimates_over_time = estimates

        logger.success(
            f"Thompson Sampling finished | "
            f"cumulative reward={rewards.sum():.2f} | "
            f"cumulative regret={regrets.sum():.2f}"
        )
        return bandits, rewards, regrets, choices

    def report(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """Store trial-level rewards in CSV and log reward / regret summaries.

        Parameters
        ----------
        csv_path : Path, optional
            Destination CSV. Defaults to ``outputs/csv/thompson_sampling_rewards.csv``.

        Returns
        -------
        pd.DataFrame
            Columns: Bandit, Reward, Algorithm.
        """
        if self.rewards is None or self.choices is None or self.regrets is None:
            raise RuntimeError("Call experiment() before report().")

        if csv_path is None:
            csv_path = CSV_DIR / "thompson_sampling_rewards.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "Bandit": self.choices.astype(int),
                "Reward": self.rewards.astype(float),
                "Algorithm": self.algorithm_name,
            }
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved Thompson Sampling rewards to {csv_path}")

        cumulative_reward = float(self.rewards.sum())
        cumulative_regret = float(self.regrets.sum())
        average_reward = float(self.rewards.mean())
        average_regret = float(self.regrets.mean())

        logger.info(f"[ThompsonSampling] Cumulative reward = {cumulative_reward:.4f}")
        logger.info(f"[ThompsonSampling] Cumulative regret = {cumulative_regret:.4f}")
        logger.info(f"[ThompsonSampling] Average reward = {average_reward:.4f}")
        logger.info(f"[ThompsonSampling] Average regret = {average_regret:.4f}")

        if self.bandits is not None:
            for i, b in enumerate(self.bandits):
                logger.debug(
                    f"[ThompsonSampling] Arm {i}: true={b.p}, "
                    f"posterior_mean={b.m:.4f}, pulls={b.N}"
                )

        return df


def comparison(
    eg: Optional[EpsilonGreedy] = None,
    ts: Optional[ThompsonSampling] = None,
    viz: Optional[Visualization] = None,
) -> Visualization:
    """Compare Epsilon-Greedy and Thompson Sampling visually and in the logs.

    Matches the starter ``comparison()`` zero-arg form: if ``eg`` / ``ts`` are
    omitted, both experiments are run automatically.

    Parameters
    ----------
    eg : EpsilonGreedy, optional
        Runner that has already executed ``experiment()``.
    ts : ThompsonSampling, optional
        Runner that has already executed ``experiment()``.
    viz : Visualization, optional
        Plot helper; a new instance is created if omitted.

    Returns
    -------
    Visualization
        Plot helper holding both algorithms' results.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if eg is None:
        eg = EpsilonGreedy(p=0.0)
        eg.experiment()
        eg.report()
    if ts is None:
        ts = ThompsonSampling(p=0.0)
        ts.experiment()
        ts.report()

    if eg.rewards is None or ts.rewards is None:
        raise RuntimeError("Run both experiments before comparison().")
    if eg.regrets is None or ts.regrets is None:
        raise RuntimeError("Regret arrays are missing; re-run experiment().")

    if viz is None:
        viz = Visualization()

    viz.eg_rewards = eg.rewards
    viz.ts_rewards = ts.rewards
    viz.eg_regrets = eg.regrets
    viz.ts_regrets = ts.regrets
    viz.eg_estimates = eg.estimates_over_time
    viz.ts_estimates = ts.estimates_over_time

    viz.plot1(
        "EpsilonGreedy",
        eg.rewards,
        Bandit_Reward,
        estimates_over_time=eg.estimates_over_time,
        save_path=FIG_DIR / "plot1_epsilon_greedy.png",
    )
    viz.plot1(
        "ThompsonSampling",
        ts.rewards,
        Bandit_Reward,
        estimates_over_time=ts.estimates_over_time,
        save_path=FIG_DIR / "plot1_thompson_sampling.png",
    )
    viz.plot2(
        eg.rewards,
        ts.rewards,
        eg.regrets,
        ts.regrets,
        save_path=FIG_DIR / "plot2_comparison.png",
    )

    # Combined CSV required by the assignment schema {Bandit, Reward, Algorithm}
    combined = pd.concat(
        [
            pd.DataFrame(
                {
                    "Bandit": eg.choices.astype(int),
                    "Reward": eg.rewards.astype(float),
                    "Algorithm": "EpsilonGreedy",
                }
            ),
            pd.DataFrame(
                {
                    "Bandit": ts.choices.astype(int),
                    "Reward": ts.rewards.astype(float),
                    "Algorithm": "ThompsonSampling",
                }
            ),
        ],
        ignore_index=True,
    )
    combined_path = CSV_DIR / "all_rewards.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    logger.info(f"Saved combined rewards CSV to {combined_path}")

    logger.info(
        f"Comparison | EG cumulative reward={eg.rewards.sum():.2f}, "
        f"TS cumulative reward={ts.rewards.sum():.2f}"
    )
    logger.info(
        f"Comparison | EG cumulative regret={eg.regrets.sum():.2f}, "
        f"TS cumulative regret={ts.regrets.sum():.2f}"
    )
    return viz


if __name__ == "__main__":
    # Starter template: demonstrate loguru levels (not a runtime failure)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    # Non-interactive backend so graders / CI can run without a display
    plt.switch_backend("Agg")

    eg_runner = EpsilonGreedy(p=0.0)
    eg_runner.experiment()
    eg_runner.report()

    ts_runner = ThompsonSampling(p=0.0)
    ts_runner.experiment()
    ts_runner.report()

    comparison(eg_runner, ts_runner, viz=Visualization())

    logger.success("Homework 2 experiment completed successfully.")
