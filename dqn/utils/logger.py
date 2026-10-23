import os, csv
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False


class TrainingLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir  = log_dir
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self.episodes = []; self.rewards = []; self.avg_rewards = []
        self.losses   = []; self.epsilons = []
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode","reward","avg_reward","loss","epsilon"])

    def log(self, episode, reward, avg_reward, loss, epsilon):
        self.episodes.append(episode); self.rewards.append(reward)
        self.avg_rewards.append(avg_reward); self.losses.append(loss)
        self.epsilons.append(epsilon)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([episode, f"{reward:.2f}", f"{avg_reward:.2f}",
                                    f"{loss:.6f}", f"{epsilon:.4f}"])

    def save_plots(self):
        out = os.path.join(self.log_dir, "training_curves.png")
        plot_curves(self.episodes, self.rewards, self.avg_rewards,
                    self.losses, self.epsilons, out)


def plot_curves(episodes, rewards, avg_rewards, losses, epsilons, out):
    if not MPL_OK: return
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 12})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("CarRacing-v3 DQN Training", fontsize=16, fontweight="bold")
    axes[0].plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")
    axes[0].plot(episodes, avg_rewards, color="steelblue", lw=2, label="Avg-100")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("Reward"); axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(episodes, losses, color="tomato", label="Loss")
    axes[1].set_ylabel("Loss"); axes[1].legend(loc="lower right"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(episodes, epsilons, color="seagreen", label="Epsilon")
    axes[2].set_ylabel("Epsilon"); axes[2].set_xlabel("Episode")
    axes[2].legend(loc="upper right"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()
    print(f"[Logger] Saved → {out}")


def replot_from_csv(csv_path, out=None):
    """Regenerate the training curves from an existing training_log.csv (does not modify it)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k, cast=float: [cast(r[k]) for r in rows]
    out = out or os.path.join(os.path.dirname(csv_path), "training_curves.png")
    plot_curves(col("episode", int), col("reward"), col("avg_reward"),
                col("loss"), col("epsilon"), out)
