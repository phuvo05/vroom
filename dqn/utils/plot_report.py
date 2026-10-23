"""
Unified single-column figures for the report (3.5 in wide, 7.5 pt text).

Usage:
    python dqn/utils/plot_report.py dqn/logs/training_log.csv fig_double_dqn.png
Any training_log.csv with the columns episode,reward,avg_reward,epsilon
(and optionally loss) can be plotted, so the same style can be used for every
algorithm in the comparison.
"""

import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"reward": "#4682B4", "loss": "#E4572E", "epsilon": "#2E8B57"}


def make_figure(csv_path, out, panel_height=0.95):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: [float(r[k]) for r in rows]
    ep, rew, avg, eps = col("episode"), col("reward"), col("avg_reward"), col("epsilon")
    has_loss = "loss" in rows[0]

    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Serif"})
    n = 3 if has_loss else 2
    fig, ax = plt.subplots(n, 1, figsize=(3.5, panel_height * n + 0.35), sharex=True)

    ax[0].plot(ep, rew, color=COLOR["reward"], alpha=0.3, lw=0.6, label="Episode reward")
    ax[0].plot(ep, avg, color=COLOR["reward"], lw=1.4, label="Avg-100")
    ax[0].axhline(0, color="gray", ls="--", lw=0.5)
    ax[0].set_ylabel("Reward")
    ax[0].legend(loc="lower right", framealpha=0.9)
    i = 1
    if has_loss:
        ax[i].plot(ep, col("loss"), color=COLOR["loss"], lw=0.8, label="Loss")
        ax[i].set_ylabel("Loss"); ax[i].legend(loc="lower right", framealpha=0.9); i += 1
    ax[i].plot(ep, eps, color=COLOR["epsilon"], lw=1.0, label="Epsilon")
    ax[i].set_ylabel("Epsilon"); ax[i].set_xlabel("Episode")
    ax[i].legend(loc="upper right", framealpha=0.9)
    for a in ax:
        a.grid(True, alpha=0.3, lw=0.4)
        a.tick_params(length=2, width=0.5)
        for s in a.spines.values(): s.set_linewidth(0.5)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"[plot_report] Saved -> {out}")


if __name__ == "__main__":
    make_figure(sys.argv[1], sys.argv[2])

