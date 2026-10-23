"""
Config cho CarRacing-v3 DQN
"""

# ── Environment ───────────────────────────────────────────────────────
ENV_CONFIG = {
    "img_size":    84,
    "frame_stack": 4,
    "skip_frames": 4,        # lặp action 4 lần mỗi step
    "render_mode": "rgb_array",
}

# ── Agent ─────────────────────────────────────────────────────────────
AGENT_CONFIG = {
    "state_dim":     (4, 84, 84),
    "action_size":   5,

    "gamma":         0.99,
    "lr":            1e-4,
    "batch_size":    32,
    "memory_size":   10_000,
    "target_update": 1_000,

    "epsilon_start": 1.0,
    "epsilon_end":   0.05,
    "epsilon_decay": 0.9999,   # ← đổi từ 0.9995 thành 0.9999
}

TRAIN_CONFIG = {
    "num_episodes":    1000,   # ← tăng từ 500 lên 1000
    "save_interval":   50,
    "checkpoint_dir":  "checkpoints",
    "log_dir":         "logs",
}
