"""Record greedy rollouts of a trained checkpoint to an mp4 (headless, no Tkinter).
usage: python dqn/utils/record_demo.py CKPT OUT.mp4 SEED [SEED ...]
needs: pip install imageio imageio-ffmpeg
Uses the repo's own env wrapper + agent; only adds frame capture + HUD overlay."""
import sys, os, json
import numpy as np, cv2, imageio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from environment.carracing_env import CarRacingEnv
from agents.dqn_agent import DQNAgent
from configs.config import ENV_CONFIG, AGENT_CONFIG

ckpt, out, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
ACTIONS = ["Idle", "Steer left", "Steer right", "Accelerate", "Brake"]
FPS = 50

env = CarRacingEnv(ENV_CONFIG)
agent = DQNAgent(AGENT_CONFIG)
import torch
agent.policy_net.load_state_dict(torch.load(ckpt, map_location=agent.device)["policy_net"])  # works with full or stripped checkpoints
agent.policy_net.eval()

frames = []
raw_step = env._env.step
def step_and_capture(a):
    res = raw_step(a)
    frames_buf.append(env._env.render())
    return res
env._env.step = step_and_capture

def hud(img, ep, seed, step, reward, action):
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 34), (13, 22, 38), -1)
    img = cv2.addWeighted(ov, 0.7, img, 0.3, 0)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Double DQN + Dueling | seed {seed}", (10, 23), f, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"reward {reward:7.1f}", (w - 330, 23), f, 0.55, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"action: {action}", (w - 160, 23), f, 0.55, (255, 200, 120), 1, cv2.LINE_AA)
    return img

summary = []
for seed in seeds:
    frames_buf = []
    state = env.reset(seed=seed)
    frames_buf.append(env._env.render())
    total, done, n, act = 0.0, False, 0, 0
    per_frame = []   # (reward_so_far, action) aligned with frames_buf
    per_frame.append((0.0, 0))
    while not done:
        act = agent.select_action(state, training=False)
        k0 = len(frames_buf)
        state, r, done, _ = env.step(act)
        total += r; n += 1
        for _ in range(len(frames_buf) - k0):
            per_frame.append((total, act))
    print(f"seed {seed}: reward={total:.1f} steps={n} frames={len(frames_buf)}", flush=True)
    summary.append({"seed": seed, "reward": total, "steps": n, "frames": len(frames_buf)})
    for i, fr in enumerate(frames_buf):
        rw, a = per_frame[min(i, len(per_frame) - 1)]
        frames.append(hud(fr, 0, seed, i, rw, ACTIONS[a]))
    # short hold on last frame
    frames.extend([frames[-1]] * (FPS // 2))

w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=None, ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart"], macro_block_size=2)
for fr in frames: w.append_data(fr)
w.close()
imageio.imwrite(out.replace(".mp4", "_poster.png"), frames[len(frames) // 3])
json.dump(summary, open(out.replace(".mp4", ".json"), "w"), indent=1)
env.close()
