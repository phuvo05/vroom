"""
Train DQN trên CarRacing-v3
Usage:
    python train.py                          # train từ đầu
    python train.py --resume checkpoints/ep_0025.pth
    python train.py --eval   checkpoints/best.pth
"""

import os, sys, signal, argparse, time
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from environment.carracing_env import CarRacingEnv
from agents.dqn_agent           import DQNAgent
from utils.logger               import TrainingLogger
from configs.config             import ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval",   type=str, default=None)
    return p.parse_args()


def train():
    args = parse_args()

    env    = CarRacingEnv(ENV_CONFIG)
    agent  = DQNAgent(AGENT_CONFIG)
    logger = TrainingLogger(TRAIN_CONFIG["log_dir"])
    os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)

    if args.resume:
        agent.load(args.resume)

    if args.eval:
        agent.load(args.eval)
        run_evaluation(agent)
        env.close()
        return

    # ── Ctrl+C handler ─────────────────────────────────────────────────
    def save_and_exit(sig, frame):
        print("\n[Train] Lưu checkpoint trước khi thoát...")
        agent.save(os.path.join(TRAIN_CONFIG["checkpoint_dir"], "interrupted.pth"))
        logger.save_plots()
        env.close()
        sys.exit(0)
    signal.signal(signal.SIGINT,  save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # ── Training loop ──────────────────────────────────────────────────
    num_episodes  = TRAIN_CONFIG["num_episodes"]
    save_interval = TRAIN_CONFIG["save_interval"]
    reward_window = deque(maxlen=100)
    best_avg      = -np.inf

    print(f"\n{'='*60}")
    print(f"  CarRacing-v3 DQN  |  Episodes: {num_episodes}")
    print(f"  Device: {agent.device}  |  Frame stack: {ENV_CONFIG['frame_stack']}")
    print(f"{'='*60}\n")

    for ep in range(1, num_episodes + 1):
        state     = env.reset()
        ep_reward = 0.0
        ep_loss   = []
        t0        = time.time()

        done = False
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                ep_loss.append(loss)
            state      = next_state
            ep_reward += reward

        reward_window.append(ep_reward)
        avg_reward = np.mean(reward_window)
        avg_loss   = np.mean(ep_loss) if ep_loss else 0.0
        elapsed    = time.time() - t0

        logger.log(ep, ep_reward, avg_reward, avg_loss, agent.epsilon)
        print(
            f"Ep {ep:4d}/{num_episodes}"
            f"  reward={ep_reward:8.1f}"
            f"  avg100={avg_reward:7.1f}"
            f"  loss={avg_loss:.4f}"
            f"  ε={agent.epsilon:.3f}"
            f"  t={elapsed:.1f}s"
        )

        if ep % save_interval == 0:
            agent.save(os.path.join(TRAIN_CONFIG["checkpoint_dir"], f"ep_{ep:04d}.pth"))

        if avg_reward > best_avg and ep >= 20:
            best_avg = avg_reward
            agent.save(os.path.join(TRAIN_CONFIG["checkpoint_dir"], "best.pth"))
            print(f"  ★ New best avg: {best_avg:.1f}")

    env.close()
    logger.save_plots()
    print("\nTraining hoàn tất!")


def run_evaluation(agent, num_eps=5):
    eval_config = {**ENV_CONFIG, "render_mode": "human"}
    env = CarRacingEnv(eval_config)
    print(f"\n[Eval] Chạy {num_eps} episodes...")
    
    seeds = [42, 123, 777, 999, 2024]  # 5 map khác nhau
    rewards = []
    
    for ep, seed in enumerate(seeds, 1):
        state = env.reset(seed=seed)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        print(f"  Ep {ep} (seed={seed}): reward={ep_reward:.1f}")
    
    print(f"\n[Eval] Mean={np.mean(rewards):.1f}  Std={np.std(rewards):.1f}")
    env.close()


if __name__ == "__main__":
    train()
