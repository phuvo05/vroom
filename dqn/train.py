"""Train/evaluate the same physical challenges used by demo_ui.py."""
import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from environment.carracing_env import CarRacingEnv
from environment.challenge_racing import SCENARIOS, DIFFICULTIES, WEATHER
from agents.dqn_agent import DQNAgent
from configs.config import ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG
from utils.logger import TrainingLogger

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--resume", type=str)
    source.add_argument("--warm-start", type=str, help="Transfer policy weights with fresh exploration/optimizer")
    source.add_argument("--eval", type=str)
    p.add_argument("--scenario", choices=SCENARIOS, default=ENV_CONFIG["scenario"])
    p.add_argument("--difficulty", choices=DIFFICULTIES, default="normal")
    p.add_argument("--weather", choices=WEATHER, default="dry")
    p.add_argument("--episodes", type=int, default=TRAIN_CONFIG["num_episodes"])
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", help="Evaluate without opening a pygame window")
    p.add_argument("--max-minutes", type=float, help="Stop training and save latest.pth at this wall-clock limit")
    p.add_argument("--save-seconds", type=float, default=120, help="Periodic latest.pth interval")
    p.add_argument("--output", type=Path, help="New run directory; existing directories are never overwritten")
    args = p.parse_args()
    if args.episodes <= 0 or args.eval_episodes <= 0 or not 0 <= args.seed <= 2**32 - 1:
        p.error("Episode counts must be positive and seed must be between 0 and 4294967295")
    if (args.max_minutes is not None and args.max_minutes <= 0) or args.save_seconds <= 0:
        p.error("Time limits and checkpoint intervals must be positive")
    return args


def run_evaluation(agent, config, seeds):
    env = CarRacingEnv(config)
    rewards, collisions, progress = [], [], []
    try:
        for seed in seeds:
            state, total, done = env.reset(seed=seed), 0., False
            while not done:
                state, reward, done, info = env.step(agent.select_action(state, training=False))
                total += reward
            rewards.append(total)
            collisions.append(info["collisions"])
            progress.append(info["progress"])
            print(f"[Eval] seed={seed} reward={total:.1f} collisions={info['collisions']} "
                  f"progress={info['progress']:.1%} damage={info['damage']}%")
        result = dict(mean_reward=float(np.mean(rewards)), std_reward=float(np.std(rewards)),
                      mean_collisions=float(np.mean(collisions)), mean_progress=float(np.mean(progress)))
        print(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


def train():
    args = parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = {**ENV_CONFIG, "scenario": args.scenario, "difficulty": args.difficulty, "weather": args.weather}
    agent = DQNAgent(AGENT_CONFIG)
    if args.eval:
        agent.load(args.eval)
        config["render_mode"] = "rgb_array" if args.headless else "human"
        run_evaluation(agent, config, [(args.seed + i) % 2**32 for i in range(args.eval_episodes)])
        return
    if args.resume:
        agent.load(args.resume)
    elif args.warm_start:
        agent.load_weights(args.warm_start)
        agent.epsilon = .5  # Old track weights need exploration around new hazards.
    run_dir = args.output or ROOT / "runs" / f"{args.scenario}-{args.weather}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"environment": config, "agent": AGENT_CONFIG,
        "episodes": args.episodes, "seed": args.seed, "resume": args.resume, "warm_start": args.warm_start,
        "max_minutes": args.max_minutes, "save_seconds": args.save_seconds}, indent=2), encoding="utf-8")
    logger = TrainingLogger(str(run_dir / "logs"))
    env = CarRacingEnv(config)
    stop_requested = False

    def stop(sig, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    window, best_avg = deque(maxlen=100), -np.inf
    training_started = time.monotonic()
    deadline = training_started + args.max_minutes * 60 if args.max_minutes else float("inf")
    last_save = training_started
    last_episode = 0
    interrupted = False

    def save_latest():
        # Atomic replacement protects the last complete checkpoint on interruption.
        pending = checkpoint_dir / "latest.pending.pth"
        agent.save(str(pending))
        pending.replace(checkpoint_dir / "latest.pth")

    print(f"[Train] {args.scenario} / {args.difficulty} / {args.weather} -> {run_dir}")
    try:
        for ep in range(1, args.episodes + 1):
            if stop_requested or time.monotonic() >= deadline:
                interrupted = True
                break
            state = env.reset(seed=(args.seed + ep - 1) % 2**32)
            total, losses, done, started = 0., [], False, time.monotonic()
            while not done and not stop_requested:
                if time.monotonic() >= deadline:
                    stop_requested = True
                    break
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                # External time limits still bootstrap; real failures do not.
                agent.remember(state, action, reward, next_state, info["terminated"])
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                state, total = next_state, total + reward
                if time.monotonic() - last_save >= args.save_seconds:
                    save_latest()
                    last_save = time.monotonic()
            if stop_requested:
                interrupted = True
                break
            last_episode = ep
            window.append(total)
            avg = float(np.mean(window))
            logger.log(ep, total, avg, float(np.mean(losses)) if losses else 0., agent.epsilon)
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
                metrics.write(json.dumps(dict(episode=ep, seed=args.seed + ep - 1, reward=total,
                    avg_reward=avg, collisions=info["collisions"], progress=info["progress"],
                    damage=info["damage"], epsilon=agent.epsilon,
                    elapsed_seconds=time.monotonic() - training_started)) + "\n")
            print(f"Ep {ep}/{args.episodes} reward={total:.1f} avg100={avg:.1f} "
                  f"hits={info['collisions']} progress={info['progress']:.1%} "
                  f"epsilon={agent.epsilon:.3f} t={time.monotonic()-started:.1f}s")
            if ep % TRAIN_CONFIG["save_interval"] == 0:
                agent.save(str(checkpoint_dir / f"ep_{ep:04d}.pth"))
            if avg > best_avg:
                best_avg = avg
                agent.save(str(checkpoint_dir / "best.pth"))
    finally:
        save_latest()
        (run_dir / "summary.json").write_text(json.dumps(dict(
            completed_episodes=last_episode, optimizer_steps=agent.step_cnt,
            elapsed_seconds=time.monotonic() - training_started, stopped_early=interrupted,
            scenario=args.scenario, difficulty=args.difficulty, weather=args.weather,
            epsilon=agent.epsilon), indent=2), encoding="utf-8")
        env.close()
        logger.save_plots()
    print(f"[Train] Outputs: {run_dir}")


if __name__ == "__main__":
    train()
