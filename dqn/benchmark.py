"""Measure the real obstacle-driving pipeline, with a bounded CUDA benchmark."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.dqn_agent import DQNAgent
from configs.config import AGENT_CONFIG, ENV_CONFIG
from environment.carracing_env import CarRacingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="Local smoke checks only")
    args = parser.parse_args()
    if not np.isfinite(args.minutes) or not 0 < args.minutes <= 10:
        parser.error("Benchmark budget must be greater than zero and at most 10 minutes")
    if not 0 <= args.seed < 2**32 or args.warmup_steps < 0:
        parser.error("Invalid seed or warmup step count")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required. Select T4/A100 in Colab before benchmarking.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**ENV_CONFIG, "scenario": "forest", "difficulty": "normal", "weather": "dry"}
    agent = DQNAgent(AGENT_CONFIG)
    env = CarRacingEnv(config)
    cuda = agent.device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU_SMOKE_ONLY"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    budget_start = time.monotonic()
    deadline = budget_start + args.minutes * 60
    state = env.reset(seed=args.seed)
    episode, episode_reward = 0, 0.
    completed_rewards, completed_collisions, completed_progress = [], [], []
    warmup_steps = 0

    def step():
        nonlocal state, episode, episode_reward
        start = time.monotonic()
        action = agent.select_action(state)
        selected = time.monotonic()
        next_state, reward, done, info = env.step(action)
        stepped = time.monotonic()
        agent.remember(state, action, reward, next_state, info["terminated"])
        loss = agent.learn()
        learned = time.monotonic()
        state = next_state
        episode_reward += reward
        reset_seconds = 0.
        if done:
            completed_rewards.append(episode_reward)
            completed_collisions.append(info["collisions"])
            completed_progress.append(info["progress"])
            episode += 1
            episode_reward = 0.
            reset_start = time.monotonic()
            state = env.reset(seed=(args.seed + episode) % 2**32)
            reset_seconds = time.monotonic() - reset_start
        return (selected-start, stepped-selected, learned-stepped, reset_seconds, loss is not None)

    print(f"[Benchmark] {gpu_name} | batch={agent.batch_size} | forest/normal/dry | budget={args.minutes}min", flush=True)
    try:
        # Same warmup on both GPUs; includes replay fill and initial CUDA kernels.
        while warmup_steps < args.warmup_steps and time.monotonic() < deadline:
            step()
            warmup_steps += 1
        if cuda:
            torch.cuda.synchronize()
        measured_start = time.monotonic()
        initial_updates = agent.step_cnt
        measured_steps = 0
        action_seconds = env_seconds = learn_seconds = reset_seconds = 0.
        last_report = measured_start
        while time.monotonic() < deadline:
            action_time, env_time, learn_time, reset_time, _ = step()
            measured_steps += 1
            action_seconds += action_time
            env_seconds += env_time
            learn_seconds += learn_time
            reset_seconds += reset_time
            now = time.monotonic()
            if now-last_report >= 30:
                elapsed = now-measured_start
                print(f"[Benchmark] measured={elapsed:.1f}s steps={measured_steps} "
                      f"steps/s={measured_steps/elapsed:.2f} updates/s={(agent.step_cnt-initial_updates)/elapsed:.2f} "
                      f"episodes={episode}", flush=True)
                last_report = now
        if cuda:
            torch.cuda.synchronize()
        measured_seconds = time.monotonic()-measured_start
        result = dict(gpu=gpu_name, torch_version=torch.__version__, batch_size=agent.batch_size,
            cpu_threads=torch.get_num_threads(), seed=args.seed, environment=config,
            requested_minutes=args.minutes, warmup_steps=warmup_steps,
            total_elapsed_seconds=time.monotonic()-budget_start, measured_seconds=measured_seconds,
            measured_agent_steps=measured_steps, measured_optimizer_steps=agent.step_cnt-initial_updates,
            agent_steps_per_second=measured_steps/measured_seconds if measured_seconds else 0,
            optimizer_steps_per_second=(agent.step_cnt-initial_updates)/measured_seconds if measured_seconds else 0,
            action_seconds=action_seconds, environment_seconds=env_seconds,
            learning_seconds=learn_seconds, reset_seconds=reset_seconds,
            completed_episodes=episode, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/2**20 if cuda else 0,
            mean_completed_reward=float(np.mean(completed_rewards)) if completed_rewards else None,
            mean_completed_progress=float(np.mean(completed_progress)) if completed_progress else None,
            mean_completed_collisions=float(np.mean(completed_collisions)) if completed_collisions else None,
            note="Throughput benchmark, not proof of a solved driving task. One agent step repeats up to four physics frames.")
        (args.output/"benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        agent.save(str(args.output/"benchmark_model.pth"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
