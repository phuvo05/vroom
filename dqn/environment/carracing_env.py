"""Frame-stacked CarRacing with optional physical obstacle challenges."""
from collections import deque

import cv2
import gymnasium as gym
import numpy as np

from environment.challenge_racing import ChallengeCarRacing

NUM_ACTIONS = 5


class CarRacingEnv:
    def __init__(self, config):
        self.img_size = config.get("img_size", 84)
        self.frame_stack = config.get("frame_stack", 4)
        self.skip_frames = config.get("skip_frames", 4)
        self.render_mode = config.get("render_mode", "rgb_array")
        self.scenario = config.get("scenario", "classic")
        self.difficulty = config.get("difficulty", "normal")
        self.weather = config.get("weather", "dry")
        self.action_size = NUM_ACTIONS
        self.state_dim = (self.frame_stack, self.img_size, self.img_size)
        self._env = gym.wrappers.TimeLimit(
            ChallengeCarRacing(scenario=self.scenario, difficulty=self.difficulty,
                               weather=self.weather, render_mode=self.render_mode),
            max_episode_steps=config.get("max_episode_steps", 1000),
        )
        self._frames = deque(maxlen=self.frame_stack)
        self._neg_reward_cnt = 0
        self.last_info = {}

    @property
    def base_env(self):
        return self._env.unwrapped

    def reset(self, seed=None):
        obs, _ = self._env.reset(seed=seed)
        self._neg_reward_cnt = 0
        self._frames.clear()
        frame = self._preprocess(obs)
        self._frames.extend([frame] * self.frame_stack)
        self.last_info = dict(collisions=0, damage=0, speed_kmh=0., progress=0.,
                              obstacle_count=len(self.base_env.obstacles),
                              car_position=tuple(self.base_env.car.hull.position))
        return self._get_state()

    def step(self, action_idx):
        total_reward = 0.
        terminated = truncated = False
        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self._env.step(action_idx)
            total_reward += reward
            if terminated or truncated:
                break
        self._frames.append(self._preprocess(obs))
        self._neg_reward_cnt = self._neg_reward_cnt + 1 if total_reward < 0 else 0
        # A deliberate task failure, unlike reaching an external time limit.
        if self._neg_reward_cnt > 25 and not (terminated or truncated):
            terminated = True
            total_reward -= 50
            info["end_reason"] = "stuck"
        if truncated:
            info["end_reason"] = "time_limit"
        info.update(terminated=terminated, truncated=truncated)
        self.last_info = info
        return self._get_state(), total_reward, terminated or truncated, info

    def render(self):
        return self._env.render()

    def map_data(self):
        return {"track": [(t[2], t[3]) for t in self.base_env.track],
                "obstacles": [tuple(o.body.position) for o in self.base_env.obstacles]}

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.

    def _get_state(self):
        return np.array(self._frames, dtype=np.float32)

    def close(self):
        self._env.close()
