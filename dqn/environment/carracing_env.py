"""
CarRacing-v3 Environment Wrapper
──────────────────────────────────
Observation : RGB frame (96x96x3) → grayscale 84x84 → stack 4 frames → (4, 84, 84)
Action space: chế độ rời rạc có sẵn của CarRacing-v3 (continuous=False), 5 hành động

Discrete actions (steering / gas / brake do gymnasium áp dụng):
  0  Không làm gì        (0.0,  0.0, 0.0)
  1  Rẽ trái             (-0.6, 0.0, 0.0)
  2  Rẽ phải             (+0.6, 0.0, 0.0)
  3  Tăng tốc (gas)      (0.0,  0.2, 0.0)
  4  Phanh               (0.0,  0.0, 0.8)
"""

import numpy as np
import cv2
import gymnasium as gym
from collections import deque


NUM_ACTIONS = 5


class CarRacingEnv:
    def __init__(self, config):
        self.img_size    = config.get("img_size", 84)
        self.frame_stack = config.get("frame_stack", 4)
        self.skip_frames = config.get("skip_frames", 4)   # action repeat
        self.render_mode = config.get("render_mode", "rgb_array")

        self.action_size = NUM_ACTIONS
        self.state_dim   = (self.frame_stack, self.img_size, self.img_size)

        self._env    = gym.make("CarRacing-v3", render_mode=self.render_mode,
                                continuous=False)
        self._frames = deque(maxlen=self.frame_stack)
        self._step   = 0
        self._episode = 0

        # Đếm số bước liên tiếp reward âm (dừng sớm nếu xe bị lạc)
        self._neg_reward_cnt = 0

    # ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None):
        obs, _ = self._env.reset(seed=seed)
        self._step = 0
        self._episode += 1
        self._neg_reward_cnt = 0

        frame = self._preprocess(obs)
        for _ in range(self.frame_stack):
            self._frames.append(frame)

        return self._get_state()

    # ──────────────────────────────────────────────────────────────────
    def step(self, action_idx):
        self._step += 1
        total_reward = 0.0
        done = False

        # Action repeat: lặp action skip_frames lần
        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self._env.step(action_idx)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        frame = self._preprocess(obs)
        self._frames.append(frame)
        state = self._get_state()

        # Dừng sớm nếu reward âm liên tục (xe bị lạc khỏi đường)
        if total_reward < 0:
            self._neg_reward_cnt += 1
        else:
            self._neg_reward_cnt = 0

        if self._neg_reward_cnt > 25:
            done = True
            total_reward -= 50  # penalty thêm

        return state, total_reward, done, info

    # ──────────────────────────────────────────────────────────────────
    def _preprocess(self, obs):
        """RGB (96,96,3) → Grayscale → resize → normalize"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _get_state(self):
        return np.array(self._frames, dtype=np.float32)  # (4, 84, 84)

    # ──────────────────────────────────────────────────────────────────
    def close(self):
        self._env.close()
        print("[CarRacingEnv] Closed.")
