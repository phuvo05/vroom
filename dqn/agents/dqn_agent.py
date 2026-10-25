
import os
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from models.q_network import QNetwork


class DQNAgent:
    def __init__(self, config):
        self.state_dim     = config["state_dim"]
        self.action_size   = config["action_size"]
        self.gamma         = config["gamma"]
        self.lr            = config["lr"]
        self.batch_size    = config["batch_size"]
        self.memory_size   = config["memory_size"]
        self.target_update = config["target_update"]

        self.epsilon       = config["epsilon_start"]
        self.epsilon_end   = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Device: {self.device}")

        self.policy_net = QNetwork(self.state_dim, self.action_size).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn   = nn.SmoothL1Loss()
        self.memory    = deque(maxlen=self.memory_size)
        self.step_cnt  = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        # Frames are k/255 grayscale, so uint8 storage is lossless and 4x smaller.
        self.memory.append((np.rint(state * 255).astype(np.uint8), action, reward,
                            np.rint(next_state * 255).astype(np.uint8), done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch  = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.from_numpy(np.array(states)).to(self.device).float() / 255.
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).to(self.device).float() / 255.
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions)
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.step_cnt += 1
        if self.step_cnt % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "step_cnt":   self.step_cnt,
        }, path)
        print(f"[DQNAgent] Saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon  = ckpt["epsilon"]
        self.step_cnt = ckpt["step_cnt"]
        print(f"[DQNAgent] Loaded ← {path}")

    def load_weights(self, path):
        """Transfer visual driving weights, keeping a fresh optimizer/exploration."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["policy_net"])
        print(f"[DQNAgent] Warm start from {path}")
