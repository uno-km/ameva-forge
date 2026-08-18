"""
==============================================================================
CartPole RL Environment & WebGPU Neural Agent (forge.rl.cartpole)
==============================================================================

WHAT:
  Physics-based CartPole environment with Deep Q-Network (DQN) and Policy Gradient
  neural agents running in-place WebGPU SGD optimization.
"""

import math
import random
from typing import Tuple, List, Optional
import numpy as np

import forge as torch
import forge.nn as nn
import forge.optim as optim
import forge.functional as F
from forge.ops import tensor


class CartPoleEnv:
    """
    Classic CartPole-v1 discrete dynamics environment.
    """
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.state = None
        self.steps_beyond_done = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps_beyond_done = None
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not done else 0.0
        return self.state, reward, done, {}


class DQNAgent(nn.Module):
    """
    Deep Q-Network Agent on WebGPU.
    """
    def __init__(self, state_dim: int = 4, hidden_dim: int = 32, action_dim: int = 2, lr: float = 0.02, device: str = "gpu"):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(state_tensor)

    async def act(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randint(0, 1)
        st = tensor([state.tolist()], dtype="float32", device=self.device)
        q_vals = self.forward(st)
        q_np = await q_vals.numpy_async()
        return int(np.argmax(q_np))

    async def train_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, gamma: float = 0.99) -> float:
        st = tensor([state.tolist()], dtype="float32", device=self.device)
        nst = tensor([next_state.tolist()], dtype="float32", device=self.device)

        # Q(s, a)
        q_values = self.forward(st)
        q_target_values = self.forward(nst)

        q_np = await q_values.numpy_async()
        q_next_np = await q_target_values.numpy_async()

        target = q_np.copy()
        if done:
            target[0, action] = reward
        else:
            target[0, action] = reward + gamma * float(np.max(q_next_np))

        target_tensor = tensor(target.tolist(), dtype="float32", device=self.device)
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, target_tensor)
        loss_val = float((await loss.numpy_async()).item())

        loss.backward()
        await self.optimizer.step_async()
        return loss_val


class PolicyGradientAgent(nn.Module):
    """
    REINFORCE Policy Gradient Agent on WebGPU.
    """
    def __init__(self, state_dim: int = 4, hidden_dim: int = 32, action_dim: int = 2, lr: float = 0.01, device: str = "gpu"):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.net(state_tensor)
        return F.softmax(logits, dim=-1)
