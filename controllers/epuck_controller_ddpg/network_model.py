# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 Mar 2023
#
# Purpose:  Network model in pytorch for DDPG within webots
#
# References
# ----------
# DDPG paper:
#       Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez,
#       Yuval Tassa, David Silver, and Daan Wierstra. ‘Continuous Control with Deep Reinforcement Learning’.
#       CoRR abs/1509.02971 (2016).
#
# This implementation based on:
#       Sanghi, Nimish. Deep Reinforcement Learning with Python: With PyTorch, TensorFlow and OpenAI Gym.
#       New York: Apress, 2021. https://doi.org/10.1007/978-1-4842-6809-4.
#       This code is licenced under the Freeware Licence as described here:
#       https://github.com/Apress/deep-reinforcement-learning-python/blob/main/LICENSE.txt
#
# # Freeware License, some rights reserved
#
# Copyright (c) 2021 Nimish Sanghi
#
# Permission is hereby granted, free of charge, to anyone obtaining a copy 
# of this software and associated documentation files (the "Software"), 
# to work with the Software within the limits of freeware distribution and fair use. 
# This includes the rights to use, copy, and modify the Software for personal use. 
# Users are also allowed and encouraged to submit corrections and modifications 
# to the Software for the benefit of other users.
#
# It is not allowed to reuse,  modify, or redistribute the Software for 
# commercial use in any way, or for a user’s educational materials such as books 
# or blog articles without prior permission from the copyright holder. 
#
# The above copyright notice and this permission notice need to be included 
# in all copies or substantial portions of the software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS OR APRESS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# Include parameterised Tanh activation function, implemented from scratch at described:
# https://www.mdpi.com/2673-2688/2/4/29
#
# Changes from reference implementation
# -------------------------------------
# Resolved some swapping between GPU and CPU.
# ==========

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from ou_noise import OrnsteinUhlenbeckActionNoise
import numpy as np


class MLPActorCritic(nn.Module):
    """Combines the actor and critic into a single nn.Module."""
    def __init__(self, observation_space: int, action_space: int, action_limit: float = 1.0):
        super().__init__()
        self.state_dim = observation_space
        self.act_dim = action_space
        self.act_limit = action_limit

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_dim), sigma=np.ones(self.act_dim) * 0.2)

        # build Q and policy functions
        self.q = MLPQFunction(self.state_dim, self.act_dim).cuda()
        self.policy = MLPActor(self.state_dim, self.act_dim, self.act_limit).cuda()
        # How can the above line have self.state_dim be value 8 when it is actually 13?

    def act(self, state):
        with torch.no_grad():
            # return self.policy(state).cpu().numpy()
            return self.policy(torch.as_tensor(state, dtype=torch.float32).cuda()).cuda()

    def get_action(self, s, noise_scale: float = 0.1):
        a = self.act(torch.as_tensor(s, dtype=torch.float32).cuda())
        # a = a + torch.tensor(self.noise()).cuda()        # Use OU noise for exploration
        a = a + torch.FloatTensor(a.shape).normal_(mean=0.15, std=noise_scale).to("cuda")
        # return a.clamp(-self.act_limit, self.act_limit)
        return a.clamp(0., self.act_limit)

    def reset(self):
        self.noise.reset()


class MLPQFunction(nn.Module):
    """Q-Function model for value estimation."""
    def __init__(self, state_dim, act_dim, layer: int = 400):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, layer)
        self.fc2 = nn.Linear(layer + act_dim, layer)
        self.fc3 = nn.Linear(layer, layer)
        self.Q = nn.Linear(layer, 1)

    def forward(self, s, a):
        # x = torch.cat([s, a], dim=-1)
        x = self.fc1(s)
        x = F.relu(x)
        x = torch.cat([x, a], dim=-1)
        x = self.fc2(x)
        x = F.selu(x)
        x = self.fc3(x)
        x = F.relu(x)
        q = self.Q(x)
        return torch.squeeze(q, -1)


class MLPActor(nn.Module):
    """Policy network for the actor critic."""
    def __init__(self, state_dim, act_dim, act_limit, layer: int = 400):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = nn.Linear(state_dim, layer)
        self.fc2 = nn.Linear(layer, layer)
        self.fc3 = nn.Linear(layer, layer)
        self.actor = nn.Linear(layer, act_dim)
        self.actor_k = nn.Linear(layer, act_dim)             # Parameterised tanh
        self.actor_x_0 = nn.Linear(layer, act_dim)           # Parameterised tanh

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.selu(x)
        x = self.fc3(x)
        x = F.relu(x)
        # action = torch.sigmoid(self.actor(x))
        x_a = self.actor(x)
        x_0 = self.actor_x_0(x)
        # k = torch.clip(self.actor_k(x), 0.1, 25)
        k = torch.clip(torch.sigmoid(self.actor_k(x)), 0.1, 1)
        action = torch.sigmoid((k * x_a) - (k * x_0))        # Parameterised sigmoid, to output in range(0,1)
        return action




