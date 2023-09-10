# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 Mar 2023
# Purpose:  Replay buffer for DDPG. This is a basic buffer, quite inefficient and not prioritised.
#
# Includes: ReplayBuffer class
#
# This implementation based on:
#       Sanghi, Nimish. Deep Reinforcement Learning with Python: With PyTorch, TensorFlow and OpenAI Gym.
#       New York: Apress, 2021. https://doi.org/10.1007/978-1-4842-6809-4.
#
#       This code is licenced under the Freeware Licence as given here:
#       https://github.com/Apress/deep-reinforcement-learning-python/blob/main/LICENSE.txt
#
# Freeware License, some rights reserved
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
# ----------


import numpy as np

# import application_log    # Debug logging
# import inspect


class ReplayBuffer:
    def __init__(self, size=1e6):
        self.size = size
        self.buffer = []
        self.next_id = 0

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def load(self, data):
        self.buffer = data
        self.next_id = len(self.buffer) % self.size

    def save(self) -> list:
        return self.buffer

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size=32):
        idxs = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in idxs]


class PrioritizedReplayBuffer:
    def __init__(self, size, n_clusters:int = 3, alpha=0.6, beta=0.4):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to hold buffer
        self.next_id = 0
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.ones(size)
        self.epsilon = 1e-5
        self.window_size = size     # Keep it single-window in this case.
        self.window_step = 000
        self.window_counter = 0

    def __len__(self):
        return len(self.buffer)

    def load(self, data):
        self.buffer, self.priorities = data
        self.next_id = len(self.buffer) % self.size

    def save(self) -> list:
        return [self.buffer, self.priorities]

    def add(self, state, state_cluster, action, reward, next_state, next_state_cluster, done):
        """Add a new item to the circular buffer."""
        item = (state, state_cluster, action, reward, next_state, next_state_cluster, done)
        max_priority = self.priorities.max()
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item

        self.priorities[self.next_id] = max_priority
        if self.window_counter == self.window_size:
            self.window_counter = 0
            self.next_id -= (self.window_size - self.window_step)
        # print(f"window_counter: {self.window_counter}, next_id: {self.next_id}")
        self.window_counter += 1
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        N = len(self.buffer)
        weights = (N * probabilities) ** (-self.beta)
        weights /= weights.max()

        # Multiprocessing means it is not guaranteed for every sample to have a priority assigned yet.
        # Therefore, we need to limit the choice to the number of samples that have a priority.  AE 2023-05-16
        idxs = np.random.choice(len(weights), batch_size, p=probabilities)

        samples = [self.buffer[i] for i in idxs]
        weights = weights[idxs]
        return samples, weights, idxs

    def update_priorities(self, idxs, new_priorities):
        self.priorities[idxs] = new_priorities + self.epsilon



class DWR:
    def __init__(self, size, wsize:int = 30_000, wstep:int = 3_000, n_clusters:int = 3, alpha=0.6, beta=0.4):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to hold buffer
        self.next_id = 0
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.ones(size)
        self.epsilon = 1e-5
        self.window_size = wsize
        self.window_step = wstep
        self.window_counter = 0

    def __len__(self):
        return len(self.buffer)

    def load(self, data):
        self.buffer, self.priorities = data
        self.next_id = len(self.buffer) % self.size

    def save(self) -> list:
        return [self.buffer, self.priorities]

    def add(self, state, state_cluster, action, reward, next_state, next_state_cluster, done):
        """Add a new item to the circular buffer."""
        item = (state, state_cluster, action, reward, next_state, next_state_cluster, done)
        max_priority = self.priorities.max()
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item

        self.priorities[self.next_id] = max_priority
        if self.window_counter == self.window_size:
            self.window_counter = 0
            self.next_id -= (self.window_size - self.window_step)
        # print(f"window_counter: {self.window_counter}, next_id: {self.next_id}")
        self.window_counter += 1
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        N = len(self.buffer)
        weights = (N * probabilities) ** (-self.beta)
        weights /= weights.max()

        # Multiprocessing means it is not guaranteed for every sample to have a priority assigned yet.
        # Therefore, we need to limit the choice to the number of samples that have a priority.  AE 2023-05-16
        idxs = np.random.choice(len(weights), batch_size, p=probabilities)

        samples = [self.buffer[i] for i in idxs]
        weights = weights[idxs]
        return samples, weights, idxs

    def update_priorities(self, idxs, new_priorities):
        self.priorities[idxs] = new_priorities + self.epsilon

