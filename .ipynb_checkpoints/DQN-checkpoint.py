import numpy as np
import torch
torch.backends.cudnn.benchmark = True

from math import log
import torch.nn as nn
import torch.nn.functional as F
from src.CNN import CNNActionValue

class DQN:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=0.00025,
            epsilon=1.0,
            epsilon_min=0.1,
            gamma=0.99,
            batch_size=32,
            warmup_steps=5000,
            buffer_size=int(1e5),
            target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)
        # self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('dqn :', torch.cuda.is_available())
        print('dqn :', self.device)
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a

    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        next_q = self.target_network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result

    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon -= self.epsilon_decay
        return result


class ReplayBuffer:
    def __init__(self, alpha, state_dim, action_dim, max_size=int(1e5), device='cuda'):
        self.device = device
        self.s = torch.zeros((max_size, *state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((max_size, *action_dim), dtype=torch.long, device=device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.s_prime = torch.zeros((max_size, *state_dim), dtype=torch.float32, device=device)
        self.terminated = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.priority = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

        self.alpha = alpha
        self.seg_tree_size = 1 << (1 + round(math.log(n) / math.log(2)))
        self.seg_tree_leaf_idx = self.seg_tree_size >> 1
        self.seg_tree = [0] * self.seg_tree_size
        
        self.ptr = 0

    def update(self, s, a, r, s_prime, terminated, priority):
        self.s[self.ptr] = torch.tensor(s, device=self.device)
        self.a[self.ptr] = torch.tensor(a, device=self.device)
        self.r[self.ptr] = torch.tensor(r, device=self.device)
        self.s_prime[self.ptr] = torch.tensor(s_prime, device=self.device)
        self.terminated[self.ptr] = torch.tensor(terminated, device=self.device)
        self.priority[self.ptr] = torch.tensor(priority, device=self.device)
        
        idx += self.seg_tree_size
        diff = priority - self.seg_tree[idx]
        while idx:
            self.seg_tree[idx] += diff
            idx //= 2
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def search(self, priority):
        idx = 1

        return idx - self.seg_tree_size

    def sample(self, batch_size):
        priorities = torch.rand(0, self.seg_tree[1], (batch_size,), device=self.device)

        ind = np.zeros(batch_size)
        for idx, priority in enumerate(priorites):
            ind[idx] = search(self.search(priority)
        
        return self.s[ind], self.a[ind], self.r[ind], self.s_prime[ind], self.terminated[ind]
