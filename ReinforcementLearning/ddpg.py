import collections

import torch
import gym
import torch.nn.functional as F
from torch.distributions.normal import Normal
import random

'''
Current implementation will not work with CNN's as Qnetwork or Pnetwork
'''


def forward(network, input_):
    for i in network:
        if torch.is_tensor(i):
            input_ = input_.matmul(i)
        else:
            input_ = i(input_)
    return input_


def create_network(dims):
    network = []
    for i, j in enumerate(dims):
        if i != len(dims) - 1:
            network.append(torch.randn([dims[i], dims[i + 1]]))
            if i < len(dims) - 2:
                network.append(F.relu_)

    return network


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __call__(self, s, a, r, s_, d):
        # add (s,a,r,s',d) to the buffer
        self.buffer.append((s, a, r, s_, d))

    def sample(self, size):
        # return a sample of size B
        batch = random.sample(self.buffer, size)
        return map(torch.FloatTensor, zip(*batch))


class QNetwork:
    def __init__(self, qnetwork_mid_dims, action_space, observation_space):
        qnetwork_mid_dims.append(1)
        qnetwork_mid_dims.insert(0, action_space + observation_space)
        self.target_network = self.current_network = create_network(qnetwork_mid_dims)

    def __call__(self, network, input_):
        if network == "target":
            return forward(self.target_network, input_)
        if network == "current":
            return forward(self.current_network, input_)


class PNetwork:
    def __init__(self, pnetwork_mid_dims, action_space, action_space_high, action_space_low, observation_space,
                 add_noise_till):
        self.action_space = action_space
        self.action_high = action_space_high
        self.action_low = action_space_low
        self.observation_space = observation_space
        pnetwork_mid_dims.append(action_space)
        pnetwork_mid_dims.insert(0, observation_space)
        self.PNetwork_target = create_network(pnetwork_mid_dims)
        self.PNetwork_current = create_network(pnetwork_mid_dims)
        self.noise = Normal(0, 1)
        self.add_noise_till = add_noise_till

    def take_action(self, observation, total_steps):
        action = forward(self.PNetwork_target, observation)
        if total_steps <= self.add_noise_till:
            noise = self.noise.sample()
            action += noise

        return self.clip_action(action)

    def clip_action(self, action):
        if action < self.action_low: action = self.action_low
        if action > self.action_high: action = self.action_high

        return action

    def __call__(self, input_, network="current"):
        if network == "current":
            return forward(self.PNetwork_current, input_)
        if network == "target":
            return forward(self.PNetwork_target, input_)


class DDPG:
    def __init__(self, pnetwork_mid_dims, qnetwork_mid_dims, action_space, action_space_high, action_space_low,
                 observation_space, buffer_size, batch_size, polyak, add_noise_till, discount_factor, lr):
        self.PNetwork_ = PNetwork(pnetwork_mid_dims, action_space, action_space_high, action_space_low,
                                  observation_space, add_noise_till)
        self.QNetwork = QNetwork(qnetwork_mid_dims, action_space, observation_space)
        self.ReplayBuffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.polyak = polyak
        self.discount_factor = discount_factor
        self.lr = lr

    def UpdateQ(self, batch):
        s, a, r, s_, d = batch
        # Compute targets
        a_targets = self.PNetwork_(s_, "target")
        q_targets = self.QNetwork("target", torch.cat((s_, a_targets), -1))
        targets = r + self.discount_factor(1 - d) * q_targets
        logits = self.QNetwork("current", torch.cat((s, a), -1))
        loss_ = F.mse_loss(logits, targets)
        loss_.backward()
        self.optim_step(self.QNetwork.current_network)
        self.zero_grad()

    def optim_step(self, network, max=False):
        params_ = []
        for i in network:
            if torch.is_tensor(i):
                params_.append(i)
        with torch.no_grad():
            for i in params_:
                if max:
                    i += self.lr * i.grad
                else:
                    i -= self.lr * i.grad

    def zero_grad(self):
        # Ideally I shud have model.get_trainable_params here
        networks = [self.PNetwork_target, self.PNetwork_current, self.current_network, self.target_network]
        for j in networks:
            for i in j:
                if torch.is_tensor(i) and i.requires_good == True:
                    i.grad.data.zero_()

    def UpdateP(self, batch):
        s, _, _, _, _ = batch
        a = self.PNetwork_(s)
        cost_func_for_policy = torch.sum(self.QNetwork("current", torch.cat((s, a), -1))) / len(s)
        cost_func_for_policy.backward()
        self.optim_step(self.PNetwork, max=True)

    def UpdateNetworks(self):
        for i, j in zip(self.PNetwork_target, self.PNetwork_current):
            if torch.is_tensor(i) and torch.is_tensor(j) and i.requires_grad == True and j.requires_grad == True:
                i.data = self.polyak * i.data + (1 - self.polyak) * j.data

        for i, j in zip(self.current_network, self.target_network):
            if torch.is_tensor(i) and torch.is_tensor(j) and i.requires_grad == True and j.requires_grad == True:
                i.data = self.polyak * i.data + (1 - self.polyak) * j.data

    def __getattr__(self, item):
        if hasattr(self,item):
            return getattr(self,item)
        elif hasattr(self.PNetwork_, item):
            return getattr(self.PNetwork, item)
        elif hasattr(self.QNetwork, item):
            return getattr(self.QNetwork, item)
        else:
            raise AttributeError


def main():
    # arguments
    epochs = 100
    max_steps_per_episode = 1500
    update_every = 50
    update_after = 1000
    batch_size = 100
    buffer_size = 10000
    polyak = 0.995
    pnetwork_mid_dims = [10, 10]
    qnetwork_mid_dims = [15, 10]
    add_noise_till = 2000
    discount_factor = 0.9
    lr = 0.001
    no_of_updates = 5

    # Environment
    env = gym.make('MountainCarContinuous-v0')
    action_space = env.action_space.shape[0]
    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]
    observation_space = env.observation_space.shape[0]
    observation = env.reset()

    # Agent
    agent = DDPG(pnetwork_mid_dims, qnetwork_mid_dims, action_space, action_space_high, action_space_low,
                 observation_space, buffer_size, batch_size, polyak, add_noise_till, discount_factor, lr)

    total_steps = 0
    for i in range(epochs):
        done = False
        j = 0
        while done and j < max_steps_per_episode:
            action = agent.take_action(observation, total_steps)
            next_observation, reward, done, _ = env.step(action)
            agent.ReplayBuffer(observation, action, reward, next_observation, done)
            observation = next_observation

            if j % update_every == 0 and total_steps > update_after:
                for k in range(no_of_updates):
                    batch = agent.ReplayBuffer.sample(batch_size)
                    agent.UpdateQ(batch)
                    agent.UpdateP(batch)
                    agent.UpdateNetworks()

            j += 1
            total_steps += 1


if __name__ == "__main__":
    main()
