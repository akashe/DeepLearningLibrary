import collections

import torch
from CostFunctions.MeanSquarredError import MeanSquarredError
from Optimizers.SGD import SGD
from TorchFunctions.dataModifications import appendOnes
import gym
import random
import time

'''
The idea is to build a simple dqn algo with only linear transformation and updating the parameters with SGD
I will use simple LunarLander gym.
The idea is to get a feel of the algo not to extend functionality. So the implementation will be bare minimum.
'''


class DQNAgent:

    def __init__(self, discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions,
                 batch_size, learning_rate, max_memory_size):
        '''

        :param discounted_factor: discount factor for future rewards
        :param epsilon_start: starting value of epsilon in e-greedy exploration
        :param epsilon_decay: rate of epsilon decay
        :param epsilon_end: ending value of epsilon
        :param observation_size: state size or observation size or input size
        :param no_of_actions: total number of discrete actions
        :param batch_size: batch size for experience replay
        :param max_memory_size: total size of memory buffer
        '''
        self.d = discounted_factor
        self.e = epsilon_start
        self.e_decay = epsilon_decay
        self.e_end = epsilon_end
        self.o_size = observation_size
        self.n_actions = no_of_actions
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=max_memory_size)
        self.lr = learning_rate
        # Initializing two networks with simple (Wx + b)..no non-linearity
        self.target_network = torch.randn(self.o_size + 1, self.n_actions, dtype=torch.float)
        self.current_network = torch.randn(self.o_size + 1, self.n_actions, dtype=torch.float)
        self.loss = []
        self.step_counter = 0
        self.learn_counter = 0

    def store_experience(self, experience):
        # observation, action, reward, next_observation, done = experience
        self.memory.appendleft(experience)

    def sample_experience(self):
        # don't forget to put check for len(deque)< batch_size
        return random.sample(self.memory, self.batch_size)

    def switch_networks(self):
        # Switch target and current network after every 10 games
        self.target_network.data = self.current_network.clone()

    def reduce_epsilon(self):
        # Putting this in take_action rather than learn
        self.e = self.e - self.e_decay if self.e > self.e_end else self.e_end

    def take_action(self, observation):
        # don't forget to decay epsilon
        c = random.random()
        self.reduce_epsilon()
        if c > self.e:
            observation = appendOnes(torch.FloatTensor(observation))[None,]
            actions = observation.mm(self.current_network)
            action = torch.argmax(actions).item()
            return action
        else:
            return random.randrange(0, self.n_actions)

    def learn(self):
        replay_experience = self.sample_experience()
        # I dont have to specifically train only current network in the beginning it wud be just
        # as random as the target network
        replay_experience = list(zip(*replay_experience))
        observations = torch.FloatTensor(replay_experience[0])
        actions = torch.FloatTensor(replay_experience[1])
        rewards = torch.FloatTensor(replay_experience[2])
        next_observations = torch.FloatTensor(replay_experience[3])
        non_terminal_state = 1 - (torch.BoolTensor(replay_experience[4])).type(torch.float)

        # Appending observations for bias
        observations = appendOnes(observations)
        next_observations = appendOnes(next_observations)

        # Loss = { target_value - actual_value}^2
        # Calculating target value using target_network

        future_rewards = self.d * torch.max(next_observations.mm(self.target_network), dim=1)[0] * non_terminal_state
        target_value = rewards + future_rewards

        # Calculating actual values
        # Question: are we broadcasting target value to full action space/Q values and subtracting them
        # error??
        actual_value = observations.mm(self.current_network)

        loss = MeanSquarredError(labels=target_value.unsqueeze(-1), targets=actual_value, batch_size=self.batch_size)
        print(" Loss for step " + str(self.step_counter) + " = " + str(loss.item()))
        self.loss.append(loss)

        # gradients transpose of observation(batch_size* input dim+1)* (target- actual)(batch*size* output dim)
        gradients = -(2 / self.batch_size) * observations.t().mm(
            (target_value.unsqueeze(-1) - actual_value))

        # apply gradients
        self.current_network = SGD(parameters=self.current_network, gradients=gradients, learning_rate=self.lr)


if __name__ == "__main__":
    # define system parameters
    (discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions, batch_size,
     learning_rate,
     max_memory_size) = \
        (0.9, 1.0, 0.0001, 0.01, 8, 4, 64, 0.003, 100000)
    # define env
    env = gym.make('LunarLander-v2')
    # define agent
    agent = DQNAgent(discounted_factor, epsilon_start, epsilon_decay, epsilon_end, observation_size, no_of_actions,
                     batch_size, learning_rate,
                     max_memory_size)
    # train a policy
    total_games = 500
    scores = []
    # train loop
    for i in range(total_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.take_action(observation)
            next_observation, reward, done, _ = env.step(action)
            agent.store_experience((observation, action, reward, next_observation, done))
            score += reward
            agent.step_counter += 1
            observation = next_observation
            if agent.step_counter < batch_size:
                continue
            else:
                agent.learn()
                agent.learn_counter += 1
                # Switch networks after every 1000 learn steps
                if agent.learn_counter == 3000:
                    print("Switching networks")
                    agent.switch_networks()
                    agent.learn_counter = 0

        scores.append(score)
        print(" Game ", i, 'score %.2f' % score, )

    # test policy
    observation = env.reset()
    test_episodes = 20
    for i in range(test_episodes):
        done = False
        observation = env.reset()
        while not done:
            env.render()
            time.sleep(1e-3)
            action = agent.take_action(observation)
            observation, r, done, _ = env.step(action)

'''
Result: non converging
1) maybe I implemented the learn equation wrong.
2) maybe I need non linearity. But I thought tha same for linear models..maybe something wrong
with my gradients or maybe I need to clip them.
'''