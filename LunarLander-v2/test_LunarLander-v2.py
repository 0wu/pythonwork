"""
Policy Gradient, Reinforcement Learning.
The LunarLander-v2 example

By Kevin Kuei

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import math
import numpy as np
import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 100  # renders environment if long-term episode reward is greater then this
RENDER = True  # rendering wastes time
current_max = 100

env = gym.make('LunarLander-v2')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.005,
    reward_decay=0.99
    # output_graph=True,
)

RL.restore_model()

for i_episode in range(5000):

    observation = env.reset()

    t = 0
    episode_reward = 0

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward)
        observation = observation_

        if (t%1000) == 0:
            print("t=", t)

        if (t>=10000):
            done = True

        t += 1

        episode_reward += reward

        if done:
            #print("RL.ep_obs: "); print(RL.ep_obs.shape)
            #print("np.vstack(RL.ep_obs).shape = "); print(np.vstack(RL.ep_obs).shape)
            #print("np.array(RL.ep_as).shape = "); print(np.array(RL.ep_as).shape)
            ep_rs_sum = sum(RL.ep_rs)
            print("episode:", i_episode, " episode_reward:", episode_reward, " t:", t)

            #vt = RL.learn()
            break


