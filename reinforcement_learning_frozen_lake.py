import numpy as np
import matplotlib.pyplot as plt
import gym
import random

EPISODES = 30000
LEARNING_RATE = 0.05
DISCOUNT = 0.95
EPSILON_MAX = 0.9
DECAY = 0.01

env = gym.make("FrozenLake-v0")
size_actions = env.action_space.n
size_os = env.observation_space.n

q_matrix = np.zeros((size_os, size_actions))

rewards = []

increments = 1000

for e in range(EPISODES):
    
  state = env.reset()
  total = 0
  done = False

  while not done:
    if random.uniform(0, 1) > EPSILON_MAX:
      action = np.argmax(q_matrix[state,:])
    else:
      action = env.action_space.sample()

    new_state, reward, done, _ = env.step(action)
    total += reward

    q_matrix[state, action] = q_matrix[state, action] + LEARNING_RATE * (reward + DISCOUNT * np.max(q_matrix[new_state, :]) - q_matrix[state, action])

    state = new_state
    
  if e%increments == 0:
    print(f'Total reward: {total}')
    env.render()
    
  EPSILON_MAX = 0.01 + np.exp(-DECAY*e)
  rewards.append(total)

sums = 0
avg_rewards = []
for i in range(len(rewards)):
  if rewards[i] == 1:
    sums+=1;
    avg_rewards.append(sums / (i+1));

plt.plot(avg_rewards)
plt.show()