import gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict

EPISODES = 20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        observation = env.reset()
        episode_reward = 0
        done = False

        for t in range(100):
            if random.uniform(0, 1) > EPSILON:
                prediction = np.array([Q_table[(observation, i)] for i in range(env.action_space.n)])
                action = np.argmax(prediction)  # picking action w/ highest reward
            else:
                action = env.action_space.sample()

            newObservation, reward, done, _ = env.step(action)  # don't care about info returned

            prediction = np.array([Q_table[(newObservation, i)] for i in range(env.action_space.n)])
            bestAction = np.argmax(prediction)  # picking action w/ highest reward

            curr = Q_table[(observation, action)]
            rhs = 0
            if not done:
                rhs = LEARNING_RATE * (reward + (DISCOUNT_FACTOR * np.max(Q_table[(newObservation, bestAction)])) - curr)
            else:
                rhs = LEARNING_RATE * (reward - curr)
            Q_table[(observation, action)] = curr + rhs

            observation = newObservation
            episode_reward += reward

            if done:
                break

        episode_reward_record.append(episode_reward)
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
            print("EPSILON: " + str(EPSILON))

        # decay epsilon
        EPSILON = EPSILON * EPSILON_DECAY

    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    #######################
