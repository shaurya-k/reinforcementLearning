from collections import deque
import gym
import random
import numpy as np
import time
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
        current = observation

        if random.uniform(0, 1) > EPSILON:
            prediction = np.array([Q_table[(current, i)] for i in range(env.action_space.n)])
            bestAction = np.argmax(prediction)
        else:
            bestAction = env.action_space.sample()
        done = False
        for t in range(100):
            observation, reward, done, _ = env.step(bestAction)
            iterate = observation

            if random.uniform(0, 1) > EPSILON:
                prediction = np.array([Q_table[(iterate, i)] for i in range(env.action_space.n)])
                bestReward = np.argmax(prediction)
            else:
                bestReward = env.action_space.sample()
            rhs = 0
            if done:
                rhs = LEARNING_RATE * (reward - Q_table[(current, bestAction)])
            else:
                rhs = LEARNING_RATE * (reward + (DISCOUNT_FACTOR * Q_table[(iterate, bestReward)]) - Q_table[
                    (current, bestAction)])
            Q_table[(current, bestAction)] = Q_table[(current, bestAction)] + rhs

            # update variables before next iteration
            current = iterate
            bestAction = bestReward
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
    model_file = open('SARSA_Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    #######################
