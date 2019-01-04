from unityagents import UnityEnvironment
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("unityagents")
#logger.setLevel(logging.NOTSET)
logger.propagate = False
env = UnityEnvironment(file_name="../VisualBanana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env.close()

action_size = 4

def dqn(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes+1):
        #state = env.reset()
        if i_episode % 100 == 1:
            env = UnityEnvironment(file_name="../VisualBanana_Linux/Banana.x86_64")
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.visual_observations[0]
        score = 0
        while True:
            action = random.choice(np.arange(action_size))
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.visual_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]

            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=20.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
        if i_episode % 100 == 0:
            env.close()
    return scores

dqn_scores = dqn()

