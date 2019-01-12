from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
import time
import gc
import cv2
import logging
import sys

logger = logging.getLogger("unityagents")
logger.propagate = False

env = UnityEnvironment(file_name="../VisualBanana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.visual_observations[0]
print('States look like:', state.shape)
#state = np.squeeze(state)
#state *= 256
#state = state.astype(np.uint8)
#state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
#cv2.imshow('display', state)
#cv2.imshow('display', cv2.resize(state, (256, 256), interpolation=cv2.INTER_NEAREST))
#cv2.imshow('display', cv2.resize(state[...,0], (512, 512)))
#cv2.waitKey(0)
#plt.imshow(np.squeeze(state))
#plt.show()
#state = np.mean(state, axis=3)
#state = np.squeeze(state)
state = np.dot(state, np.array([0.299, 0.587, 0.114]))
#state = state.reshape(state.shape[1], state.shape[2], 1)
#print(state)
print('States look like:', state.shape) # -> (1, 84, 84)
#plt.imshow(np.squeeze(state), cmap='gray')
#plt.show()

env.close()

def preprocess(state):
    state = np.dot(state, np.array([0.299, 0.587, 0.114]))
    state *= 256
    return state.astype(np.uint8)
    #return np.transpose(state, (2, 0, 1))

FRAME_SIZE = 4
print("FRAME_SIZE:{}".format(FRAME_SIZE), file=sys.stderr)
SIM_RESET_INTERVAL = 500

from dqn_agent import Agent
from double_dqn_agent import DDQNAgent
from duel_double_dqn_agent import DuelDDQNAgent

#agent = Agent(state_size=(FRAME_SIZE, state.shape[1], state.shape[2]), action_size=action_size, seed=0)
#agent = DDQNAgent(state_size=(FRAME_SIZE, state.shape[1], state.shape[2]), action_size=action_size, seed=0)
agent = DuelDDQNAgent(state_size=(FRAME_SIZE, state.shape[1], state.shape[2]), action_size=action_size, seed=0)
print("Agent's Name:{}".format(agent.__class__.__name__), file=sys.stderr)

basename = ''

def dqn(n_episodes=int(5e4), max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
  
    obs_queue = deque(maxlen=FRAME_SIZE)
    start = time.time()
    
    for i_episode in range(1, n_episodes+1):
        if i_episode % SIM_RESET_INTERVAL == 1:
            env = UnityEnvironment(file_name="../VisualBanana_Linux/Banana.x86_64")

        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.visual_observations[0]
        obs = np.squeeze(obs)
        obs = preprocess(obs)
        obs_queue.append(obs)
        state = None
        score = 0
        while True:
            if state is not None:
                action = agent.act(state, eps)
            else:
                action = random.choice(np.arange(action_size)) 
            env_info = env.step(action)[brain_name]        # send the action to the environment
            obs = env_info.visual_observations[0]   # get the next state
            obs = np.squeeze(obs)
            obs = preprocess(obs)
            #cv2.imshow('display', cv2.resize(np.flip(obs, 2), (528, 528), interpolation=cv2.INTER_NEAREST))
#            cv2.imshow('display', cv2.resize(obs, (256, 256), interpolation=cv2.INTER_NEAREST))
#            ch = cv2.waitKey(300)
#            if ch == 27:
#                import sys
#                sys.exit()
            obs_queue.append(obs)
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0] 

            if len(obs_queue) == FRAME_SIZE:
                next_state = np.array(obs_queue)
                #print('next_state.shape', next_state.shape)
                if state is not None:
                    agent.step(state, action, reward, next_state, done)
                state = next_state

            score += reward
            #time.sleep(0.2)
            if done:
                obs_queue.clear()
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if agent.l_step > 0:
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        current = time.time()
        elapsed = current - start
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print('\rEpisode {}\tAverage Score: {:.2f}\t{}\t{}\t{}\t{}\t{}'.format(i_episode, np.mean(scores_window), elapsed_str, action, agent.l_step, len(agent.memory), eps), end="", file=sys.stderr)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\t{}\t{}\t{}\t{}\t{}'.format(i_episode, np.mean(scores_window), elapsed_str, action, agent.l_step, len(agent.memory), eps), file=sys.stderr)
        if np.mean(scores_window)>=14.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break

        if i_episode % SIM_RESET_INTERVAL == 0:
            env.close()
    time_str = time.strftime("%Y-%m-%d_%I-%M-%S%p")
    basename = agent.__class__.__name__ + time_str
    torch.save(agent.qnetwork_local.state_dict(), './checkpoint_' + basename + '.pth')
    return scores

dqn_scores = dqn()

## plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(dqn_scores)), dqn_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
print('\nsave learning graph on {}.'.format(basename), file=sys.stderr)
plt.savefig('learning_graph_' + basename + '.png')
np.save('learning_graph_' + basename + '.npy', dqn_scores)
#plt.show()
#
