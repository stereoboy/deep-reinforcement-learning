from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name='./Tennis_Linux/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])



import time
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

from ddpg.ddpg_agent import Agent, NoisyAgent

#agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=2)
agent = NoisyAgent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=2)

def ddpg(n_episodes=2000):
    scores_deque = deque(maxlen=100)
    scores = []
    mean_scores = []
    start = time.time()
    for i_episode in range(1, n_episodes+1):
        agent.reset()
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        add_noise=True
        score = np.zeros(num_agents)
        t = 0

        while True:
            #print('\r{}: {}'.format(t, score), end="")
            t += 1
            #action = agent.act(state)
            #next_state, reward, done, _ = env.step(action)
            #env.render()
            #actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            #actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            #actions = agent.act(states, add_noise=add_noise) # select an action (for each agent)
            actions = agent.act(states) # select an action (for each agent)
            #print('\r[{}]{}'.format(t, actions[0]), end="")

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            for i in range(num_agents):
                agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            num_update = 5
            for _ in range(num_update):
                agent.update()
            states = next_states
            score += np.array(rewards)
            #print('\r{}: {} {} {}'.format(t, score, actions[0], actions[1]), end="")
            if np.any(dones):
                break
        max_score = np.max(score)
        scores_deque.append(max_score)
        scores.append(max_score)
        mean_scores.append(np.mean(scores_deque))

        current = time.time()
        elapsed = current - start
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print('\rEpisode {}\tAverage Score: {:.2f}\t{}'.format(i_episode, np.mean(scores_deque), elapsed_str), end="")
        if i_episode%50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\t{}'.format(i_episode, np.mean(scores_deque), elapsed_str))
        if np.mean(scores_deque) > 1.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, color='blue')
plt.plot(np.arange(1, len(max_score)+1), mean_scores, color='orange')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()

