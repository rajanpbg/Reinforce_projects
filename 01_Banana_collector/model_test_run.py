#!/usr/bin/python
# Author: Govinda
# testing the trained model Weghts
# This  script used the trained neural network weights and  runs the agent

import sys
sys.path.append('./python/')

from unityagents import UnityEnvironment
import numpy as np

## loading the ENviroment and Brain
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
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
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

### start the agent
from DDQN_PE import Agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent.qnetwork_local.load_state_dict(torch.load('./DDQN_ER_checkpoint.pth'))
#agent.qnetwork_local.load_state_dict(torch.load('./baisc_dqn_checkpoint.pth'),strict='False')
env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
def lets_make_test():
   env_info = env.reset(train_mode=True)[brain_name] # reset the environment
   state = env_info.vector_observations[0]            # get the current state
   score = 0                                          # initialize the score
   while True:
     action = agent.act(state, 0.)       # select an action
     env_info = env.step(action)[brain_name]# send the action to the environment
     #env.render()
     next_state = env_info.vector_observations[0]   # get the next state
     reward = env_info.rewards[0]                   # get the reward
     done = env_info.local_done[0]                  # see if episode has finished
     score += reward                                # update the score
     state = next_state                             # roll over the state to next time step
     if done:                                       # exit loop if episode finished
         break

   return score

## Run 5 episodes
myresults = []
for i in range(5):
    myresults.append(lets_make_test())
print(myresults)

