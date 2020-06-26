#!/usr/bn/python
# Author: Govinda
# Details: This is the main module used to intlize /train / run model
# this uses DDPG technique
# (Deep Determenstic Policy Gradient) Method for traing a doulbe-joint ARM
# DDPG uses 4  networks
# 2 - for Actor (local,target), 2- critic (local,target)
# This module will provide functionaties for Agent
# Taking step, Training the model, soft updating the target network
# Adding Noise for exploration
######################################################################
import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import argparse
from unityagents import UnityEnvironment
import numpy as np
from Agent import Agent
import torch
from collections import deque


def lets_add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-example-run", help="Example Run", action="store_true")
    parser.add_argument("--get-size-of-space", help="Get Size of environment", action="store_true")
    parser.add_argument("--get-sample-space-actions", help="Get sample space", action="store_true")
    parser.add_argument("--Qmodel-run", help="Run an exampke from saved model", action="store_true")
    parser.add_argument("--train", help="Run training", action="store_true")
    return parser

def lets_get_arguments(parser):
    env = UnityEnvironment(file_name='./Reacher_Linux_single/Reacher.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    args = parser.parse_args()
    if args.display_example_run:
        display_example_run(env,num_agents)

    if args.get_size_of_space:
        get_size_of_space(env,num_agents)
    if args.get_sample_space_actions:
        get_sample_space_actions(env,num_agents)
    if args.train:
        train_the_agent(env)



def display_example_run(env,num_agents):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    #env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    # initialize the score (for each agent)
    count = 0
    while True:
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
        count += 1
        if count > 100:
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    env.close()


def get_size_of_space(env,num_agents):
    # examine the state space
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    env.close()

def get_sample_space_actions(env,num_agents):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    print('Sample action:', np.random.randn(1, action_size))
    print("Example state shape {}  and example {} ".format(states.shape[0], state_size))
    env.close()

def train_the_agent(env,n_episodes = 100,max_t = 700):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    print(state_size,action_size)
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score),
              end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    print(scores)
    env.close()



if __name__ == "__main__":
    parser = lets_add_arguments()
    parser.parse_args()
    lets_get_arguments(parser)
