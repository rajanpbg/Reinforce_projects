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
from Agent_multi import Agent_multi
import torch
from collections import deque,defaultdict



def lets_add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-example-run", help="Example Run", action="store_true")
    parser.add_argument("--get-size-of-space", help="Get Size of environment", action="store_true")
    parser.add_argument("--get-sample-space-actions", help="Get sample space", action="store_true")
    parser.add_argument("--trained-model", help="Run an exampke from saved model", action="store_true")
    parser.add_argument("--train", help="Run training", action="store_true")
    return parser

def lets_get_arguments(parser):
    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

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
    if args.trained_model:
        trained_qmodel_run(env)



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
        print(type(actions))
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
        count += 1
        if count > 300:
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
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape, state_size))
    print('The state for the first agent looks like:', states)
    print(type(states))
    env.close()

def get_sample_space_actions(env,num_agents):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    print('Sample action:', np.random.randn(1, action_size))
    print('Sample shape:', action_size )
    print("Example state shape {}  and example {} ".format(states.shape[0], state_size))
    print(type(states))
    env.close()

def train_the_agent(env,n_episodes = 400,max_t = 700):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    #print(state_size, action_size)
    num_agents = len(env_info.agents)
    agent = Agent_multi(state_size=state_size, action_size=action_size, number_agents=num_agents,random_seed=10,sigma=0.05)
    scores_deque = deque(maxlen=100)
    scores_overall = []
    max_score = -np.Inf
    individual_scores = defaultdict(list)
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        time = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            time += 1
            agent.step(state, action, reward, next_state, done,time)
            state = next_state
            scores += reward
            if np.any(done):
                break
        agnet_num = 0
        score = np.mean(scores)
        scores_deque.append(score)
        scores_overall.append(score)
        for i in scores:
            individual_scores[agnet_num].append(i)
            agnet_num += 1
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f} \t Max: {:.2f} \t Min: {:.2f}'.format(i_episode, np.mean(scores_deque), score, np.max(scores),np.min(scores)),
              end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_multi.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_multi.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    print(scores)
    env.close()


def trained_qmodel_run(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    # env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations  # get the current state (for each agent)
    num_agents = len(env_info.agents)
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    agent = Agent_multi(state_size=state_size, action_size=action_size, number_agents=num_agents, random_seed=10,
                        sigma=0.05)
    scores = np.zeros(num_agents)
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    # initialize the score (for each agent)
    count = 0
    scores = np.zeros(num_agents)
    while True:
        actions = agent.trained_act(states)  # select an action (for each agent)
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {} , individual scores: {}'.format(np.mean(scores),scores))
    env.close()


if __name__ == "__main__":
    parser = lets_add_arguments()
    parser.parse_args()
    lets_get_arguments(parser)
