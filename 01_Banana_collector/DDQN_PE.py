#! /usr/bin/python
# Author: Govinda Rajan
# Purpose :  This is the model Agent for training the Banana Collector
# this is the Base Model  Consists of
#       Fixed Q-Networks ,  Expirence Replay
# So we collect expirence reply from model using random action and we store them in deque which will be used for training
# Then we use  the Neural Network (Model.py) to train on this expirences and use it as Brain to model
######################################
import numpy as np
import random
from collections import namedtuple, deque

from modified_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,learnrate=LR):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.beta = 0.4  ## beta chnages by steps
        self.learnrate = learnrate


        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learnrate)

        # Replay memory
        self.memory = Priority_ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        beta_now = min(1.0,self.beta)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE,beta_now)
                self.learn(experiences, GAMMA)
        ## we have 1000 time steps  till beta becomes 1
        self.beta = self.beta + 0.0006

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indicies, weights = experiences

        # Get max predicted Q values (for next states) from target model
        ## DQN Model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #print(Q_targets)
         
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # print(Q_expected)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        loss =  ( Q_expected - Q_targets ).pow(2) * weights
        ## lets get the  updates to priorities
        prios = loss + 1e-5
        loss = loss.mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        ## update the probabilties of reply buffer
        self.memory.update_priorities(indicies, prios.data.cpu().numpy())
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Priority_ReplayBuffer:
    """Prioritized Replay Buffer.  In normal expirence buffer we learn from each exp tuple with same
    priority. But in Prioritized action replay.. we assign priorities to each  state.  So by this we leanrn more on
    the exp which has more loss.. But this has one issue if we learn only from the ones which we have more loss..
    the model misses normal  expirences then we will be more biased. So we add some normalization

     for prob  to each state -- > ( p(i)^alpha / sum(p)^alpha)
     for normalization
             ( (1 /Nnumber of samples) *  (1 / p) ) ^ beta
        where beta -- is conistant and  will be increased from 0.1 to 1
        when beta is 1  we are fully compansating the bias
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, prob_alpha=0.6):
        self.maxdata = buffer_size
        self.prob_alpha = prob_alpha
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.pos = 0


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        ## We intilize priority as default priority of 1 or the max in the prioity list
        max_prio = self.priorities.max() if self.memory else 1.0
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.maxdata:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.maxdata


    def sample(self,batch_size, beta=0.4):
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) == self.maxdata:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        ## lets choose samples based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        for idx in  indices:
            states = self.memory[idx]

        samples = [self.memory[idx] for idx in indices]

        ## lets caliculate weight
        len_memory = len(self.memory)
        weights = (len_memory * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)


        states = torch.from_numpy(np.vstack([self.memory[idx].state for idx in indices if self.memory[idx] is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[idx].action for idx in indices if self.memory[idx] is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].reward for idx in indices if self.memory[idx] is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].next_state for idx in indices if self.memory[idx] is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([self.memory[idx].done for idx in indices if self.memory[idx] is not None]).astype(np.uint8)).float().to(
            device)
        weights = torch.from_numpy(np.vstack(weights)).to(device)

        return (states, actions, rewards, next_states, dones, indices , weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)