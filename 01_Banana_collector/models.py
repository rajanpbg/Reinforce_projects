#!/usr/bin/python
# Author : Govinda
# This module is used to provide the base models for training the agent of Banana Collector
##################
#lets import pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F


## there is 1 Agnent with 4 actions and state size of 37.. So lets use  nn layers
## The DQN model
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, action_size)
         

    def forward(self, state):
        """Build a network that maps state -> action values."""  
        return self.fc1(state)

