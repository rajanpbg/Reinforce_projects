[//]: # (Image References)
[image1]: https://github.com/rajanpbg/Reinforce_projects/raw/master/03_multi_agent/images/rewards.png "Rewardsavg "
[image2]: https://github.com/rajanpbg/Reinforce_projects/raw/master/03_multi_agent/images/rewards_avg.png "Rewards"
 
# Project 3: Collaboration and Competition

### Introduction
In this environment, we have to train  a Tennis playing agents,  so two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

Lets start understanding the environment  for   agent 

     we have 
     24  values for state (2,24)  each agent gets a state value of 24
      2  values for one action     (2,2)  for 2 agents --> continus values 
      
we collect the states/action/reward from both  agents and store them in replybuffer. 
We will use single policy/critic network for both the agents
 
To solve this  we  will use DDPG (Deep Deterministic Policy Gradient)

Start by visualizing DDPG as an algorithm with the same architecture as DQN. The training process is very similar: the agent collects experiences in an online manner and stores these online experience samples into a replay buffer. On every step, the agent pulls out a mini- batch from the replay buffer that is commonly sampled uniformly at random. The agent then uses this mini-batch to calculate a bootstrapped TD target and train a Q-function. 


      DDPG   looks like a DQN model ..  DQN for continous action space ,  DDPG uses many of the same techniques found in DQN
            replay buffer 
            off-policy training 
            Fixed Q-Targets
            
      In addtion to above DDPG also have policy network which will give action values
      
      DDPG  is also seen as Actor critic model. As it uses the below technique to train model 
      
     1)  Actor : Pocliy network --> takes state as input  outputs Action -- > it has 2 neural networks.. One for Local and other for target  

     2)  Critic : Value function --> takes state and action as input and give the value of the state --> it also has 2 neural nets   one for local and other for target  

 

        Exploration: 

          We add a gaussian noise   to value so that we can do exploration  

        Training: 

              Critic:  

                 To train critic we follow same DQN technique  

                      We caliculate next action  using actor taget network  it takes next state as input
                      Q_target caliculated using the  critic target network ( value fucntion) using     (input next_staate /next action from policy network(actor_target network)..  
                      

                      Then total Q_tar =  rewards + ( gamma * Q_target * (1 – done)) 

                      Then expected q_val caliculated via local network  of critic :  
                      
                      Q_exp = critic_local.network(state,action) 

                      Loss = mse( Q_exp – Qtar)  then backpropagate and train model  

              Actor: 

                  TO train the actor  

                      Caliculate action value via local net action = actor_local(state) 

                      Feed state and action to critic_local   q_val = critic_local(state,action) 

                      Take mean of it   q_val.mean() then propagate that back        

     We make soft update rather than updating alll parameter values at once  to target network
     


Hyperparameters:

Parameter | Value
--- | ---
replay buffer size | int(1e6)
minibatch size | 128
discount factor | 0.99  
tau (soft update) | 1e-3
learning rate actor | 1e-3
learning rate critic | 1e-3
L2 weight decay | 0
NOISE_SIGMA | 0.2


### Model performence :

The model seems to be not able to perform better untill the 1000 episodes. Once after  1000 episodes it seems to be
started learning. So it is using OUnosie to  explore the diffrent actions to  perform better. 
At one point it has reached to a score of +2.7 which confirms that model is doing better 



### Agent Reward plot

For  Agent score :

score plotted over 2000 episodes  

![image1]  

all agents AVG score 
After trianing the probelem in DDPG technique we got below score over 2000 eoisides .. The model seems to be doing better after 1750 episode 


![image2]  


    
## Idea of Future Work

using below algorithms and trying to see

trying a DDPG with Expirence reply buffer 

This DDPG implementation was very dependent on hyperparameter, noise settings and random seed. Solving the environment using PPO, TRPO or D4PG might allow a more robust solution to this task.
