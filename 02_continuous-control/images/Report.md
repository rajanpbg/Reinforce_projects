[//]: # (Image References)
[image1]: https://github.com/rajanpbg/Reinforce_projects/raw/master/02_continuous-control/images/hyper_parameters.png "Hyperparameter Single "
[image2]: https://github.com/rajanpbg/Reinforce_projects/raw/master/02_continuous-control/images/hyper_parameters_1.png "Hyperparameter Avg single"
[image3]: https://github.com/rajanpbg/Reinforce_projects/raw/master/02_continuous-control/images/multi_arm_final_single_agent.gif "Multi Agents "
[image4]: https://github.com/rajanpbg/Reinforce_projects/raw/master/02_continuous-control/images/multi_arm_final_single_agent.gif "Multi Agents single "
[image5]: https://github.com/rajanpbg/Reinforce_projects/raw/master/02_continuous-control/images/rewards_multi.png "rewardmulti"
[image6]: https://github.com/rajanpbg/Reinforce_projects/raw/master/02_continuous-control/images/rewards_multi_avg.png "rewardmulyiavg"
[image7]: https://github.com/rajanpbg/Reinforce_projects/raw/master/02_continuous-control/images/single_agent_avg.png "singleagnetavg"
# Project 2: Continuous Control

### Introduction
In this environment, we have to train  a double-jointed arm   so that it can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of   agent is to maintain its position at the target location for as many time steps as possible.

Lets start understanding the environment  for single agent 

     we have 
     33  values for state (1,33)
      4  values for one action     (1,4)  --> continus values 
      
 
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
minibatch size | 256
discount factor | 0.99  
tau (soft update) | 1e-3
learning rate actor | 1e-3
learning rate critic | 1e-3
L2 weight decay | 0
UPDATE_EVERY | 20
NUM_UPDATES | 10
EPSILON | 1.0
EPSILON_DECAY | 1e-6
NOISE_SIGMA | 0.05


### For single Agent score :

Using diffrent hyperparamaters 

Looking at grpah LR#0.0001 and sigma#0.05 is doing Good

![image1] 

![image2]  

Using the best parameters (LR#0.0001, sigma#0.05) for the single model 
fter trianing the probelem in DDPG technique we got below score over 300 eoisides .. The model seems to be doing better after 180 episode

![image7]  

### Multi Agnet with same parameters

For Multi Agent score :

For all agents  individual score 

![image5]  

all agents AVG score 
After trianing the probelem in DDPG technique we got below score over 300 eoisides .. The model seems to be doing better after 180 episode 


![image6]  


    
## Idea of Future Work

using below algorithms and trying to see

trying a DDPG with Expirence reply buffer 

This DDPG implementation was very dependent on hyperparameter, noise settings and random seed. Solving the environment using PPO, TRPO or D4PG might allow a more robust solution to this task.
