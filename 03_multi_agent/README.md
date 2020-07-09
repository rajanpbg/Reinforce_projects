[//]: # (Image References)

[image1]: https://github.com/rajanpbg/Reinforce_projects/raw/master/03_multi_agent/images/model_before_train.gif "Untrained Agent"
[image2]: https://github.com/rajanpbg/Reinforce_projects/raw/master/03_multi_agent/images/model_after_train.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Solved the Environment
The task is episodic, and in order to solve the environment,  your agent must get an average score of +0.5 over 100 consecutive episodes.
Please see the jif files which shows how the model is running after training 

![image2]
### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

### state and action spaces 
The action space for the model is having 2 values in list. All are continous. 
Since we have 2 agents it will be 2x2 
  ex:  [[ 0.09969839 -1.00243755] [ 0.09969839 -1.00243755]]
  
For state we have shape of (2,24). So each agents takes 24 values as a state 

### python modules needed 

pandas, numpy , mlagents, gym, matplotlib , unityagents, torch 

## In unix Shell 

you can also train the model in unix shell or in a headless cloud by following process 

clone the repo 

#### Running environment 
$ python  main.py --train ---> to train model 

$ python  main.py --trained-model  --> to watch the trained model 

other option

$ python  main.py  -h 

 optional arguments:
 
  -h, --help            show this help message and exit
  
  --display-example-run
                        Example Run
  
  --get-size-of-space   Get Size of environment
  
  --get-sample-space-actions
                        Get sample space
  
  --trained-model       Run an exampke from saved model
  
  --train               Run training


## Configuring AWS conole for nohead  training on Deep Learing AMI

https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/Training-on-Amazon-Web-Service.md

sudo apt install -y xserver-xorg mesa-utils

sudo nvidia-xconfig --query-gpu-info

sudo nvidia-xconfig --busid=PCI:0:30:0 --use-display-device=none --virtual=1280x1024

sudo Xorg :1&

export DISPLAY=:1

python main.py --train 

## For Azure based Deep learning image vms for headless training 

nvidia-xconfig --query-gpu-info 

apt install -y xserver-xorg mesa-utils

cat /etc/X11/xorg.conf

nvidia-xconfig -a --use-display-device=None --virtual=1280x1024cat /etc/X11/xorg.conf 

/usr/bin/X :0 &

export DISPLAY=:0
