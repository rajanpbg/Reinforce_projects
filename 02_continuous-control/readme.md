[//]: # (Image References)

[image1]: images/multi_joint.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]




## Configuring AWS conole for nohead  training 
https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/Training-on-Amazon-Web-Service.md
sudo nvidia-xconfig --query-gpu-info
sudo nvidia-xconfig --busid=PCI:0:30:0 --use-display-device=none --virtual=1280x1024
sudo Xorg :1&
export DISPLAY=:1
python main.py --train 
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
