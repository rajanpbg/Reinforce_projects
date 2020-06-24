[//]: # (Image References)

[image1]: ./images/basic_dqn.png "Basic DQN"
[image2]: ./images/hyperparam_search.png "hyper"
[image3]: ./images/decide_model.png "decide"
[image4]: ./images/trainedmodel.png "final"
#  Banana Colletor: Details 

### Introduction
The aim of the Project is to collect the number of bananas in a given timeframe of 300.
Inorder to achive this we need to train a suitable model which will solve the  problem and get  13+ score for this enviroment. 

lets start understanding the enviroment. 
   
    we have 
      37 states 
      4 actions
    
To solve this we will start with a basic DQN model.
    
    
    In basic DQN model. we will use the Neural networks to decide  a optimal  policy for the enviroment.
    So  the plan is to create the neural network which will take state and return value function used to take best actions 
    
    So inorder to get a model. we need to train the neural network. The process of training the neural network is 
       1) Colelcting the experince replays.. (So we collect all expirences   which includes state,action,reward,next_state,done 
            so  that mean this single  tuple tells us on which state it got which reward and next state
            We collect all these details and we train the model using these expirence tuples 
            
       2) We use 2 networks for training .. 
             a) Local network: this is where we  will send the expirence tuple and we get actions values
             b) target network: As we know inorder to caliculate loss we also need to values of next state and next reward it gets
          we cannot do this using single network  beacuse we will be applying gradient descent for error correction. so both values never converge 
          This is called Fixed Q network 
          
After trianing the DQN technique we got below  score over 1000 eoisides .. 
The model seems to be doing better after 600 episode .. But we it has not solved the issues 

![Basic DQN][image1]

This model useed 
   1) Neural netowork : 
                a) single linear layer 
   2) Expirence reply 
   3) Fixed Q target 
 
### Solving the issue with Expirence replay techinque  and  more layers 
There are 2 improvements 
    
    1) Adding priorities to Expirence replays..
    2) Adding more layers to neural network

In normal Expirence reply technique we used equal random policies. 
No weights(probabilities) given to specific frame. 
But in this technique we use Prioritized expirence reply. 
Which mean we add probabilities to each frame(state)..

We add more weight to frame where we  got more error. So it get trained better

But this needs bias correction.so we divide the number of samples x probablities raised 
to power of beta Also while adding probabilities we each state we will 
normalize it via (probabiliy /sum(probabluities)^ alpha 
We also need to add minimum probabilities to state so that it wonot be ignored 


    for prob  to each state -- > ( p(i)^alpha / sum(p)^alpha)
    
    for normalization of weights 
             ( (1 /Nnumber of samples) *  (1 / p) ) ^ beta
        where beta -- is conistant and  will be increased from 0.1 to 1
        when beta is 1  we are fully compansating the bias

Also we need to do hypre paramter search to make sure which one fits better . So we use 
below hyper parameters .. learning rate (lrates) and epsolon decay (eps_decay)
lrates = [1e-4, 2e-4, 5e-4]
eps_decay = [0.99,0.95,0.9]

Learning rate is more importnant paramter which helps the neural network to understand at which pace it has to update weights EPS_decay is another parameter which tell when to stop taking random actions and when to start actions from model


![decide][image3]

![hyper][image2]

As you can see from above plots we get better results if we select 

    Learning rate : 5e-4
    Epsolon_deacy : 0.9

After  choosing the model we add these paramters and start the  intilize Agent  and train the model 

This is final  score graph for the model.

![decide][image4]

As you can see now the model has score of more than 13 (13+)

The model has solved the  problem 