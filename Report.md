# Learning Algorithm
Agent uses deep Q-network algorythm (DQN) for learning how to behave and maximize it's score in a provided environmet.

The project is split into three main parts:
1. scripts/main.py - runs the interaction loop between an agent and an envitonment over many episodes and time steps.
At the beginnig of each episod the environment state is reset which produces initial state for the agent.
Agent observes the state and selects an action for interaction. After executing previously selected action environment produces the next state, reward and boolean value which notifies about the end of the episode.
Agent continues the loop of selecting an action and interacting with the environment untill it reaches the terminal state or exceeds the maximum amount of timesteps.
The procees mentioned above repeats until maximum amount of episodes is reached or the agent reaches the average score greater than 13 which is considered as the environment being solved.
Apart of the trainning loop the main.py script tracks and plots the agent's average score and saves agent's network weights in the case when the enviroment is solved.
"train" function contains the following set of hyperparameters:  
    * n_episodes (int): maximum number of training episodes.
    * max_t (int): maximum number of timesteps per episode.
    * eps_start (float): starting value of epsilon, for epsilon-greedy action selection.
    * eps_end (float): minimum value of epsilon.
    * eps_decay (float): multiplicative factor (per episode) for decreasing epsilon.

2. scripts/dqn_agent - definens the Agent class which implements [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) algorythm with two extensions (Target Networks and Double DQN). 

 * The Agent class has two main functions:

 * 2.1 "act" - selects an action for a given state using epsilon-greedy action selection. Epsilon-greedy action selection helps the agent to balance between exploration and exploitantion.
The are two possibilities of how action can be selected. 
Either the agent selects action randomly (exploration) or uses function aproximation algorythm for selecting an action with a highest value for a given state (exploitation, Q function).
One of the ideas of DQN is to use supervised learning algorythm such as deep neural networks as a function aproximator, which has a state as an input parameters of the DNN and action values as an output.

 * 2.2 "learn" - observes a new state and a reward after selecting an action which gives the agent a possibility to learn and improve the action selection function/network from its expirience. Agent keeps track of it's previous expirience by storing the last k (state, action, next state, reward) tupels in it's replay buffer. Later a minibutches selected random-uniformly fro the replay buffer are used for traning DNN to map states to actions values (Q function).

 * 2.3. The are two extensions to the original DQN algorythm:
 * 2.3.1. Target network. This technic helps to stabilise trainning motivated by the fact that the optimal Q value function learned by the neural network is constantly changing and dependent on Q value functions itself. The main idea of the target network approach is to introduce a second, target network, which is a lagged copy of the original network for stabilizing the target Q value function.

 * 2.3.2 Double DQN. Due to the noises in the environment and errors in fuction approximation using neural networks agent may not fully explore environment which can lead to Q values being overestimated. The Double DQN algorithm reduces the overestimation of Q-values by selecting the best action for a state using one network and estimating the Q-value by using second network. Local and target networks can be used correspondingly for action selection and target Q-value estimation.
 * the agent and replay buffer contains the following set of hyperparameters:
   * BUFFER_SIZE = int(1e5)  # replay buffer size
   * BATCH_SIZE = 64  # minibatch size
   * GAMMA = 0.99  # discount factor
   * TAU = 1e-3  # for soft update of target parameters
   * LR = 5e-4  # learning rate
   * UPDATE_EVERY = 4  # how often to update the network

3. scripts/model.py - defines neural network model.
The network model contains three linear layes, input layer with size (state_size, fc1_units=64), hidden layer with size (fc1_units=64, fc2_units=64) and output layer with size (fc2_units, action_size). Relu activation function is used for the input and the hidden layer.

# Plot of Rewards
Environment is usually solved in less than 600 episodes:

![Scores:](/images/scores_1.png)

![Scores:](/images/scores_2.png)
# Ideas for Future Work

Prioritezed expirience replay. The intuition behind this technic is that some of the expiriences are more informative for the agent and as a result the agent can learn faster from such expiriences. Main idea is to determine how informative is a given expirience record (it can be measured wiht absolute difference between expected and actual Q-value) and increase the probability of sampling of such expiriences.

