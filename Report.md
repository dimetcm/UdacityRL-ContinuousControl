# Learning Algorithm
Agent uses deep deterministic policy gradiend algorithm (DDPG)(https://arxiv.org/abs/1509.02971). DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum was used as a base algorithm implementation adopted to the "Reacher" environment, with minor modifications.

### Hyperpameters
The algorithm has the folloving set of hyperparameters:
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay

### Model Configuration
#### Actor
| Layer | In | Out
| ----------- | ----------- |----------- |
| Linear | state_size | 400
| BatchNorm1d | 400
| ReLU | 400 | 400
| Linear | 400 | 300
| ReLU | 300 | 300
| Linear | 300 | action_size
| tanh | action_size | action_size

#### Critic
| Layer | In | Out
| ----------- | ----------- |----------- |
| Linear | state_size | 400
| ReLU | 400 | 400
| Linear | 400 + action_size | 300
| ReLU | 300 | 300
| Linear | 300 | 1

### Additional extentions
#### Experience replay buffer
Agents use shared expirence replay buffer.

#### Gradient clipping
Gradient clipping technique helps to deal with the irregular loss landscape of the model.

#### Distrubuted distributional deterministic policy gradients
Since the environment containts 20 simultenios agents it was reasonable to apply D4PG for speeding up the agents training procees by sharing expirience and actor/critic network weights. Probably it's the most significant addition to the algorithm since the agents show a stable and significant imporvement in the average learning score. Also due to the fact that every agent starts in a randon environmen state the learning process tends to generalize very well for different tasks like reaching spere that moves clockwise or counterclockwise or almost stationary. Which seems to be a problem for a single agent learning procees since the environment dynamics changes only after completing an episode.           
#### Ornstein–Uhlenbeck noise
Ornstein–Uhlenbeck process can be applied to the algorithm for generating temporally correlated exploration, but it's disabled by default since it doesn't add much to the agents learning process.

# Plot of Rewards
Average agents's score stabilizes after 80th episode reaching the average reward around 36 points. 

![Scores:](/images/scores.png)
# Ideas for Future Work

### Comparing performance to other algoritnms, like PPO. Addinig prioritized expirience replay.
### Testing the performance of the algorithm in a "reacher" Reacher environment, for example with the arm containing four joints 

