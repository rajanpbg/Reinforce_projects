#!/usr/bin/python
# Author: Govinda
## Discretization probelm
#################################
import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import argparse

def lets_add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-example-run", help="Example Run", action="store_true")
    parser.add_argument("--get-size-of-space", help="Get Size of environment", action="store_true")
    parser.add_argument("--get-sample-space-actions", help="Get sample space", action="store_true")
    parser.add_argument("--display-sample-grid", help="Display Grid", action="store_true")
    parser.add_argument("--train-with-10bins", help="Train Q model with 10 bins", action="store_true")
    parser.add_argument("--train-with-20bins", help="Train Q model with 10 bins", action="store_true")
    parser.add_argument("--Qmodel-run", help="Run an exampke from saved model", action="store_true")
    return parser

def lets_get_arguments(parser):
    args = parser.parse_args()
    if args.display_example_run:
        display_example_run()

    if args.get_size_of_space:
        get_size_of_space()
    if args.get_sample_space_actions:
        get_sample_space_actions()

    if args.display_sample_grid:
        display_sample_grid()

    if args.train_with_10bins:
        train_with_10bins()

    if args.train_with_20bins:
        train_with_20bins()

    if args.Qmodel_run:
        let_make_test_run()


### Example run
def display_example_run():
    import time
    count = 0
    state = env.reset()
    score = 0
    for t in range(200):
      action = env.action_space.sample()
      env.render(mode='human')
      time.sleep(1.0/30)
      state, reward, done, _ = env.step(action)
      score += reward
      count +=1
      if done:
         break
    print('Final score:', score)
    print('total steps:', count )
    env.close()

def get_size_of_space():
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)

def get_sample_space_actions():
    print("State space samples:")
    print(np.array([env.observation_space.sample() for i in range(10)]))
    print("Action space:", env.action_space)
    # Generate some samples from the action space
    print("Action space samples:")
    print(np.array([env.action_space.sample() for i in range(10)]))

#get_sample_space_actions()


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    """Visualize original and discretized samples on a given 2-dimensional grid."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Show grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)

    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples

    ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
    ax.plot(locs[:, 0], locs[:, 1], 's')  # plot discretized samples in mapped locations
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)),
                                        colors='orange'))  # add a line connecting each original-discretized sample
    ax.legend(['original', 'discretized'])
    plt.show()


def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid


##try One with 10 bins


def display_sample_grid():
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
    state_samples = np.array([env.observation_space.sample() for i in range(10)])
    discretized_state_samples = np.array([discretize(sample, state_grid) for sample in state_samples])

    print(state_samples,discretized_state_samples)
    visualize_samples(state_samples, discretized_state_samples, state_grid,
                      env.observation_space.low, env.observation_space.high)

#display_sample_grid()


class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)

        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate  # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon

        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                                                                   (reward + self.gamma * max(self.q_table[state]) -
                                                                    self.q_table[self.last_state + (self.last_action,)])

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action


def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes + 1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        # Save final score
        scores.append(total_reward)

        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores

def train_with_10bins():
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
    q_agent = QLearningAgent(env, state_grid)
    scores = run(q_agent, env)
    np.save(f"mountaincar-qtable.npy", q_agent.q_table)
    # Plot scores obtained per episode
    #plt.plot(scores); plt.title("Scores");

##try One with 20 bins
def train_with_20bins():
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
    q_agent_20 = QLearningAgent(env, state_grid)
    scores = run(q_agent_20, env)
    np.save(f"mountaincar-qtable.npy", q_agent_20.q_table)
    #plt.plot(scores)
#plt.title("Scores");
## to show plot uncomment
#plt.show()

## lets test the model

class Qagentready:
        """Q-Learning agent that can act on a continuous state space by discretizing it."""

        def __init__(self, env, state_grid):
            """Initialize variables, create grid for discretization."""
            # Environment info
            self.env = env
            self.state_grid = state_grid

            try:
                self.q_table = np.load(f"mountaincar-qtable.npy")
            except:
                exit("Qtable not found")
            print("Q table size:", self.q_table.shape)

        def preprocess_state(self, state):
            """Map a continuous state to its discretized representation."""
            # TODO: Implement this
            return tuple(discretize(state, self.state_grid))

        def reset_episode(self, state):
            """Reset variables for a new episode."""
            # Gradually decrease exploration rate
            self.epsilon *= self.epsilon_decay_rate
            self.epsilon = max(self.epsilon, self.min_epsilon)

            # Decide initial action
            self.last_state = self.preprocess_state(state)
            self.last_action = np.argmax(self.q_table[self.last_state])
            return self.last_action

        def reset_exploration(self, epsilon=None):
            """Reset exploration rate used when training."""
            self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

        def act(self, state, reward=None, done=None, mode='train'):
            """Pick next action and update internal Q table (when mode != 'test')."""
            state = self.preprocess_state(state)
            action = np.argmax(self.q_table[state])
            return action


def let_make_test_run():
    import time
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
    learnet_agent = Qagentready(env,state_grid)
    state = env.reset()
    score = 0
    count = 0
    for t in range(200):
       action = learnet_agent.act(state, mode='test')
       env.render(mode='human')
       time.sleep(1.0/30)
       state, reward, done, _ = env.step(action)
       score += reward
       count += 1
       if done:
         break
    print('Final score:', score)
    print('total steps:', count)
    env.close()

if __name__ == "__main__":
    # Set plotting options
    plt.style.use('ggplot')
    np.set_printoptions(precision=3, linewidth=120)
    # Create an environment and set random seed
    env = gym.make('MountainCar-v0')
    env.seed(505);
    parser = lets_add_arguments()
    parser.parse_args()
    lets_get_arguments(parser)

