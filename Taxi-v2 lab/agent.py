import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self.epsilon = 1.0        # Exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001

        self.alpha = 0.1          # Learning rate
        self.gamma = 0.9          # Discount rate
        self.num_episodes = 1

    def _get_epsilon_greedy_policy(self, state):
        """ Get actions probabilities for the epsilon-greedy policy """

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min) 
        policy = np.ones(self.nA) * (self.epsilon / self.nA)
        best_action = np.argmax(self.Q[state])
        policy[best_action] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        policy = self._get_epsilon_greedy_policy(state)
        action = np.random.choice(self.nA, p=policy)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # Q-Learning update rule

        if done:
            self.Q[state][action] += self.alpha * (reward + (self.gamma * 0) - self.Q[state][action])
            self.num_episodes += 1
        else:
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
