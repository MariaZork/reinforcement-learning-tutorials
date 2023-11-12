import time
import random
import numpy as np
from Frozen_Lake import FrozenLakeEnv

seed = 1
random.seed(seed)
np.random.seed(seed)


class PolicyIteration:
    def __init__(self, env, iter_n, eval_iter_n):
        """
        Initializes the PolicyIteration class with the given environment and parameters.

        Parameters:
        - env: The environment to be used for policy iteration.
        - iter_n: The number of policy iteration steps.
        - eval_iter_n: The number of iterations for policy evaluation.

        """
        self.env = env
        self.iter_n = iter_n
        self.eval_iter_n = eval_iter_n

    def get_q_values(self, v_values, gamma):
        """
        Computes the Q-values for all state-action pairs based on the current value function.

        Parameters:
        - v_values: The value function for each state.
        - gamma: The discount factor.

        Returns:
        A dictionary of Q-values for all state-action pairs.
        """
        q_values = {}
        for state in self.env.get_all_states():
            q_values[state] = {}
            for action in self.env.get_possible_actions(state):
                q_values[state][action] = 0
                for next_state in self.env.get_next_states(state, action):
                    q_values[state][action] += self.env.get_transition_prob(
                        state, action, next_state
                    ) * self.env.get_reward(state, action, next_state)
                    q_values[state][action] += (
                        gamma
                        * self.env.get_transition_prob(state, action, next_state)
                        * v_values[next_state]
                    )
        return q_values

    def init_policy(self):
        """
        Initializes a uniform random policy for all states.

        Returns:
        A dictionary representing the initial policy.
        """
        policy = {}
        for state in self.env.get_all_states():
            policy[state] = {}
            for action in self.env.get_possible_actions(state):
                policy[state][action] = 1 / len(self.env.get_possible_actions(state))
        return policy

    def init_v_values(self):
        """
        Initializes the value function for all states to zero.

        Returns:
        A dictionary with zero values for all states.
        """
        v_values = {}
        for state in self.env.get_all_states():
            v_values[state] = 0
        return v_values

    def policy_evaluation_step(self, v_values, policy, gamma):
        """
        Performs a single step of policy evaluation to update the value function.

        Parameters:
        - v_values: The current value function.
        - policy: The current policy.
        - gamma: The discount factor.

        Returns:
        A dictionary with updated value function for all states.
        """
        q_values = self.get_q_values(v_values, gamma)
        new_v_values = self.init_v_values()
        for state in self.env.get_all_states():
            new_v_values[state] = 0
            for action in self.env.get_possible_actions(state):
                new_v_values[state] += policy[state][action] * q_values[state][action]
        return new_v_values

    def policy_evaluation(self, policy, gamma):
        """
        Iteratively evaluates the policy to update the value function.

        Parameters:
        - policy: The policy to be evaluated.
        - gamma: The discount factor.

        Returns:
        A dictionary of Q-values after policy evaluation.
        """
        v_values = self.init_v_values()
        for _ in range(self.eval_iter_n):
            v_values = self.policy_evaluation_step(v_values, policy, gamma)
        q_values = self.get_q_values(v_values, gamma)
        return q_values

    def policy_improvement(self, q_values):
        """
        Improves the policy based on the Q-values by selecting the action with the highest Q-value for each state.

        Parameters:
        - q_values: The Q-values for all state-action pairs.

        Returns:
        A dictionary representing the improved policy.
        """
        policy = {}
        for state in self.env.get_all_states():
            policy[state] = {}
            argmax_action = None
            max_q_value = float("-inf")
            for action in self.env.get_possible_actions(state):
                policy[state][action] = 0
                if q_values[state][action] > max_q_value:
                    argmax_action = action
                    max_q_value = q_values[state][action]
            policy[state][argmax_action] = 1
        return policy

    def train_policy_iteration(self, gamma):
        """
        Performs policy iteration to learn the optimal policy for the given environment.

        Parameters:
        - gamma: The discount factor.

        Returns:
        The mean reward achieved by the learned policy.
        """
        policy = self.init_policy()
        for _ in range(self.iter_n):
            q_values = self.policy_evaluation(policy, gamma)
            policy = self.policy_improvement(q_values)

        total_rewards = []

        for _ in range(1000):
            total_reward = 0
            state = self.env.reset()
            for _ in range(1000):
                action = np.random.choice(
                    self.env.get_possible_actions(state), p=list(policy[state].values())
                )
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    break

            total_rewards.append(total_reward)

        self.final_policy = policy
        return np.mean(total_rewards)

    def test_policy(self, policy):
        total_reward = 0
        state = self.env.reset()
        num_steps = 0
        while True:
            action = np.random.choice(
                self.env.get_possible_actions(state),
                p=list(policy[state].values()),
            )
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            num_steps += 1
            self.env.render()
            time.sleep(0.5)

            if done:
                break

        return total_reward, num_steps


env = FrozenLakeEnv()

best_gamma = None
best_mean_reward = -float("inf")
gammas = [0.7, 0.8, 0.9, 0.95]
best_policy = None

for gamma in gammas:
    policy_iteration = PolicyIteration(env, iter_n=100, eval_iter_n=100)
    mean_reward = policy_iteration.train_policy_iteration(gamma)
    print(f"Gamma: {gamma}, Mean Reward: {mean_reward}")
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        best_gamma = gamma
        best_policy = policy_iteration.final_policy

print(f"Best Gamma: {best_gamma}")
print(f"Best Mean Reward: {best_mean_reward}")

if best_policy is not None:
    policy_iteration = PolicyIteration(env, iter_n=100, eval_iter_n=100)
    policy_iteration.final_policy = best_policy
    total_reward, num_steps = policy_iteration.test_policy(best_policy)
    print(
        f"Total Reward with Best Policy: {total_reward} Number of steps to final state: {num_steps}"
    )
