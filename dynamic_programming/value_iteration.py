import time
import numpy as np
import matplotlib.pyplot as plt
from Frozen_Lake import FrozenLakeEnv


class ValueIteration:
    def __init__(self, env, epsilon=1e-6):
        self.env = env
        self.epsilon = epsilon
        self.v_values = self.init_v_values()

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

    def update_q_values(self, gamma):
        """
        Computes the Q-values for all state-action pairs based on the current value function.

        Parameters:
        - v_values: The value function for each state.
        - gamma: The discount factor.

        Returns:
        A dictionary of Q-values for all state-action pairs.
        """
        delta = 0
        q_values = {}
        for state in self.env.get_all_states():
            temp = self.v_values[state]
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
                        * self.v_values[next_state]
                    )
            if q_values[state]:
                self.v_values[state] = max(list(q_values[state].values()))

            delta = max(delta, abs(temp - self.v_values[state]))
        return delta

    def train_value_iteration(self, gamma):
        delta_history = []

        while True:
            delta = self.update_q_values(gamma)
            delta_history.append(delta)

            if delta < self.epsilon:
                break

        fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
        ax.plot(
            np.arange(len(delta_history)) + 1,
            delta_history,
            marker="o",
            markersize=4,
            alpha=0.7,
            color="#2ca02c",
            label=r"$\gamma= $" + f"{gamma}",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Delta")
        ax.legend()
        plt.tight_layout()
        plt.show()

        return self.extract_policy(gamma), len(delta_history)

    def extract_policy(self, gamma):
        policy = self.init_policy()

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
                        * self.v_values[next_state]
                    )

            if q_values[state]:
                best_action = max(q_values[state], key=q_values[state].get)
                policy[state][best_action] = 1.0

                total_prob = sum(policy[state].values())
                if total_prob != 0:
                    for action in policy[state]:
                        policy[state][action] /= total_prob
            else:
                pass

        return policy

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
best_number_iterations = -float("inf")
gammas = [0.7, 0.8, 0.9, 0.95, 0.99]

for gamma in gammas:
    value_iteration = ValueIteration(env)
    policy, num_iterations_to_converge = value_iteration.train_value_iteration(gamma)
    print(
        f"Gamma: {gamma}, Number of iterations to converge: {num_iterations_to_converge}"
    )
    if num_iterations_to_converge > best_number_iterations:
        best_mean_reward = num_iterations_to_converge
        best_gamma = gamma

print(f"Best Gamma: {best_gamma}")

env = FrozenLakeEnv()
value_iteration = ValueIteration(env)
gamma = 0.9
policy, num_iterations_to_converge = value_iteration.train_value_iteration(gamma)
total_reward, num_steps = value_iteration.test_policy(policy)
print(
    f"Total Reward with Best Policy: {total_reward} Number of steps to final state: {num_steps}"
)
