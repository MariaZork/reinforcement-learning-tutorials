import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


class SmoothCrossEntropyAgent:
    def __init__(
        self,
        env: gym.Env,
        num_iterations: int,
        num_trajectories: int,
        elite_quantile: float,
        laplace_smoothing: float = None,
        policy_smoothing: float = None,
        t_max: int = 10**4,
    ):
        """
        Initialize the SmoothCrossEntropyAgent.

        Args:
            env (gym.Env): The Gym environment.
            num_iterations (int): Number of SmoothCrossEntropyAgent iterations.
            num_trajectories (int): Number of trajectories to collect per policy.
            elite_quantile (float): Quantile value for selecting elite trajectories.
            laplace_smoothing (float): Laplace smoothing rate.
            policy_smoothing (float): Policy smoothing rate.
            t_max (int): Length of one episode.
        """
        self.env = env
        self.num_iterations = num_iterations
        self.num_trajectories = num_trajectories
        self.elite_quantile = elite_quantile
        self.state_dim = env.observation_space.n  # 500
        self.action_dim = env.action_space.n  # 6
        self.laplace_smoothing = laplace_smoothing
        self.policy_smoothing = policy_smoothing
        self.t_max = t_max

        # Initialize the policy table with uniform probabilities initially
        self.policy_table = np.ones((self.state_dim, self.action_dim)) / self.action_dim

        # Initialize a list to store rewards for each iteration
        self.iteration_rewards = []

    def choose_action(self, state: int) -> int:
        """
        Choose an action based on the current policy.

        Args:
            state (int): Current state.

        Returns:
            int: Selected action.
        """
        # Select action based on the policy table
        action_probabilities = self.policy_table[state]
        return np.random.choice(self.action_dim, p=action_probabilities)

    def run_episode(self) -> tuple:
        """
        Run a single episode in the environment and collect states, actions, and rewards.

        Returns:
            tuple: States, actions, and rewards for the episode.
        """
        states, actions, episode_reward = [], [], 0.0
        state = self.env.reset()

        for _ in range(self.t_max):
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            episode_reward += reward
            state = next_state
            if done:
                break

        return states, actions, episode_reward

    def select_elites(self, *episodes) -> tuple:
        """
        Sample and select elite trajectories based on their performance.

        Args:
            episodes (list): States, actions and reward sum list.

        Returns:
            tuple: Elite states and elite actions.
        """
        elite_states, elite_actions, elite_rewards = [], [], []
        states_batch, actions_batch, rewards_batch = zip(*episodes)

        quantile_threshold = np.quantile(rewards_batch, self.elite_quantile)

        for i, reward in enumerate(rewards_batch):
            if reward > quantile_threshold:
                elite_states.extend(states_batch[i])
                elite_actions.extend(actions_batch[i])
                elite_rewards.append(reward)
        return elite_states, elite_actions, elite_rewards

    def update_policy(self, elite_states: list, elite_actions: list):
        """
        Update the policy using the elite trajectories.

        Args:
            elite_states (list): Elite states.
            elite_actions (list): Elite actions.
        """
        new_policy = np.zeros((self.state_dim, self.action_dim))

        # Count the occurrences of actions in elite trajectories
        for state, action in zip(elite_states, elite_actions):
            new_policy[state, action] += 1

        state_counts = new_policy.sum(axis=1)

        for idx, state_count in enumerate(state_counts):
            if state_count == 0:
                new_policy[idx] = np.ones([self.action_dim]) / self.action_dim
            else:
                new_policy[idx] = new_policy[idx] / state_count

        if self.laplace_smoothing:
            self.policy_table = (new_policy + self.laplace_smoothing) / (
                np.sum(new_policy, axis=1, keepdims=True)
                + self.laplace_smoothing * self.action_dim
            )

        elif self.policy_smoothing:
            self.policy_table = (
                self.policy_smoothing * new_policy
                + (1 - self.policy_smoothing) * self.policy_table
            )

    def run(self):
        """
        Run the SmoothCrossEntropyAgent optimization process.
        """

        for iteration in range(self.num_iterations):
            episodes = [self.run_episode() for _ in range(self.num_trajectories)]

            elite_states, elite_actions, elite_rewards = self.select_elites(*episodes)

            self.update_policy(elite_states, elite_actions)

            mean_reward = np.mean(elite_rewards)
            print(
                f"Iteration {iteration + 1}/{self.num_iterations} - Total Reward: {mean_reward:.2f}"
            )
            self.iteration_rewards.append(mean_reward)

        self.plot_learning_curve()

    def plot_learning_curve(self):
        """
        Plot the learning curve (rewards over iterations).
        """
        plt.figure(figsize=[8, 4])
        plt.plot(
            range(1, self.num_iterations + 1),
            self.iteration_rewards,
            label="Learning Curve",
            marker="o",
        )

        plt.xlabel("Iteration")
        plt.ylabel("Total Reward")
        plt.title(
            f"Hyperparameters:\nNum Iterations: {self.num_iterations}, Elite Quantile: {self.elite_quantile}, Num Trajectories: {self.num_trajectories}"
        )
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    agent = SmoothCrossEntropyAgent(
        env, num_iterations=200, num_trajectories=300, elite_quantile=0.5
    )
    agent.run()

    env.close()
