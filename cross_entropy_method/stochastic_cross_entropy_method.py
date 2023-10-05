import gym
import numpy as np
import matplotlib.pyplot as plt


class StochasticCrossEntropyAgent:
    def __init__(
        self,
        env: gym.Env,
        num_iterations: int,
        num_policies: int,
        num_trajectories: int,
        elite_quantile: float,
        sigma: float,
        t_max: int = 10**4,
    ):
        """
        Initialize the Stochastic Cross-Entropy Agent.

        Args:
            env (gym.Env): The Gym environment.
            num_iterations (int): Number of iterations for the optimization process.
            num_policies (int): Number of deterministic policies to sample in each iteration.
            num_trajectories (int): Number of trajectories to collect per policy.
            elite_quantile (float): Quantile value for selecting elite trajectories.
            sigma (float): Standard deviation for policy sampling (noise level).
            t_max (int): Maximum time steps per trajectory (default is 10,000).
        """
        self.env = env
        self.num_iterations = num_iterations
        self.num_policies = num_policies
        self.num_trajectories = num_trajectories
        self.elite_quantile = elite_quantile
        self.t_max = t_max

        self.state_dim = env.observation_space.n  # 500
        self.action_dim = env.action_space.n  # 6

        self.policy_mean = np.random.rand(self.state_dim, self.action_dim)
        self.sigma = sigma

        # Initialize a list to store rewards for each iteration
        self.iteration_rewards = []

    def sample_stochastic_policy(self) -> np.ndarray:
        """
        Sample a stochastic policy from the current policy distribution.

        Returns:
            np.ndarray: A sampled stochastic policy.
        """
        return self.policy_mean + self.sigma * np.random.randn(
            self.state_dim, self.action_dim
        )

    def sample_deterministic_policies(self) -> list:
        """
        Sample a list of deterministic policies based on the current stochastic policy.

        Returns:
            list: A list of deterministic policies.
        """
        deterministic_policies = []
        for _ in range(self.num_policies):
            deterministic_policies.append(self.sample_stochastic_policy())
        return deterministic_policies

    def sample_action(self, policy: np.ndarray, state: np.ndarray) -> int:
        """
        Sample an action from a given policy for a given state.

        Args:
            policy (np.ndarray): The policy from which to sample.
            state (np.ndarray): The current state.

        Returns:
            int: The sampled action.
        """
        action_mean = policy[state]
        action = np.argmax(action_mean)
        return action

    def run_episode(self, policy: np.ndarray) -> tuple:
        """
        Run a single episode in the environment and collect states, actions, and rewards.

        Args:
            policy (np.ndarray): The policy to follow during the episode.

        Returns:
            tuple: States, actions, and rewards for the episode.
        """
        states, actions, episode_reward = [], [], 0.0
        state = self.env.reset()

        for _ in range(self.t_max):
            action = self.sample_action(policy, state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            episode_reward += reward
            state = next_state
            if done:
                break

        return states, actions, episode_reward

    def select_elites(self, trajectories) -> list:
        """
        Sample and select elite trajectories based on their performance.

        Args:
            trajectories (list): List of trajectories to select from.

        Returns:
            list: Elite trajectories.
        """
        returns = [trajectory_reward for (_, _, trajectory_reward) in trajectories]

        quantile_threshold = np.quantile(returns, self.elite_quantile)

        elite_trajectories = [
            traj for traj in trajectories if traj[2] > quantile_threshold
        ]
        return elite_trajectories

    def update_policy(self, elite_trajectories: list):
        """
        Update the policy using the elite trajectories.

        Args:
            elite_trajectories (list): Elite trajectories.
        """
        new_policies = []

        for elite_states, elite_actions, _ in elite_trajectories:
            new_policy = np.zeros((self.state_dim, self.action_dim))

            for state, action in zip(elite_states, elite_actions):
                new_policy[state, action] += 1

            state_counts = new_policy.sum(axis=1)

            for idx, state_count in enumerate(state_counts):
                if state_count == 0:
                    new_policy[idx] = np.ones([self.action_dim]) / self.action_dim
                else:
                    new_policy[idx] = new_policy[idx] / state_count

            new_policies.append(new_policy)

        self.policy_mean = np.mean(new_policies, axis=0)

    def run(self):
        """
        Run the StochasticCrossEntropyAgent optimization process.
        """

        for iteration in range(self.num_iterations):
            deterministic_policies = self.sample_deterministic_policies()

            trajectories, rewards_per_policy = [], []
            for policy in deterministic_policies:
                reward_per_policy = 0.0
                trajectories_per_policy = [
                    self.run_episode(policy) for _ in range(self.num_trajectories)
                ]
                elite_trajectories = self.select_elites(trajectories_per_policy)

                trajectories.extend(elite_trajectories)

                reward_per_policy = np.nanmean([traj[2] for traj in elite_trajectories])

                rewards_per_policy.append(reward_per_policy)

            mean_reward = np.nanmean(rewards_per_policy)

            self.update_policy(trajectories)

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
            f"Hyperparameters:\nNum Iterations: {self.num_iterations}, Num Policies: {self.num_policies}, Elite Quantile: {self.elite_quantile}, Num Trajectories: {self.num_trajectories}"
        )
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    agent = StochasticCrossEntropyAgent(
        env,
        num_iterations=200,
        num_policies=5,
        num_trajectories=300,
        elite_quantile=0.5,
        sigma=0.1,
    )
    agent.run()

    env.close()
