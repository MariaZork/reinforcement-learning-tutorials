import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from itertools import product

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        """
        Initialize the policy network.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_units (list): List of hidden units in the policy network.
        """
        super(PolicyNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_units[0]), nn.ReLU()]
        for i in range(1, len(hidden_units)):
            layers.extend([nn.Linear(hidden_units[i - 1], hidden_units[i]), nn.ReLU()])
        layers.append(nn.Linear(hidden_units[-1], action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepCrossEntropyAgent:
    def __init__(
        self,
        env: gym.Env,
        num_iterations: int,
        num_trajectories: int,
        elite_quantile: float,
        policy_hidden_units: list = [128, 64],
        epsilon: float = 1.0,
    ):
        """
        Initialize the Deep Cross-Entropy Agent.

        Args:
            env (gym.Env): The Gym environment.
            num_iterations (int): Number of Cross-Entropy iterations.
            num_trajectories (int): Number of trajectories to collect per policy.
            elite_quantile (float): Quantile value for selecting elite trajectories.
            policy_hidden_units (list): List of hidden units in the policy network.
            epsilon (float): Exploration trade-off.
        """
        self.env = env
        self.env.seed(0)
        self.num_iterations = num_iterations
        self.num_trajectories = num_trajectories
        self.elite_quantile = elite_quantile
        self.policy_hidden_units = policy_hidden_units
        self.epsilon = epsilon

        self.state_dim = env.observation_space.shape[0]  # 8
        self.action_dim = env.action_space.n  # 4

        self.policy_net = PolicyNetwork(
            self.state_dim, self.action_dim, self.policy_hidden_units
        )

        print(self.policy_net)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.policy_net.to(self.device)
        self.elite_states = []
        self.elite_actions = []
        self.iteration_rewards = []

    def uniform_policy(self, action_dim):
        return np.ones((action_dim)) / action_dim

    def choose_action(self, state):
        """
        Choose an action based on the current policy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        state = torch.FloatTensor(np.array(state)).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(state)
        action_prob_nn = self.softmax(logits).detach().cpu().numpy()
        action_prob = (
            1 - self.epsilon
        ) * action_prob_nn + self.epsilon * self.uniform_policy(self.action_dim)

        action_prob /= action_prob.sum()

        action = np.random.choice(self.action_dim, p=action_prob)

        return action

    def run_episode(self):
        """
        Run a single episode in the environment and collect states, actions, and rewards.

        Returns:
            tuple: States, actions, and rewards for the episode.
        """
        states, actions, rewards = [], [], 0.0
        state = self.env.reset()

        for _ in range(10**4):
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards += reward
            state = next_state
            if done:
                break

        return states, actions, rewards

    def select_elites(self, episodes):
        """
        Select elite trajectories based on their performance.

        Args:
            episodes (list): List of (states, actions, rewards) for multiple episodes.

        Returns:
            tuple: Elite states and elite actions.
        """
        rewards = [episode[2] for episode in episodes]
        quantile_threshold = np.quantile(rewards, self.elite_quantile)
        elite_episodes = [
            episode for episode in episodes if episode[2] >= quantile_threshold
        ]
        elite_states, elite_actions, _ = zip(*elite_episodes)
        return elite_states, elite_actions

    def run(self):
        """
        Run the Deep Cross-Entropy Agent optimization process.
        """
        for iteration in range(self.num_iterations):
            episodes = [self.run_episode() for _ in range(self.num_trajectories)]
            self.elite_states, self.elite_actions = self.select_elites(episodes)
            self.policy_net.train()
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.optimizer.step()
            self.policy_net.eval()

            self.epsilon = 1 / self.num_iterations

            mean_reward = np.mean([episode[2] for episode in episodes])
            print(
                f"Iteration {iteration + 1}/{self.num_iterations} - Total Reward: {mean_reward:.2f}"
            )
            self.iteration_rewards.append(mean_reward)

            if mean_reward >= 200.0:
                print(
                    "\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}".format(
                        iteration + 1, mean_reward
                    )
                )
                break

        self.plot_learning_curve(iteration + 1)

    def compute_loss(self):
        losses = []

        for states, actions in zip(self.elite_states, self.elite_actions):
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(np.array(actions)).to(self.device)

            action_logits = self.policy_net(states)
            loss = self.loss_fn(action_logits, actions)
            losses.append(loss)

        loss = torch.stack(losses).mean()

        return loss

    def plot_learning_curve(self, iterations_num):
        """
        Plot the learning curve (rewards over iterations).
        """
        plt.figure(figsize=[8, 4])
        plt.plot(
            range(1, iterations_num + 1),
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


def grid_search():
    env = gym.make("LunarLander-v2")

    num_iterations_space = [100, 200]
    num_trajectories_space = [100, 200, 300]
    elite_quantile_space = [0.5, 0.7, 0.8, 0.9]
    policy_hidden_units = [[16], [32], [32, 16]]

    best_params = None
    best_reward = float("-inf")

    for num_iterations, num_trajectories, elite_quantile, policy_hidden_unit in product(
        num_iterations_space,
        num_trajectories_space,
        elite_quantile_space,
        policy_hidden_units,
    ):
        agent = DeepCrossEntropyAgent(
            env,
            num_iterations=num_iterations,
            num_trajectories=num_trajectories,
            elite_quantile=elite_quantile,
            policy_hidden_units=policy_hidden_unit,
        )
        agent.run()

        final_reward = np.mean(agent.iteration_rewards)

        if final_reward > best_reward:
            best_reward = final_reward
            best_params = (num_iterations, num_trajectories, elite_quantile)

    print("Best Hyperparameters:")
    print(f"Number of Iterations: {best_params[0]}")
    print(f"Number of Trajectories: {best_params[1]}")
    print(f"Elite Quantile: {best_params[2]}")
    print(f"Policy Hidden Units: {best_params[3]}")
    print(f"Best Reward: {best_reward}")


def render_episode():
    env = gym.make("LunarLander-v2")

    agent = DeepCrossEntropyAgent(
        env,
        num_iterations=300,
        num_trajectories=100,
        elite_quantile=0.8,
        policy_hidden_units=[200],
        epsilon=1.0,
    )

    agent.policy_net.load_state_dict(
        torch.load("deep_cross_entropy_method/discrete_policy.pth")
    )

    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state


def main():
    env = gym.make("LunarLander-v2")
    agent = DeepCrossEntropyAgent(
        env,
        num_iterations=300,
        num_trajectories=100,
        elite_quantile=0.8,
        policy_hidden_units=[200],
        epsilon=1.0,
    )
    agent.run()
    env.close()

    torch.save(
        agent.policy_net.state_dict(), "deep_cross_entropy_method/discrete_policy.pth"
    )


if __name__ == "__main__":
    main()
    render_episode()
