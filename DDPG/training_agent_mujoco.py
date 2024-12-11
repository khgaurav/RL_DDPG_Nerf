from time import sleep
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torchrl.data import ReplayBuffer, LazyMemmapStorage, PrioritizedReplayBuffer
import argparse
import gymnasium as gym
import fancy_gym
import copy
import matplotlib.pyplot as plt
import datetime
import tqdm
from tensordict import TensorDict
import os
from gymnasium.wrappers import RecordVideo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, *, hidden_dims=[256, 400]):
        """Actor Network for DDPG.

        Args:
            state_dim: Dimensionality of states
            action_dim: Dimensionality of actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = [
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        ]
        
        # Add hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU()
            ])
            
        # Output layer with Tanh activation for bounded actions
        layers.extend([
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()
        ])
        
        self.net = nn.Sequential(*layers)
        # Initialize weights
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Maps states to deterministic actions.

        Args:
            state: (batch_size, state_dim) tensor
        Returns:
            actions: (batch_size, action_dim) tensor with values in [-1, 1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state.float())

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, *, hidden_dims=[256, 400]):
        """Critic Network for DDPG.
        
        Args:
            state_dim: Dimensionality of states
            action_dim: Dimensionality of actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # First layer processes state input
        self.input_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # Second layer combines first hidden layer output with action
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1]))
        
        # Add remaining hidden layers
        for i in range(1, len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, input):
        """Forward pass of the critic network.
        
        Args:
            input: Tuple of (states, actions)
        Returns:
            Q-values: (batch_size, 1) tensor
        """
        state, action = input
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            
        x = self.input_net(state.float())
        x = torch.cat([x, action.float()], dim=1)
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            
        return self.output_layer(x)
    

class OrnsteinUhlenbeckProcess(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.buffer = []
#         self.priorities = np.zeros((capacity,), dtype=np.float32)
#         self.position = 0

#     def add(self, transition):
#         max_priority = self.priorities.max() if self.buffer else 1.0
        
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(transition)
#         else:
#             self.buffer[self.position] = transition
        
#         self.priorities[self.position] = max_priority
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size, beta=0.4):
#         if len(self.buffer) == self.capacity:
#             prios = self.priorities
#         else:
#             prios = self.priorities[:self.position]

#         # Calculate sampling probabilities
#         probs = prios ** self.alpha
#         probs /= probs.sum()

#         # Sample indices and calculate importance weights
#         indices = np.random.choice(len(self.buffer), batch_size, p=probs)
#         samples = [self.buffer[idx] for idx in indices]
        
#         total = len(self.buffer)
#         weights = (total * probs[indices]) ** (-beta)
#         weights /= weights.max()
        
#         batch = list(zip(*samples))
#         return batch, indices, torch.FloatTensor(weights).to(device)

#     def update_priorities(self, indices, priorities):
#         for idx, priority in zip(indices, priorities):
#             self.priorities[idx] = priority

#     def __len__(self):
#         return len(self.buffer)


class DDPG(object):
    def __init__(self, state_dim, action_dim, params, hidden_dims=[256, 400]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, hidden_dims=hidden_dims).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(device)
        
        # Initialize target networks with the same weights
        self.target_actor = Actor(state_dim, action_dim, hidden_dims=hidden_dims).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=params['actor_lr'])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=params['critic_lr'])
        
        # Hyperparameters
        self.gamma = params.get('gamma', 0.99)
        self.tau = params.get('tau', 0.005)
        self.epsilon = params.get('epsilon', 0.1)
        self.batch_size = params.get('batch_size', 64)
        
        # Random process for action exploration
        self.exploration = OrnsteinUhlenbeckProcess(action_dim)
        
        # Replay buffer
        # self.buffer = ReplayBuffer(storage=LazyMemmapStorage(params['buffer_size']))
        self.buffer = PrioritizedReplayBuffer(
            storage=LazyMemmapStorage(params['buffer_size']),
            alpha=0.6,  # Priority exponent (typically between 0.0 and 1.0)
            beta=0.4,   # Initial importance sampling weight (typically starts at 0.4 and annealed to 1.0)
        )
        
        # Training mode
        self.training = True
        
        # Observations
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
        self.timesteps_episode = []
        self.q_values = []
        
    def get_action(self, state):
        """Select an action from the actor network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action = self.actor(state_tensor)
            action = action.cpu().numpy()
            if self.training:
                noise = self.epsilon * self.exploration.sample()
                action = action + noise
            return np.clip(action, -1, 1)
        
        
    def run_train(self):
        # Sample a batch from the replay buffer
        batch, info = self.buffer.sample(self.batch_size, return_info=True)
        # states = batch[0].float().squeeze(1).to(device)  # Remove extra dimension
        # actions = batch[1].float().squeeze(1).to(device)  # Remove extra dimension
        # rewards = batch[2].float().to(device)
        # next_states = batch[3].float().squeeze(1).to(device)  # Remove extra dimension
        # dones = batch[4].to(device)
        states = batch.get("state").float().squeeze(1).to(device)
        actions = batch.get("action").float().squeeze(1).to(device)
        rewards = batch.get("reward").float().to(device)
        next_states = batch.get("next_state").float().squeeze(1).to(device)
        dones = batch.get("terminated").to(device)
        indices = info.get("index").to(device)
        weights = info.get("_weight").to(device)
        # batch, indices, weights = self.buffer.sample(self.batch_size)
        # states, actions, rewards, next_states, dones = batch
        # Current q values from critic network
        current_q_values = self.critic((states, actions))

        # Next q values from target networks
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic((next_states, next_actions))
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (~dones).unsqueeze(1)
        
        # Prioritized Experience Replay
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        # self.buffer.update_priorities(indices, td_errors + 1e-6)
        
        # Compute Critic loss
        # critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        critic_loss = (weights * F.mse_loss(current_q_values, target_q_values.detach(), reduction='none')).mean()
        # Store Q values for monitoring
        self.q_values.append(current_q_values.mean().item())
        

        # Update Critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute Actor loss
        # policy_loss = -self.critic((states, self.actor(states))).mean()
        policy_loss = -(weights * self.critic((states, self.actor(states)))).mean()
        # Update Actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update priorities in the buffer
        self.buffer.update_priority(indices, torch.tensor(td_errors.squeeze(), device=device))
        # Update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        self.actor_losses.append(policy_loss.item())
        self.critic_losses.append(critic_loss.item())
    
    def learn(self, total_timesteps=10000, callback=None):
        """Train the agent for a given number of timesteps."""
        self.training = True
        previous_timestep = 0
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        episode = 0
        total_reward = 0
        pbar = tqdm.tqdm(range(total_timesteps), desc="Time Steps", unit="steps")
        for t in pbar:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action.flatten())
            # Concatenate next observation dictionary
            
            # Store transition in replay buffer
            # self.buffer.add((
            #     torch.tensor(state).to(device),
            #     torch.tensor(action).to(device),
            #     torch.tensor(reward).to(device),
            #     torch.tensor(next_state).to(device),
            #     torch.tensor(terminated).to(device)
            # ))
            self.buffer.extend(TensorDict({
                "state": torch.tensor(state, device=device).unsqueeze(0),
                "action": torch.tensor(action, device=device).unsqueeze(0),
                "reward": torch.tensor(reward, device=device).unsqueeze(0),
                "next_state": torch.tensor(next_state, device=device).unsqueeze(0),
                "terminated": torch.tensor(terminated, device=device).unsqueeze(0),
                "priority": torch.ones(1, device=device)  # Initial priority set to 1
            }, batch_size=[1]))
            
            state = next_state
            total_reward += self.gamma * float(reward)
            
            if len(self.buffer) > self.batch_size:
                self.run_train()
            
            if terminated or truncated:
                self.episode_rewards.append(total_reward)
                self.timesteps_episode.append(t - previous_timestep)
                pbar.set_description_str(f"Episode {episode} - Total Reward: {total_reward:.2f}")
                # print(f"Episode {episode} - Total Reward: {total_reward:.2f} - Timesteps: {t}")
                state, _ = env.reset()
                total_reward = 0
                previous_timestep = t
                episode += 1
                self.exploration.reset()
    
    def save(self, path):
        """Save the model to a file."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
        
    def load(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
    
    def test(self, total_episodes=10):
        """Test the agent in the environment."""
        self.training = False
        for episode in range(total_episodes):
            state, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            
            while not terminated and not truncated:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action.flatten())
                state = next_state
                total_reward += float(reward)
                print(f"Reward: {reward}")
                sleep(0.1)
                
            print(f"Episode {episode} - Total Reward: {total_reward:.2f}")
    
def plot_all_curves(curves, titles, xlabel, ylabel, args, params):
    """
    Args:
        curves (list): list of tuples, each containing (arr_list, legend_list, color_list)
        titles (list): list of titles for each subplot
        xlabel (string): label of the X axis
        ylabel (string): label of the Y axis
        fig_title (string): title of the entire figure
    """
    num_plots = len(curves)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12 * num_cols, 8 * num_rows))
    fig.suptitle(f"{params}", fontsize=8)
    
    for i, (arr_list, legend_list, color_list) in enumerate(curves):
        ax = axes[i // num_cols, i % num_cols] if num_plots > 1 else axes
        ax.set_ylabel(ylabel[i], fontsize=8)
        ax.set_xlabel(xlabel[i], fontsize=8)
        ax.set_title(titles[i], fontsize=8)
        
        h_list = []
        for arr, legend, color in zip(arr_list, legend_list, color_list):
            arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
            h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
            arr_err *= 1.96
            ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3, color=color)
            h_list.append(h)
        
        ax.legend(handles=h_list)
    
    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])
    plt.subplots_adjust(hspace=0.25)
    if args.save:
        # Save the plot to a file
        if not os.path.exists(args.env):
            os.makedirs(args.env)
        if args.param_optimize:
            fig.savefig(os.path.join(args.env, f"optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        else:
            fig.savefig(os.path.join(args.env, f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    else:
        plt.show()

def match_len_of_arrays(arrays):
    # Find minimum length
    min_length = min(len(arr) for arr in arrays)

    # Create new array with consistent dimensions
    uniform_array = np.array([arr[:min_length] for arr in arrays])
    return uniform_array
       
class WhaleOptimizer:
    def __init__(self, num_whales=30, max_iter=50, timesteps=5000, early_stopping_rounds=10):
        self.num_whales = num_whales
        self.max_iter = max_iter
        self.timesteps = timesteps
        self.early_stopping_rounds = early_stopping_rounds
        self.bounds = {
            'actor_lr': (0.0001, 0.001),      # Actor learning rate[1][3]
            'critic_lr': (0.001, 0.01),       # Critic learning rate[1][3]
            'gamma': (0.95, 0.99),            # Discount factor[1]
            'tau': (0.001, 0.01),             # Target network update rate[2]
            'batch_size': (64, 256),          # Mini-batch size[1][2]
            'buffer_size': (1e5, 1e6),        # Replay buffer size[2]
            'epsilon': (0.1, 0.3)             # Exploration noise[2]
        }

        self.episodes_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        self.timesteps_episodes = []

    def initialize_population(self):
        population = []
        for _ in range(self.num_whales):
            whale = {
                'actor_lr': np.random.uniform(self.bounds['actor_lr'][0], self.bounds['actor_lr'][1]),
                'critic_lr': np.random.uniform(self.bounds['critic_lr'][0], self.bounds['critic_lr'][1]),
                'gamma': np.random.uniform(self.bounds['gamma'][0], self.bounds['gamma'][1]),
                'tau': np.random.uniform(self.bounds['tau'][0], self.bounds['tau'][1]),
                'epsilon': np.random.uniform(self.bounds['epsilon'][0], self.bounds['epsilon'][1]),
                'batch_size': int(np.random.uniform(self.bounds['batch_size'][0], self.bounds['batch_size'][1])),
                'buffer_size': int(np.random.uniform(self.bounds['buffer_size'][0], self.bounds['buffer_size'][1])),
            }
            population.append(whale)
        return population

    def evaluate_fitness(self, params):
        print(f"Evaluating Parameters: {params}")
        agent = DDPG(state_dim, action_dim, params)
        agent.learn(total_timesteps=self.timesteps)
        self.episodes_rewards.append(np.array(agent.episode_rewards))
        self.actor_losses.append(np.array(agent.actor_losses))
        self.critic_losses.append(np.array(agent.critic_losses))
        self.q_values.append(np.array(agent.q_values))
        self.timesteps_episodes.append(np.array(agent.timesteps_episode))
        return np.mean(agent.episode_rewards[-10:])

    def optimize(self, args):
        population = self.initialize_population()
        best_whale = None
        best_fitness = float('-inf')
        no_improvement_rounds = 0
        
        for iteration in range(self.max_iter):
            for i in range(self.num_whales):
                fitness = self.evaluate_fitness(population[i])
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_whale = population[i].copy()
                    no_improvement_rounds = 0
                else:
                    no_improvement_rounds += 1
                
                if no_improvement_rounds >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {iteration} due to no improvement for {self.early_stopping_rounds} rounds.")
                    return best_whale
                
                a = 2 - iteration * (2/self.max_iter)
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * r
                
                if np.random.random() < 0.5:
                    if abs(A) < 1:
                        for param in population[i]:
                            D = abs(C * best_whale[param] - population[i][param])
                            population[i][param] = best_whale[param] - A * D
                    else:
                        random_whale = population[np.random.randint(0, self.num_whales)]
                        for param in population[i]:
                            D = abs(C * random_whale[param] - population[i][param])
                            population[i][param] = random_whale[param] - A * D
                else:
                    for param in population[i]:
                        D = abs(best_whale[param] - population[i][param])
                        l = np.random.uniform(-1, 1)
                        population[i][param] = D * np.exp(l) * np.cos(2 * np.pi * l) + best_whale[param]
                
                for param in population[i]:
                    if param in self.bounds:
                        population[i][param] = np.clip(
                            population[i][param], 
                            self.bounds[param][0], 
                            self.bounds[param][1]
                        )
                        if param == 'batch_size':
                            population[i][param] = int(population[i][param])
            
                print(f"Iteration {iteration}, Best Fitness: {best_fitness}")
            if args.save:
                if not os.path.exists(args.env):
                    os.makedirs(args.env)
                with open(f"{args.env}/params.txt", "a") as f:
                    f.write(f"Iteration {iteration}:\n")
                    for key, value in best_whale.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                curves = [([match_len_of_arrays(self.episodes_rewards)], ["Episode Rewards"], ["blue"]),
                    ([match_len_of_arrays(self.actor_losses)], ["Actor Losses"], ["red"]),
                    ([match_len_of_arrays(self.critic_losses)], ["Critic Losses"], ["blue"]),
                    ([match_len_of_arrays(self.q_values)], ["Q-Values"], ["green"]),
                    ([match_len_of_arrays(self.timesteps_episodes)], ["Timesteps per Episode"], ["purple"])
                ]
                titles = ["Episode Rewards", "Actor Losses", "Critic Losses", "Q-Values", "Timesteps"]
                plot_all_curves(curves, titles, ["Episodes", "Timestep", "Timestep", "Timestep", "Episodes"], ["Rewards", "Loss", "Loss", "Q-Value", "Timesteps"], args, params)
                    
                print(f"Iteration {iteration}, Best Whale: {best_whale}")
        
        return best_whale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG train/test')
    parser.add_argument("--test", default=False)
    parser.add_argument("--env", default="metaworld/drawer-open-v2")
    parser.add_argument("--runs", default=1)
    parser.add_argument("--timesteps", default=10000)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--param_optimize", action="store_true")
    args = parser.parse_args()
    params = {
        'actor_lr': 0.0007630819606555925,
        'critic_lr': 0.007630819606555926,
        'gamma': 0.95,
        'tau': 0.007630819606555926,
        'epsilon': 0.22892458819667777,
        'batch_size': 195,
        'buffer_size': 306119.3288955592,
        'hidden_dims': [256, 400]
    }

    if args.test:
        # Create Isaac Lab environment
        env = gym.make(args.env, render_mode='human')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] # type: ignore
        print(f"State Dimension: {state_dim}, Action Dimension: {action_dim}")
        agent = DDPG(state_dim, action_dim, params)
        agent.load(args.test)
        
        if not os.path.exists(args.env):
            os.makedirs(args.env)
        
        # Save video
        # env = RecordVideo(env, video_folder=args.env, name_prefix=f"{args.env}-{args.test}")
        
        agent.test(total_episodes=10)
    else: 
        # Create Isaac Lab environment
        env = gym.make(args.env)
        # Set seed for reproducibility
        seed = np.random.randint(0, 1000)
        print(f"Seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] # type: ignore
        print(f"State Dimension: {state_dim}, Action Dimension: {action_dim}")
        # Initialize WOA
        # Find optimal parameters
        if args.param_optimize:
            optimizer = WhaleOptimizer(num_whales=30, max_iter=30, timesteps=5000, early_stopping_rounds=1000)
            params = optimizer.optimize(args)
        print(f"Optimal Parameters: {params}")
        episodes_rewards = []
        actor_losses = []
        critic_losses = []
        q_values = []
        timesteps_episodes = []
        for i in range(int(args.runs)):
            agent = DDPG(state_dim, action_dim, params)
            agent.learn(total_timesteps=int(args.timesteps))
            if(args.save):
                if not os.path.exists(args.env):
                    os.makedirs(args.env)
                agent.save(os.path.join(args.env, f"run_{i}_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"))
            episodes_rewards.append(np.array(agent.episode_rewards))
            actor_losses.append(np.array(agent.actor_losses))
            critic_losses.append(np.array(agent.critic_losses))
            q_values.append(np.array(agent.q_values))
            timesteps_episodes.append(np.array(agent.timesteps_episode))
        
        curves = [([match_len_of_arrays(episodes_rewards)], ["Episode Rewards"], ["blue"]),
                ([match_len_of_arrays(actor_losses)], ["Actor Losses"], ["red"]),
                ([match_len_of_arrays(critic_losses)], ["Critic Losses"], ["blue"]),
                ([match_len_of_arrays(q_values)], ["Q-Values"], ["green"]),
                ([match_len_of_arrays(timesteps_episodes)], ["Timesteps per Episode"], ["purple"])
            ]
        args.param_optimize = False
        titles = ["Episode Rewards", "Actor Losses", "Critic Losses", "Q-Values", "Timesteps"]
        plot_all_curves(curves, titles, ["Episodes", "Timestep", "Timestep", "Timestep", "Episodes"], ["Rewards", "Loss", "Loss", "Q-Value", "Timesteps"], args, params)
