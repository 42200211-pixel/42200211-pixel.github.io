"""
ManagerAgent - PPO-based Manager for global optimization.

The Manager agent operates at Level 1 of the HMAPPO hierarchy:
- Observes compressed global state (NOT full grid)
- Decides on strategic parameters (Z, traffic flow policies)
- Receives episodic reward based on throughput and collision metrics
- Does NOT control individual robot movements

Key Design Principles:
1. Manager uses compressed state vectors, not full grid observation
2. Manager creates constraints/policies, not step-by-step control
3. Manager reward is based on episode-level metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..networks.ppo_networks import ActorCriticNetwork


@dataclass
class ManagerExperience:
    """Single experience for Manager agent."""
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    log_prob: float
    value: float


class ManagerAgent:
    """
    PPO-based Manager Agent for global optimization.
    
    The Manager operates at episodic level, making decisions about:
    1. Aisle width Z (discrete: keep, increase, decrease)
    2. (Optional) Traffic flow policies
    
    Observation Space (7-dimensional compressed vector):
        - avg_travel_time: Normalized average delivery time
        - robot_density_mean: Mean robot density across regions
        - robot_density_std: Std of robot density
        - collision_rate: Collisions per timestep
        - deadlock_ratio: Deadlocks per timestep
        - throughput: Deliveries per timestep
        - current_Z: Normalized aisle width
    
    Action Space (discrete, 3 actions):
        0: Keep current Z
        1: Increase Z by 1
        2: Decrease Z by 1 (minimum 2)
    
    Attributes:
        obs_dim (int): Observation dimension (7)
        action_dim (int): Action dimension (3)
        network (ActorCriticNetwork): PPO actor-critic network
        optimizer (torch.optim.Optimizer): Network optimizer
    """
    
    def __init__(
        self,
        obs_dim: int = 7,
        action_dim: int = 3,
        hidden_dims: list = [128, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize Manager agent.
        
        Args:
            obs_dim: Observation dimension (compressed state)
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions for network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            device: Device for computation
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize network
        self.network = ActorCriticNetwork(
            input_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Experience buffer
        self.experience_buffer: List[ManagerExperience] = []
        
        # Training state
        self.is_frozen = False
        self.total_updates = 0
        
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given observation.
        
        Args:
            observation: Compressed state observation (7-dim)
            deterministic: If True, return argmax action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            action, log_prob, _, value = self.network.get_action_and_value(
                obs_tensor, deterministic=deterministic
            )
            
            return (
                int(action.item()),
                float(log_prob.item()),
                float(value.item())
            )
    
    def store_experience(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store experience in buffer."""
        exp = ManagerExperience(
            observation=observation.copy(),
            action=action,
            reward=reward,
            next_observation=next_observation.copy(),
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.experience_buffer.append(exp)
        
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            last_value: Value estimate for final state
            
        Returns:
            Tuple of (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self,
        n_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            n_epochs: Number of optimization epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of training metrics
        """
        if self.is_frozen or len(self.experience_buffer) == 0:
            return {}
        
        # Prepare data
        observations = np.array([e.observation for e in self.experience_buffer])
        actions = np.array([e.action for e in self.experience_buffer])
        rewards = np.array([e.reward for e in self.experience_buffer])
        dones = np.array([e.done for e in self.experience_buffer])
        old_log_probs = np.array([e.log_prob for e in self.experience_buffer])
        old_values = np.array([e.value for e in self.experience_buffer])
        
        # Get last value for GAE
        with torch.no_grad():
            last_obs = self.experience_buffer[-1].next_observation
            last_obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            last_value = self.network.get_value(last_obs_tensor).item()
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, old_values, dones, last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        observations_t = torch.FloatTensor(observations).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # PPO update
        n_samples = len(observations)
        indices = np.arange(n_samples)
        
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Get batch
                obs_batch = observations_t[batch_indices]
                act_batch = actions_t[batch_indices]
                old_log_probs_batch = old_log_probs_t[batch_indices]
                adv_batch = advantages_t[batch_indices]
                ret_batch = returns_t[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values = self.network.evaluate_actions(obs_batch, act_batch)
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, ret_batch)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear buffer
        self.experience_buffer = []
        self.total_updates += n_updates
        
        return {
            "manager/policy_loss": total_policy_loss / n_updates,
            "manager/value_loss": total_value_loss / n_updates,
            "manager/entropy": total_entropy / n_updates,
            "manager/updates": n_updates
        }
    
    def freeze(self):
        """Freeze the network (no updates)."""
        self.is_frozen = True
        for param in self.network.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze the network (allow updates)."""
        self.is_frozen = False
        for param in self.network.parameters():
            param.requires_grad = True
    
    def set_learning_rate(self, lr: float):
        """Set new learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']
        
    def get_stats(self) -> Dict[str, any]:
        """Get agent statistics."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "total_updates": self.total_updates,
            "is_frozen": self.is_frozen,
            "buffer_size": len(self.experience_buffer)
        }


# Example usage
if __name__ == "__main__":
    # Create Manager agent
    manager = ManagerAgent(
        obs_dim=7,
        action_dim=3,
        hidden_dims=[128, 64],
        lr=3e-4
    )
    
    print("Manager Agent created:")
    print(f"Observation dim: {manager.obs_dim}")
    print(f"Action dim: {manager.action_dim}")
    
    # Test action selection
    obs = np.random.randn(7).astype(np.float32)
    action, log_prob, value = manager.get_action(obs)
    
    print(f"\nTest action selection:")
    print(f"Observation: {obs[:3]}...")
    print(f"Action: {action}")
    print(f"Log prob: {log_prob:.4f}")
    print(f"Value: {value:.4f}")
    
    # Store some experiences
    for i in range(10):
        next_obs = np.random.randn(7).astype(np.float32)
        manager.store_experience(
            observation=obs,
            action=np.random.randint(0, 3),
            reward=np.random.randn(),
            next_observation=next_obs,
            done=(i == 9),
            log_prob=np.random.randn(),
            value=np.random.randn()
        )
        obs = next_obs
    
    # Test update
    metrics = manager.update(n_epochs=2, batch_size=4)
    print(f"\nTraining metrics: {metrics}")
