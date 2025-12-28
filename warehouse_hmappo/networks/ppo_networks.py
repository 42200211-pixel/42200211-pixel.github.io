"""
PPO Network Architectures for HMAPPO.

This module implements the neural network architectures for:
1. Manager Agent (PPO): Actor-Critic networks for global optimization
2. Worker Agents (MAPPO): Shared policy with centralized critic

Architecture Design:
- Actor networks output action distributions (discrete)
- Critic networks output value estimates
- Separate networks for Manager and Worker to avoid interference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional
import numpy as np


def init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """Initialize network weights using orthogonal initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ActorNetwork(nn.Module):
    """
    Actor network for discrete action space.
    
    Outputs a categorical distribution over actions.
    
    Args:
        input_dim: Dimension of input observations
        action_dim: Number of discrete actions
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Output layer for action logits
        self.action_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.action_head, gain=0.01)  # Small initialization for policy head
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action logits.
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            
        Returns:
            Action logits [batch_size, action_dim]
        """
        features = self.feature_net(obs)
        return self.action_head(features)
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and log probability.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return argmax action
            
        Returns:
            Tuple of (action, log_prob)
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions and return log probs and entropy.
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            actions: Action tensor [batch_size]
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation.
    
    For MAPPO: Can take global state for centralized critic.
    
    Args:
        input_dim: Dimension of input (observation or global state)
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Value output
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.value_head, gain=1.0)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning value estimate.
        
        Args:
            obs: Observation/state tensor [batch_size, input_dim]
            
        Returns:
            Value estimate [batch_size, 1]
        """
        features = self.feature_net(obs)
        return self.value_head(features)


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network with shared feature extraction.
    
    More parameter-efficient than separate networks.
    Used for both Manager and Worker agents.
    
    Args:
        input_dim: Dimension of input observations
        action_dim: Number of discrete actions
        hidden_dims: List of hidden layer dimensions for shared network
        actor_hidden_dims: Additional hidden dims for actor head
        critic_hidden_dims: Additional hidden dims for critic head
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128],
        actor_hidden_dims: list = [64],
        critic_hidden_dims: list = [64]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Shared feature extraction
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        shared_out_dim = prev_dim
        
        # Actor head
        actor_layers = []
        prev_dim = shared_out_dim
        
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor_head = nn.Sequential(*actor_layers)
        
        # Critic head
        critic_layers = []
        prev_dim = shared_out_dim
        
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic_head = nn.Sequential(*critic_layers)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        
        # Small initialization for policy output
        for module in self.actor_head:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=0.01)
        
    def forward(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and value.
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (action_logits, value)
        """
        shared_features = self.shared_net(obs)
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        return action_logits, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value in one pass.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return argmax action
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions.
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            actions: Action tensor [batch_size]
            
        Returns:
            Tuple of (log_probs, entropy, values)
        """
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, value.squeeze(-1)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get only value estimate (for GAE computation)."""
        shared_features = self.shared_net(obs)
        return self.critic_head(shared_features).squeeze(-1)


class MAPPOCentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO that takes global state as input.
    
    In MAPPO, the critic uses centralized information during training
    while actors use only local observations during execution.
    
    The global state is a concatenation of all agent observations
    plus any additional global information.
    
    Args:
        num_agents: Number of agents
        local_obs_dim: Dimension of each agent's local observation
        global_state_dim: Additional global state dimension (optional)
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        num_agents: int,
        local_obs_dim: int,
        global_state_dim: int = 0,
        hidden_dims: list = [512, 256, 128]
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim
        
        # Total input dimension
        total_input_dim = num_agents * local_obs_dim + global_state_dim
        
        # Build network
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Output value for each agent
        self.value_head = nn.Linear(prev_dim, num_agents)
        
        # Initialize
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.value_head, gain=1.0)
        
    def forward(
        self,
        all_obs: torch.Tensor,
        global_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            all_obs: All agent observations [batch_size, num_agents, local_obs_dim]
            global_state: Optional global state [batch_size, global_state_dim]
            
        Returns:
            Value estimates for each agent [batch_size, num_agents]
        """
        batch_size = all_obs.shape[0]
        
        # Flatten all observations
        flat_obs = all_obs.view(batch_size, -1)
        
        # Concatenate global state if provided
        if global_state is not None:
            inputs = torch.cat([flat_obs, global_state], dim=-1)
        else:
            inputs = flat_obs
        
        features = self.feature_net(inputs)
        values = self.value_head(features)
        
        return values


# Example usage and testing
if __name__ == "__main__":
    # Test ActorCriticNetwork
    print("Testing ActorCriticNetwork...")
    
    obs_dim = 62  # Example worker observation dim
    action_dim = 5
    batch_size = 32
    
    network = ActorCriticNetwork(obs_dim, action_dim)
    
    # Test forward pass
    obs = torch.randn(batch_size, obs_dim)
    logits, value = network(obs)
    
    print(f"Input shape: {obs.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test get_action_and_value
    action, log_prob, entropy, value = network.get_action_and_value(obs)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    # Test MAPPOCentralizedCritic
    print("\nTesting MAPPOCentralizedCritic...")
    
    num_agents = 4
    local_obs_dim = 62
    
    critic = MAPPOCentralizedCritic(num_agents, local_obs_dim)
    
    all_obs = torch.randn(batch_size, num_agents, local_obs_dim)
    values = critic(all_obs)
    
    print(f"All obs shape: {all_obs.shape}")
    print(f"Values shape: {values.shape}")
    
    print("\nAll tests passed!")
