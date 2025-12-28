"""
RolloutBuffer - Experience buffer for PPO training.

This module provides buffer implementations for storing and processing
experience data during HMAPPO training.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class RolloutData:
    """Single rollout data point."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray


class RolloutBuffer:
    """
    Rollout buffer for collecting trajectories during training.
    
    Stores experience data and computes advantages using GAE.
    
    Attributes:
        buffer_size (int): Maximum number of transitions to store
        obs_dim (int): Observation dimension
        n_agents (int): Number of agents (for multi-agent)
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        n_agents: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum buffer size
            obs_dim: Observation dimension per agent
            n_agents: Number of agents
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Initialize arrays
        self.reset()
        
    def reset(self):
        """Reset buffer to empty state."""
        if self.n_agents > 1:
            self.observations = np.zeros((self.buffer_size, self.n_agents, self.obs_dim), dtype=np.float32)
            self.actions = np.zeros((self.buffer_size, self.n_agents), dtype=np.int64)
            self.rewards = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
            self.dones = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
            self.values = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
            self.log_probs = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
            self.advantages = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
            self.returns = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        else:
            self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
            self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
            self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
            self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
            self.values = np.zeros((self.buffer_size,), dtype=np.float32)
            self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
            self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
            self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False
        
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray
    ):
        """
        Add a transition to the buffer.
        
        Args:
            observation: Observation(s)
            action: Action(s) taken
            reward: Reward(s) received
            done: Done flag(s)
            value: Value estimate(s)
            log_prob: Log probability of action(s)
        """
        assert self.ptr < self.buffer_size, "Buffer is full"
        
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1
        
    def finish_path(self, last_value: np.ndarray):
        """
        Finish a trajectory and compute advantages.
        
        Call this at the end of an episode or when buffer is full.
        
        Args:
            last_value: Value estimate for the final state
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # Append last value for GAE computation
        values_extended = np.concatenate([values, np.expand_dims(last_value, 0)], axis=0)
        
        # Compute GAE
        path_len = self.ptr - self.path_start_idx
        
        if self.n_agents > 1:
            advantages = np.zeros((path_len, self.n_agents), dtype=np.float32)
            
            for agent in range(self.n_agents):
                gae = 0
                for t in reversed(range(path_len)):
                    next_non_terminal = 1.0 - dones[t, agent]
                    delta = (rewards[t, agent] + 
                            self.gamma * values_extended[t + 1, agent] * next_non_terminal - 
                            values[t, agent])
                    gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                    advantages[t, agent] = gae
        else:
            advantages = np.zeros(path_len, dtype=np.float32)
            gae = 0
            
            for t in reversed(range(path_len)):
                next_non_terminal = 1.0 - dones[t]
                delta = (rewards[t] + 
                        self.gamma * values_extended[t + 1] * next_non_terminal - 
                        values[t])
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[t] = gae
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values
        
        self.path_start_idx = self.ptr
        
    def get(self, batch_size: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Generate batches from the buffer.
        
        Args:
            batch_size: Batch size (if None, return all data)
            
        Yields:
            Dictionary containing batch data
        """
        assert self.ptr > 0, "Buffer is empty"
        
        # Normalize advantages
        adv = self.advantages[:self.ptr].copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        if batch_size is None:
            batch_size = self.ptr
        
        indices = np.random.permutation(self.ptr)
        
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_indices = indices[start:end]
            
            yield {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices],
                'values': self.values[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'advantages': adv[batch_indices],
                'returns': self.returns[batch_indices]
            }
    
    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all data from buffer."""
        return {
            'observations': self.observations[:self.ptr].copy(),
            'actions': self.actions[:self.ptr].copy(),
            'rewards': self.rewards[:self.ptr].copy(),
            'dones': self.dones[:self.ptr].copy(),
            'values': self.values[:self.ptr].copy(),
            'log_probs': self.log_probs[:self.ptr].copy(),
            'advantages': self.advantages[:self.ptr].copy(),
            'returns': self.returns[:self.ptr].copy()
        }
    
    @property
    def size(self) -> int:
        """Current buffer size."""
        return self.ptr


class MultiAgentRolloutBuffer:
    """
    Multi-agent rollout buffer for MAPPO training.
    
    Handles experience collection for multiple agents with
    proper global state handling for centralized critic.
    """
    
    def __init__(
        self,
        buffer_size: int,
        n_agents: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Initialize multi-agent buffer.
        
        Args:
            buffer_size: Maximum transitions to store
            n_agents: Number of agents
            obs_dim: Local observation dimension per agent
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()
        
    def reset(self):
        """Reset buffer."""
        # Per-agent data
        self.local_obs = np.zeros((self.buffer_size, self.n_agents, self.obs_dim), dtype=np.float32)
        self.global_state = np.zeros((self.buffer_size, self.n_agents * self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_agents), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        
        # Computed
        self.advantages = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        
        self.ptr = 0
        
    def add(
        self,
        local_obs: np.ndarray,
        global_state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        values: np.ndarray,
        log_probs: np.ndarray
    ):
        """
        Add transition for all agents.
        
        Args:
            local_obs: [n_agents, obs_dim]
            global_state: [n_agents * obs_dim]
            actions: [n_agents]
            rewards: [n_agents]
            done: Single bool
            values: [n_agents]
            log_probs: [n_agents]
        """
        if self.ptr >= self.buffer_size:
            return
        
        self.local_obs[self.ptr] = local_obs
        self.global_state[self.ptr] = global_state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        
        self.ptr += 1
        
    def compute_advantages(self, last_values: np.ndarray):
        """
        Compute advantages using GAE.
        
        Args:
            last_values: [n_agents] value estimates for final state
        """
        for agent in range(self.n_agents):
            gae = 0
            
            for t in reversed(range(self.ptr)):
                if t == self.ptr - 1:
                    next_value = last_values[agent]
                else:
                    next_value = self.values[t + 1, agent]
                
                next_non_terminal = 1.0 - self.dones[t]
                delta = (self.rewards[t, agent] + 
                        self.gamma * next_value * next_non_terminal - 
                        self.values[t, agent])
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                self.advantages[t, agent] = gae
        
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
        
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Generator[Dict, None, None]:
        """
        Generate batches for training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle indices
            
        Yields:
            Dictionary with batch data
        """
        # Normalize advantages
        adv = self.advantages[:self.ptr].copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Create indices for all agent-timestep pairs
        total_samples = self.ptr * self.n_agents
        indices = np.arange(total_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            # Convert flat indices to (timestep, agent) pairs
            timesteps = batch_indices // self.n_agents
            agents = batch_indices % self.n_agents
            
            yield {
                'local_obs': self.local_obs[timesteps, agents],
                'global_state': self.global_state[timesteps],
                'actions': self.actions[timesteps, agents],
                'values': self.values[timesteps, agents],
                'log_probs': self.log_probs[timesteps, agents],
                'advantages': adv[timesteps, agents],
                'returns': self.returns[timesteps, agents]
            }
    
    @property
    def size(self) -> int:
        """Current buffer size."""
        return self.ptr


# Example usage
if __name__ == "__main__":
    # Test single-agent buffer
    print("Testing single-agent RolloutBuffer...")
    buffer = RolloutBuffer(buffer_size=100, obs_dim=10, n_agents=1)
    
    # Add some data
    for _ in range(20):
        buffer.add(
            observation=np.random.randn(10),
            action=np.random.randint(0, 5),
            reward=np.random.randn(),
            done=np.array(0.0),
            value=np.random.randn(),
            log_prob=np.random.randn()
        )
    
    buffer.finish_path(last_value=np.array(0.0))
    
    print(f"Buffer size: {buffer.size}")
    
    for batch in buffer.get(batch_size=8):
        print(f"Batch observations shape: {batch['observations'].shape}")
        break
    
    # Test multi-agent buffer
    print("\nTesting MultiAgentRolloutBuffer...")
    ma_buffer = MultiAgentRolloutBuffer(buffer_size=100, n_agents=4, obs_dim=10)
    
    for _ in range(20):
        ma_buffer.add(
            local_obs=np.random.randn(4, 10),
            global_state=np.random.randn(40),
            actions=np.random.randint(0, 5, size=4),
            rewards=np.random.randn(4),
            done=False,
            values=np.random.randn(4),
            log_probs=np.random.randn(4)
        )
    
    ma_buffer.compute_advantages(last_values=np.zeros(4))
    
    print(f"Buffer size: {ma_buffer.size}")
    
    for batch in ma_buffer.get_batches(batch_size=32):
        print(f"Batch local_obs shape: {batch['local_obs'].shape}")
        break
    
    print("\nAll tests passed!")
