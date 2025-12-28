"""
WorkerAgent - MAPPO-based Worker for individual robot control.

The Worker agents operate at Level 2 of the HMAPPO hierarchy:
- Each robot is an agent sharing the same policy
- Observe local grid around the robot
- Execute step-by-step navigation actions
- Centralized training (global state for critic), decentralized execution

Key Design Principles:
1. Shared policy across all workers (parameter sharing)
2. Local observations for execution (decentralized)
3. Centralized critic using global state during training
4. Reward shaping for navigation, delivery, collision avoidance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..networks.ppo_networks import ActorCriticNetwork, MAPPOCentralizedCritic


@dataclass
class WorkerExperience:
    """Single experience for a worker agent."""
    local_obs: np.ndarray  # Local observation for this agent
    global_state: np.ndarray  # Global state (all agents' obs concatenated)
    action: int
    reward: float
    next_local_obs: np.ndarray
    next_global_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    agent_id: int


class WorkerAgent:
    """
    MAPPO-based Worker Agent for individual robot control.
    
    Implements Multi-Agent PPO with:
    - Shared policy: All robots share the same actor network
    - Centralized critic: Uses global state during training
    - Decentralized execution: Each robot acts based on local observation
    
    Observation Space (per robot):
        - Local grid (local_obs_size x local_obs_size): Flattened
        - Direction to target (dx, dy): Normalized
        - Distance to target: Normalized
        - Current Z: Normalized (signal from Manager)
        - Robot state: Encoded
        - Nearby robot positions: Up to 4 neighbors
        
    Action Space (discrete, 5 actions):
        0: UP
        1: DOWN
        2: LEFT
        3: RIGHT
        4: WAIT
    
    Attributes:
        num_agents (int): Number of robot agents
        obs_dim (int): Local observation dimension
        action_dim (int): Action dimension (5)
        actor (ActorCriticNetwork): Shared actor-critic network
        centralized_critic (MAPPOCentralizedCritic): Centralized critic
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int = 5,
        actor_hidden_dims: list = [256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        lr_actor: float = 3e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize Worker agent.
        
        Args:
            num_agents: Number of robot agents
            obs_dim: Local observation dimension per agent
            action_dim: Number of discrete actions (5)
            actor_hidden_dims: Hidden dims for actor network
            critic_hidden_dims: Hidden dims for centralized critic
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm
            device: Computation device
        """
        self.num_agents = num_agents
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
        
        # Shared Actor-Critic for decentralized execution
        # Each robot uses this network with its local observation
        self.actor = ActorCriticNetwork(
            input_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims
        ).to(self.device)
        
        # Centralized Critic for training
        # Takes global state (all agents' observations) as input
        self.centralized_critic = MAPPOCentralizedCritic(
            num_agents=num_agents,
            local_obs_dim=obs_dim,
            global_state_dim=0,  # No additional global state
            hidden_dims=critic_hidden_dims
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.centralized_critic.parameters(), lr=lr_critic)
        
        # Experience buffer (stores experiences for all agents)
        self.experience_buffer: List[WorkerExperience] = []
        
        # Training state
        self.is_frozen = False
        self.total_updates = 0
        
    def get_actions(
        self,
        observations: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get actions for all robots given their local observations.
        
        This is used during decentralized execution.
        
        Args:
            observations: All local observations [num_agents, obs_dim]
            deterministic: If True, return argmax actions
            
        Returns:
            Tuple of (actions, log_probs, values_local)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            
            # Get actions from shared actor
            actions, log_probs, _, values = self.actor.get_action_and_value(
                obs_tensor, deterministic=deterministic
            )
            
            return (
                actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.cpu().numpy()
            )
    
    def get_centralized_values(
        self,
        all_observations: np.ndarray
    ) -> np.ndarray:
        """
        Get value estimates using centralized critic.
        
        Args:
            all_observations: All agents' observations [batch, num_agents, obs_dim]
            
        Returns:
            Value estimates [batch, num_agents]
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(all_observations).to(self.device)
            
            if obs_tensor.dim() == 2:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            values = self.centralized_critic(obs_tensor)
            
            return values.cpu().numpy()
    
    def store_experience(
        self,
        local_obs: np.ndarray,
        global_state: np.ndarray,
        action: int,
        reward: float,
        next_local_obs: np.ndarray,
        next_global_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
        agent_id: int
    ):
        """Store experience for a single agent."""
        exp = WorkerExperience(
            local_obs=local_obs.copy(),
            global_state=global_state.copy(),
            action=action,
            reward=reward,
            next_local_obs=next_local_obs.copy(),
            next_global_state=next_global_state.copy(),
            done=done,
            log_prob=log_prob,
            value=value,
            agent_id=agent_id
        )
        self.experience_buffer.append(exp)
        
    def store_batch_experience(
        self,
        local_obs: np.ndarray,
        global_state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_local_obs: np.ndarray,
        next_global_state: np.ndarray,
        done: bool,
        log_probs: np.ndarray,
        values: np.ndarray
    ):
        """
        Store experiences for all agents at once.
        
        Args:
            local_obs: [num_agents, obs_dim]
            global_state: [num_agents, obs_dim] (all obs)
            actions: [num_agents]
            rewards: [num_agents]
            next_local_obs: [num_agents, obs_dim]
            next_global_state: [num_agents, obs_dim]
            done: Single bool (episode done)
            log_probs: [num_agents]
            values: [num_agents]
        """
        for i in range(self.num_agents):
            self.store_experience(
                local_obs=local_obs[i],
                global_state=global_state.flatten(),
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_local_obs=next_local_obs[i],
                next_global_state=next_global_state.flatten(),
                done=done,
                log_prob=float(log_probs[i]),
                value=float(values[i]),
                agent_id=i
            )
    
    def compute_gae_per_agent(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE for experiences from a single agent."""
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
        
        returns = advantages + np.array(values)
        return advantages, returns
    
    def update(
        self,
        n_epochs: int = 4,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Update actor and centralized critic using PPO.
        
        Args:
            n_epochs: Number of optimization epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of training metrics
        """
        if self.is_frozen or len(self.experience_buffer) == 0:
            return {}
        
        # Organize experiences by agent
        agent_experiences: Dict[int, List[WorkerExperience]] = {
            i: [] for i in range(self.num_agents)
        }
        
        for exp in self.experience_buffer:
            agent_experiences[exp.agent_id].append(exp)
        
        # Compute advantages for each agent
        all_data = []
        
        for agent_id in range(self.num_agents):
            exps = agent_experiences[agent_id]
            if not exps:
                continue
            
            rewards = [e.reward for e in exps]
            values = [e.value for e in exps]
            dones = [e.done for e in exps]
            
            # Get last value for this agent
            last_exp = exps[-1]
            with torch.no_grad():
                last_obs = torch.FloatTensor(last_exp.next_local_obs).unsqueeze(0).to(self.device)
                last_value = self.actor.get_value(last_obs).item()
            
            advantages, returns = self.compute_gae_per_agent(rewards, values, dones, last_value)
            
            for i, exp in enumerate(exps):
                all_data.append({
                    'local_obs': exp.local_obs,
                    'global_state': exp.global_state,
                    'action': exp.action,
                    'old_log_prob': exp.log_prob,
                    'advantage': advantages[i],
                    'return': returns[i],
                    'agent_id': agent_id
                })
        
        if not all_data:
            return {}
        
        # Convert to arrays
        local_obs = np.array([d['local_obs'] for d in all_data])
        global_state = np.array([d['global_state'] for d in all_data])
        actions = np.array([d['action'] for d in all_data])
        old_log_probs = np.array([d['old_log_prob'] for d in all_data])
        advantages = np.array([d['advantage'] for d in all_data])
        returns = np.array([d['return'] for d in all_data])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        local_obs_t = torch.FloatTensor(local_obs).to(self.device)
        global_state_t = torch.FloatTensor(global_state).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Note: For centralized critic training, we would use global_state_t
        # Here we use the actor's value head which takes local observations
        # The centralized critic is reserved for full MAPPO implementation
        # global_state is stored but not used in this simplified version
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        n_samples = len(all_data)
        indices = np.arange(n_samples)
        
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Get batch
                local_obs_batch = local_obs_t[batch_indices]
                actions_batch = actions_t[batch_indices]
                old_log_probs_batch = old_log_probs_t[batch_indices]
                adv_batch = advantages_t[batch_indices]
                ret_batch = returns_t[batch_indices]
                
                # Get global state for critic (simplified: use local obs for now)
                # In full implementation, would properly batch global states
                
                # Actor update
                log_probs, entropy, values = self.actor.evaluate_actions(
                    local_obs_batch, actions_batch
                )
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (using actor's value head for simplicity)
                value_loss = nn.functional.mse_loss(values, ret_batch)
                
                # Actor loss
                actor_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear buffer
        self.experience_buffer = []
        self.total_updates += n_updates
        
        return {
            "worker/policy_loss": total_policy_loss / n_updates,
            "worker/value_loss": total_value_loss / n_updates,
            "worker/entropy": total_entropy / n_updates,
            "worker/updates": n_updates
        }
    
    def freeze(self):
        """Freeze networks (no updates)."""
        self.is_frozen = True
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.centralized_critic.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze networks."""
        self.is_frozen = False
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.centralized_critic.parameters():
            param.requires_grad = True
    
    def set_learning_rate(self, lr_actor: float, lr_critic: Optional[float] = None):
        """Set new learning rates."""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr_actor
        
        if lr_critic is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr_critic
                
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.centralized_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_updates': self.total_updates,
            'num_agents': self.num_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.centralized_critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']
        
    def get_stats(self) -> Dict[str, any]:
        """Get agent statistics."""
        return {
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "total_updates": self.total_updates,
            "is_frozen": self.is_frozen,
            "buffer_size": len(self.experience_buffer)
        }


# Example usage
if __name__ == "__main__":
    # Create Worker agent
    num_agents = 4
    obs_dim = 62  # Example: 7x7 grid + additional features
    
    worker = WorkerAgent(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=5
    )
    
    print("Worker Agent created:")
    print(f"Number of agents: {worker.num_agents}")
    print(f"Observation dim: {worker.obs_dim}")
    print(f"Action dim: {worker.action_dim}")
    
    # Test action selection
    observations = np.random.randn(num_agents, obs_dim).astype(np.float32)
    actions, log_probs, values = worker.get_actions(observations)
    
    print(f"\nTest action selection:")
    print(f"Actions: {actions}")
    print(f"Log probs: {log_probs}")
    print(f"Values: {values}")
    
    # Store some experiences
    for step in range(20):
        next_observations = np.random.randn(num_agents, obs_dim).astype(np.float32)
        rewards = np.random.randn(num_agents).astype(np.float32)
        
        worker.store_batch_experience(
            local_obs=observations,
            global_state=observations,
            actions=actions,
            rewards=rewards,
            next_local_obs=next_observations,
            next_global_state=next_observations,
            done=(step == 19),
            log_probs=log_probs,
            values=values
        )
        
        observations = next_observations
        actions, log_probs, values = worker.get_actions(observations)
    
    print(f"\nBuffer size: {len(worker.experience_buffer)}")
    
    # Test update
    metrics = worker.update(n_epochs=2, batch_size=32)
    print(f"\nTraining metrics: {metrics}")
