"""
RobotAgent - Individual robot agent wrapper for MAPPO.

This module provides the RobotAgent class which wraps individual robot
state and interfaces with the Worker policy network.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RobotExperience:
    """Single experience tuple for a robot."""
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    log_prob: float
    value: float


class RobotAgent:
    """
    Individual robot agent that interfaces with Worker policy.
    
    Each robot agent:
    - Stores local state and task information
    - Requests actions from shared Worker policy
    - Collects experience for training
    
    Attributes:
        id (int): Unique robot identifier
        item_type (int): Type of item this robot handles
        experience_buffer (list): Buffer of experiences for training
    """
    
    def __init__(self, robot_id: int, item_type: int):
        """
        Initialize robot agent.
        
        Args:
            robot_id: Unique identifier for this robot
            item_type: Item type this robot is responsible for
        """
        self.id = robot_id
        self.item_type = item_type
        
        # Experience buffer
        self.experience_buffer = []
        
        # Current state cache
        self.last_observation: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_log_prob: Optional[float] = None
        self.last_value: Optional[float] = None
        
        # Statistics
        self.total_steps = 0
        self.total_deliveries = 0
        self.total_rewards = 0.0
        
    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """
        Store a transition in the experience buffer.
        
        Args:
            observation: State observation
            action: Action taken
            reward: Reward received
            next_observation: Next state observation
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value estimate from critic
        """
        experience = RobotExperience(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.experience_buffer.append(experience)
        
        # Update statistics
        self.total_steps += 1
        self.total_rewards += reward
        
    def get_experiences(self) -> list:
        """Get all stored experiences."""
        return self.experience_buffer
    
    def clear_buffer(self):
        """Clear the experience buffer."""
        self.experience_buffer = []
        
    def set_last_action_info(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        value: float
    ):
        """
        Cache action information for later transition storage.
        
        Args:
            observation: Current observation
            action: Action to take
            log_prob: Log probability of action
            value: Value estimate
        """
        self.last_observation = observation.copy()
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value
        
    def complete_transition(
        self,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        """
        Complete a transition using cached action info and new info.
        
        Args:
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        if self.last_observation is not None:
            self.store_transition(
                observation=self.last_observation,
                action=self.last_action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                log_prob=self.last_log_prob,
                value=self.last_value
            )
            
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "id": self.id,
            "item_type": self.item_type,
            "total_steps": self.total_steps,
            "total_deliveries": self.total_deliveries,
            "total_rewards": self.total_rewards,
            "avg_reward": self.total_rewards / max(1, self.total_steps),
            "buffer_size": len(self.experience_buffer)
        }
        
    def reset_stats(self):
        """Reset statistics for new training period."""
        self.total_steps = 0
        self.total_deliveries = 0
        self.total_rewards = 0.0
        
    def __repr__(self) -> str:
        return f"RobotAgent(id={self.id}, type={self.item_type}, steps={self.total_steps})"


class RobotAgentManager:
    """
    Manager for multiple robot agents.
    
    Coordinates multiple RobotAgent instances for multi-agent training.
    """
    
    def __init__(self, num_robots: int):
        """
        Initialize robot agent manager.
        
        Args:
            num_robots: Number of robots to manage
        """
        self.num_robots = num_robots
        self.agents = [
            RobotAgent(robot_id=i, item_type=i)
            for i in range(num_robots)
        ]
        
    def get_agent(self, robot_id: int) -> RobotAgent:
        """Get agent by ID."""
        return self.agents[robot_id]
    
    def store_all_transitions(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray
    ):
        """
        Store transitions for all agents.
        
        Args:
            observations: Shape [num_robots, obs_dim]
            actions: Shape [num_robots]
            rewards: Shape [num_robots]
            next_observations: Shape [num_robots, obs_dim]
            dones: Shape [num_robots]
            log_probs: Shape [num_robots]
            values: Shape [num_robots]
        """
        for i, agent in enumerate(self.agents):
            agent.store_transition(
                observation=observations[i],
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_observation=next_observations[i],
                done=bool(dones[i]),
                log_prob=float(log_probs[i]),
                value=float(values[i])
            )
            
    def get_all_experiences(self) -> Tuple[list, ...]:
        """
        Get all experiences from all agents, combined.
        
        Returns:
            Tuple of (observations, actions, rewards, next_obs, dones, log_probs, values)
        """
        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        all_dones = []
        all_log_probs = []
        all_values = []
        
        for agent in self.agents:
            for exp in agent.get_experiences():
                all_obs.append(exp.observation)
                all_actions.append(exp.action)
                all_rewards.append(exp.reward)
                all_next_obs.append(exp.next_observation)
                all_dones.append(exp.done)
                all_log_probs.append(exp.log_prob)
                all_values.append(exp.value)
        
        if not all_obs:
            return tuple([np.array([]) for _ in range(7)])
        
        return (
            np.array(all_obs),
            np.array(all_actions),
            np.array(all_rewards),
            np.array(all_next_obs),
            np.array(all_dones),
            np.array(all_log_probs),
            np.array(all_values)
        )
    
    def clear_all_buffers(self):
        """Clear experience buffers for all agents."""
        for agent in self.agents:
            agent.clear_buffer()
            
    def get_all_stats(self) -> Dict[str, Any]:
        """Get combined statistics for all agents."""
        stats = {
            "num_robots": self.num_robots,
            "total_steps": sum(a.total_steps for a in self.agents),
            "total_deliveries": sum(a.total_deliveries for a in self.agents),
            "total_rewards": sum(a.total_rewards for a in self.agents),
            "per_robot": [a.get_stats() for a in self.agents]
        }
        return stats
    
    def reset_all_stats(self):
        """Reset statistics for all agents."""
        for agent in self.agents:
            agent.reset_stats()
