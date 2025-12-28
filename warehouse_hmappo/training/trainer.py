"""
HMAPPOTrainer - Main training loop for Hierarchical Multi-Agent PPO.

This module implements the complete training pipeline for HMAPPO with:
1. Phase 1: Pretrain Workers (MAPPO) with fixed large Z
2. Phase 2: Train Manager (PPO) with frozen Workers  
3. Phase 3: Joint fine-tuning with small learning rate

The trainer handles:
- Centralized training with decentralized execution for Workers
- Episode-level Manager decisions
- Step-level Worker actions
- Proper separation of concerns between hierarchy levels
"""

import os
import numpy as np
import torch
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

from ..envs.warehouse_env import WarehouseEnv, ManagerAction
from ..agents.manager_agent import ManagerAgent
from ..agents.worker_agent import WorkerAgent
from ..utils.logger import TrainingLogger


@dataclass
class TrainingConfig:
    """Configuration for HMAPPO training."""
    # Environment
    N: int = 10  # Number of shelves
    X: int = 4  # Number of item types / robots
    Z_init: int = 5  # Initial aisle width
    max_steps: int = 500  # Max steps per episode
    local_obs_size: int = 7  # Local observation grid size
    
    # Training phases
    pretrain_episodes: int = 500  # Phase 1: Pretrain Workers
    manager_episodes: int = 300  # Phase 2: Train Manager
    finetune_episodes: int = 200  # Phase 3: Joint fine-tuning
    
    # Manager hyperparameters
    manager_lr: float = 3e-4
    manager_gamma: float = 0.99
    manager_hidden_dims: list = None
    manager_decision_freq: int = 100  # Manager decides every N steps
    
    # Worker hyperparameters
    worker_lr_actor: float = 3e-4
    worker_lr_critic: float = 5e-4
    worker_gamma: float = 0.99
    worker_hidden_dims: list = None
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 256
    gae_lambda: float = 0.95
    
    # Fine-tuning
    finetune_lr_scale: float = 0.1  # Scale down LR for fine-tuning
    
    # Logging
    log_dir: str = "./logs"
    save_freq: int = 50
    verbose: bool = True
    
    # Device
    device: str = "cpu"
    
    def __post_init__(self):
        if self.manager_hidden_dims is None:
            self.manager_hidden_dims = [128, 64]
        if self.worker_hidden_dims is None:
            self.worker_hidden_dims = [256, 128]


class HMAPPOTrainer:
    """
    Hierarchical Multi-Agent PPO Trainer.
    
    Implements the three-phase training strategy:
    
    Phase 1 - Pretrain Workers:
        - Fix Z at large value (reduce congestion)
        - Train MAPPO workers to navigate and avoid collisions
        - Manager is inactive
    
    Phase 2 - Train Manager:
        - Freeze Worker policy
        - Train Manager PPO to optimize Z
        - Manager receives episode-level rewards
    
    Phase 3 - Joint Fine-tuning:
        - Unfreeze Workers with small learning rate
        - Continue training both levels
        - Shorter episodes for stability
    
    Attributes:
        config (TrainingConfig): Training configuration
        env (WarehouseEnv): Warehouse environment
        manager (ManagerAgent): Manager agent
        worker (WorkerAgent): Worker agent (shared policy)
        logger (TrainingLogger): Training logger
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration (uses default if None)
        """
        self.config = config or TrainingConfig()
        
        # Initialize environment
        self.env = WarehouseEnv(
            N=self.config.N,
            X=self.config.X,
            Z=self.config.Z_init,
            max_steps=self.config.max_steps,
            local_obs_size=self.config.local_obs_size,
            manager_decision_freq=self.config.manager_decision_freq
        )
        
        # Initialize agents
        self._init_agents()
        
        # Initialize logger
        self.logger = TrainingLogger(
            log_dir=self.config.log_dir,
            save_freq=self.config.save_freq,
            verbose=self.config.verbose
        )
        
        # Save config
        self.logger.save_config(self._config_to_dict())
        
        # Training state
        self.total_episodes = 0
        self.current_phase = 0
        
    def _init_agents(self):
        """Initialize Manager and Worker agents."""
        # Manager agent
        self.manager = ManagerAgent(
            obs_dim=self.env.manager_obs_dim,
            action_dim=self.env.manager_action_dim,
            hidden_dims=self.config.manager_hidden_dims,
            lr=self.config.manager_lr,
            gamma=self.config.manager_gamma,
            gae_lambda=self.config.gae_lambda,
            clip_epsilon=self.config.clip_epsilon,
            value_coef=self.config.value_coef,
            entropy_coef=self.config.entropy_coef,
            max_grad_norm=self.config.max_grad_norm,
            device=self.config.device
        )
        
        # Worker agent (shared policy for all robots)
        self.worker = WorkerAgent(
            num_agents=self.config.X,
            obs_dim=self.env.worker_obs_dim,
            action_dim=self.env.worker_action_dim,
            actor_hidden_dims=self.config.worker_hidden_dims,
            critic_hidden_dims=[512, 256, 128],
            lr_actor=self.config.worker_lr_actor,
            lr_critic=self.config.worker_lr_critic,
            gamma=self.config.worker_gamma,
            gae_lambda=self.config.gae_lambda,
            clip_epsilon=self.config.clip_epsilon,
            value_coef=self.config.value_coef,
            entropy_coef=self.config.entropy_coef,
            max_grad_norm=self.config.max_grad_norm,
            device=self.config.device
        )
        
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "N": self.config.N,
            "X": self.config.X,
            "Z_init": self.config.Z_init,
            "max_steps": self.config.max_steps,
            "local_obs_size": self.config.local_obs_size,
            "pretrain_episodes": self.config.pretrain_episodes,
            "manager_episodes": self.config.manager_episodes,
            "finetune_episodes": self.config.finetune_episodes,
            "manager_lr": self.config.manager_lr,
            "worker_lr_actor": self.config.worker_lr_actor,
            "worker_lr_critic": self.config.worker_lr_critic,
            "clip_epsilon": self.config.clip_epsilon,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "device": self.config.device
        }
    
    def train(self):
        """
        Run complete three-phase training.
        
        This is the main entry point for training.
        """
        print("="*60)
        print("HMAPPO Training for Smart Warehouse Robot Optimization")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Shelves (N): {self.config.N}")
        print(f"  Robots (X): {self.config.X}")
        print(f"  Initial Z: {self.config.Z_init}")
        print(f"  Device: {self.config.device}")
        print()
        
        # Phase 1: Pretrain Workers
        self.phase1_pretrain_workers()
        
        # Phase 2: Train Manager
        self.phase2_train_manager()
        
        # Phase 3: Joint Fine-tuning
        self.phase3_joint_finetune()
        
        # Save final models
        self.save_models("final")
        
        # Close logger
        self.logger.close()
        
        print("\nTraining complete!")
        
    def phase1_pretrain_workers(self):
        """
        Phase 1: Pretrain Workers with fixed large Z.
        
        Goal: Train workers to navigate and avoid collisions without
        Manager intervention. Uses large Z to reduce congestion.
        """
        self.logger.log_phase_start("pretrain_workers")
        self.current_phase = 1
        
        # Set large Z to reduce congestion
        large_Z = max(self.config.Z_init, 5)
        self.env.reset(new_Z=large_Z)
        
        # Freeze manager (not used in this phase)
        self.manager.freeze()
        
        # Train workers
        for episode in range(self.config.pretrain_episodes):
            episode_metrics = self._run_episode(
                train_worker=True,
                train_manager=False,
                fixed_Z=large_Z
            )
            
            # Log progress
            self._log_episode(episode, episode_metrics)
            
            # Update worker
            if (episode + 1) % 10 == 0:
                worker_metrics = self.worker.update(
                    n_epochs=self.config.n_epochs,
                    batch_size=self.config.batch_size
                )
                self.logger.log_training_update("worker", worker_metrics)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_freq == 0:
                self.save_models(f"phase1_ep{episode+1}")
        
        self.total_episodes += self.config.pretrain_episodes
        self.logger.log_phase_end("pretrain_workers")
        
    def phase2_train_manager(self):
        """
        Phase 2: Train Manager with frozen Workers.
        
        Goal: Train Manager to optimize Z (and potentially traffic flow)
        while Workers execute their learned policy.
        """
        self.logger.log_phase_start("train_manager")
        self.current_phase = 2
        
        # Freeze workers, unfreeze manager
        self.worker.freeze()
        self.manager.unfreeze()
        
        # Train manager
        for episode in range(self.config.manager_episodes):
            episode_metrics = self._run_episode(
                train_worker=False,
                train_manager=True
            )
            
            # Log progress
            self._log_episode(episode + self.config.pretrain_episodes, episode_metrics)
            
            # Update manager
            if (episode + 1) % 10 == 0:
                manager_metrics = self.manager.update(
                    n_epochs=self.config.n_epochs,
                    batch_size=min(64, self.config.batch_size)
                )
                self.logger.log_training_update("manager", manager_metrics)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_freq == 0:
                self.save_models(f"phase2_ep{episode+1}")
        
        self.total_episodes += self.config.manager_episodes
        self.logger.log_phase_end("train_manager")
        
    def phase3_joint_finetune(self):
        """
        Phase 3: Joint fine-tuning of Manager and Workers.
        
        Goal: Fine-tune both levels together with small learning rate
        for improved coordination.
        """
        self.logger.log_phase_start("joint_finetune")
        self.current_phase = 3
        
        # Unfreeze both agents
        self.manager.unfreeze()
        self.worker.unfreeze()
        
        # Reduce learning rates for stability
        scale = self.config.finetune_lr_scale
        self.manager.set_learning_rate(self.config.manager_lr * scale)
        self.worker.set_learning_rate(
            self.config.worker_lr_actor * scale,
            self.config.worker_lr_critic * scale
        )
        
        # Fine-tune with shorter episodes
        original_max_steps = self.config.max_steps
        self.env.max_steps = min(200, original_max_steps)
        
        for episode in range(self.config.finetune_episodes):
            episode_metrics = self._run_episode(
                train_worker=True,
                train_manager=True
            )
            
            # Log progress
            total_ep = (self.config.pretrain_episodes + 
                       self.config.manager_episodes + episode)
            self._log_episode(total_ep, episode_metrics)
            
            # Update both agents
            if (episode + 1) % 5 == 0:
                worker_metrics = self.worker.update(
                    n_epochs=self.config.n_epochs // 2,
                    batch_size=self.config.batch_size
                )
                manager_metrics = self.manager.update(
                    n_epochs=self.config.n_epochs // 2,
                    batch_size=min(64, self.config.batch_size)
                )
                self.logger.log_training_update("worker", worker_metrics)
                self.logger.log_training_update("manager", manager_metrics)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_freq == 0:
                self.save_models(f"phase3_ep{episode+1}")
        
        # Restore max steps
        self.env.max_steps = original_max_steps
        
        self.total_episodes += self.config.finetune_episodes
        self.logger.log_phase_end("joint_finetune")
        
    def _run_episode(
        self,
        train_worker: bool = True,
        train_manager: bool = False,
        fixed_Z: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            train_worker: Whether to collect worker experiences
            train_manager: Whether to collect manager experiences
            fixed_Z: If provided, use this Z value (bypass manager)
            
        Returns:
            Dictionary of episode metrics
        """
        # Reset environment
        if fixed_Z is not None:
            obs = self.env.reset(new_Z=fixed_Z)
        else:
            obs = self.env.reset()
        
        manager_obs = obs["manager_obs"]
        worker_obs = obs["worker_obs"]
        
        # Episode tracking
        total_worker_reward = 0.0
        manager_reward = 0.0
        episode_steps = 0
        
        # Manager decision tracking
        manager_obs_at_decision = manager_obs.copy()
        manager_action = 0  # Default: keep Z
        manager_log_prob = 0.0
        manager_value = 0.0
        
        while True:
            # Manager decision (at start and periodically)
            if train_manager and episode_steps % self.config.manager_decision_freq == 0:
                # Get manager action
                manager_action, manager_log_prob, manager_value = self.manager.get_action(
                    manager_obs, deterministic=not train_manager
                )
                manager_obs_at_decision = manager_obs.copy()
                
                # Apply manager action (modify Z)
                if fixed_Z is None:
                    self.env.manager_step(manager_action)
            
            # Worker actions (step-by-step)
            actions, log_probs, values = self.worker.get_actions(
                worker_obs, deterministic=not train_worker
            )
            
            # Execute actions
            next_obs, rewards, done, info = self.env.step(actions)
            
            next_manager_obs = next_obs["manager_obs"]
            next_worker_obs = next_obs["worker_obs"]
            
            # Collect worker experiences
            if train_worker:
                self.worker.store_batch_experience(
                    local_obs=worker_obs,
                    global_state=worker_obs,  # Use local obs as global state
                    actions=actions,
                    rewards=rewards["worker_rewards"],
                    next_local_obs=next_worker_obs,
                    next_global_state=next_worker_obs,
                    done=done,
                    log_probs=log_probs,
                    values=values
                )
            
            # Update tracking
            total_worker_reward += float(rewards["worker_rewards"].sum())
            manager_reward += rewards["manager_reward"]
            episode_steps += 1
            
            # Update observations
            manager_obs = next_manager_obs
            worker_obs = next_worker_obs
            
            if done:
                break
        
        # Store manager experience (episode-level)
        if train_manager and not (fixed_Z is not None):
            self.manager.store_experience(
                observation=manager_obs_at_decision,
                action=manager_action,
                reward=manager_reward,
                next_observation=manager_obs,
                done=True,
                log_prob=manager_log_prob,
                value=manager_value
            )
        
        return {
            "total_steps": episode_steps,
            "worker_reward": total_worker_reward,
            "manager_reward": manager_reward,
            "env_metrics": info["metrics"],
            "current_Z": self.env.Z,
            "deliveries": info["deliveries"],
            "collisions": info["collisions"]
        }
    
    def _log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log episode metrics."""
        self.logger.log_episode_end(
            episode=episode,
            total_steps=metrics["total_steps"],
            rewards={
                "worker": metrics["worker_reward"],
                "manager": metrics["manager_reward"]
            },
            metrics=metrics["env_metrics"],
            extra_info={
                "current_Z": metrics["current_Z"],
                "phase": self.current_phase
            }
        )
    
    def save_models(self, tag: str):
        """
        Save model checkpoints.
        
        Args:
            tag: Tag for the checkpoint (e.g., "phase1_ep100")
        """
        checkpoint_dir = os.path.join(self.logger.experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.manager.save(os.path.join(checkpoint_dir, f"manager_{tag}.pt"))
        self.worker.save(os.path.join(checkpoint_dir, f"worker_{tag}.pt"))
        
        if self.config.verbose:
            print(f"Models saved: {tag}")
    
    def load_models(self, tag: str, checkpoint_dir: Optional[str] = None):
        """
        Load model checkpoints.
        
        Args:
            tag: Tag for the checkpoint
            checkpoint_dir: Directory containing checkpoints
        """
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.logger.experiment_dir, "checkpoints")
        
        self.manager.load(os.path.join(checkpoint_dir, f"manager_{tag}.pt"))
        self.worker.load(os.path.join(checkpoint_dir, f"worker_{tag}.pt"))
        
        if self.config.verbose:
            print(f"Models loaded: {tag}")
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of average metrics
        """
        metrics_sum = {
            "total_reward": 0.0,
            "deliveries": 0.0,
            "collisions": 0.0,
            "throughput": 0.0,
            "avg_travel_time": 0.0
        }
        
        for _ in range(n_episodes):
            episode_metrics = self._run_episode(
                train_worker=False,
                train_manager=False
            )
            
            metrics_sum["total_reward"] += (episode_metrics["worker_reward"] + 
                                           episode_metrics["manager_reward"])
            metrics_sum["deliveries"] += episode_metrics["deliveries"]
            metrics_sum["collisions"] += episode_metrics["collisions"]
            metrics_sum["throughput"] += episode_metrics["env_metrics"].throughput
            metrics_sum["avg_travel_time"] += episode_metrics["env_metrics"].avg_travel_time
        
        # Average
        for key in metrics_sum:
            metrics_sum[key] /= n_episodes
        
        return metrics_sum


# Pseudo-code for HMAPPO Training Loop
"""
HMAPPO Training Algorithm
=========================

Input:
    - Warehouse parameters: N (shelves), X (robots), Z_init (initial aisle width)
    - Hyperparameters: learning rates, discount factors, etc.

Output:
    - Trained Manager policy π_M
    - Trained Worker policy π_W (shared across all robots)

Algorithm:

Phase 1: Pretrain Workers (MAPPO)
---------------------------------
1. Set Z = Z_large (e.g., 5) to reduce congestion
2. Initialize Worker policy π_W randomly
3. For episode = 1 to pretrain_episodes:
    a. Reset environment with fixed Z
    b. For timestep t = 1 to max_steps:
        i.   For each robot i:
             - Get local observation o_i = LocalGrid + TargetDir + NearbyRobots
        ii.  Sample actions a_i ~ π_W(·|o_i) for all robots
        iii. Execute actions, observe rewards r_i and next states
        iv.  Store (o_i, a_i, r_i, o_i', done) in Worker buffer
    c. Compute advantages using GAE with centralized value function
    d. Update π_W using PPO objective with clipped surrogate loss

Phase 2: Train Manager (PPO)
----------------------------
1. Freeze Worker policy π_W
2. Initialize Manager policy π_M randomly
3. For episode = 1 to manager_episodes:
    a. Get compressed state s_M = [avg_travel_time, robot_density, 
                                   collision_rate, deadlock_ratio, 
                                   throughput, current_Z]
    b. Sample Manager action a_M ~ π_M(·|s_M) 
       Actions: {KEEP_Z, INCREASE_Z, DECREASE_Z}
    c. Apply Manager action (update Z in environment)
    d. Run episode with frozen Workers
    e. Compute episode reward R_M based on:
       - Throughput bonus
       - Travel time penalty
       - Collision penalty
       - Deadlock penalty (strong)
    f. Store (s_M, a_M, R_M, s_M', done) in Manager buffer
    g. Update π_M using PPO

Phase 3: Joint Fine-tuning
--------------------------
1. Unfreeze Worker policy π_W
2. Set small learning rates: lr' = lr * 0.1
3. For episode = 1 to finetune_episodes:
    a. Run episode with both Manager and Workers active
    b. Manager decides Z at start and every K steps
    c. Workers execute step-by-step within Manager's constraints
    d. Update both π_M and π_W with reduced learning rates

Key Design Decisions:
--------------------
1. Manager Observation: Compressed vector, NOT full grid
   - This prevents Manager from micromanaging
   - Forces Manager to make strategic, not tactical decisions

2. Worker Observation: Local grid + relative information
   - Enables decentralized execution
   - Workers don't need global knowledge to act

3. Reward Separation:
   - Workers: Step-level rewards (progress, delivery, collision penalty)
   - Manager: Episode-level rewards (throughput, travel time, safety)

4. No Conflict Design:
   - Manager sets constraints (Z value)
   - Workers operate within constraints
   - Workers handle collision avoidance autonomously
"""


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = TrainingConfig(
        N=4,
        X=3,
        Z_init=4,
        max_steps=200,
        pretrain_episodes=50,  # Reduced for demo
        manager_episodes=30,
        finetune_episodes=20,
        verbose=True
    )
    
    # Create trainer
    trainer = HMAPPOTrainer(config)
    
    print("\nEnvironment info:")
    print(f"  Grid size: {trainer.env.layout.get_dimensions()}")
    print(f"  Worker obs dim: {trainer.env.worker_obs_dim}")
    print(f"  Manager obs dim: {trainer.env.manager_obs_dim}")
    
    # Run training (uncomment for actual training)
    # trainer.train()
    
    # Run a single test episode
    print("\nRunning test episode...")
    metrics = trainer._run_episode(train_worker=False, train_manager=False)
    print(f"  Steps: {metrics['total_steps']}")
    print(f"  Worker reward: {metrics['worker_reward']:.2f}")
    print(f"  Deliveries: {metrics['deliveries']}")
    print(f"  Collisions: {metrics['collisions']}")
    print(f"  Current Z: {metrics['current_Z']}")
    
    print("\nTrainer test completed!")
