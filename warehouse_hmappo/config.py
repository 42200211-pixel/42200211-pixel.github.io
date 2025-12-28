"""
Configuration module for HMAPPO warehouse robot optimization.

This module provides default configurations and utilities for
customizing training parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
import os


@dataclass
class EnvironmentConfig:
    """Configuration for warehouse environment."""
    N: int = 10  # Number of shelves
    X: int = 4  # Number of item types (= robots)
    Z_init: int = 5  # Initial aisle width
    Z_min: int = 2  # Minimum aisle width
    Z_max: int = 10  # Maximum aisle width
    max_steps: int = 500  # Max steps per episode
    local_obs_size: int = 7  # Local observation grid (7x7)
    manager_decision_freq: int = 100  # Manager decision frequency


@dataclass  
class ManagerConfig:
    """Configuration for Manager agent."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64


@dataclass
class WorkerConfig:
    """Configuration for Worker agent."""
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 256


@dataclass
class TrainingPhasesConfig:
    """Configuration for training phases."""
    # Phase 1: Pretrain Workers
    pretrain_episodes: int = 500
    pretrain_Z: int = 5  # Fixed Z during pretraining
    
    # Phase 2: Train Manager
    manager_episodes: int = 300
    
    # Phase 3: Joint Fine-tuning
    finetune_episodes: int = 200
    finetune_lr_scale: float = 0.1
    finetune_max_steps: int = 200  # Shorter episodes


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    log_dir: str = "./logs"
    experiment_name: Optional[str] = None
    save_freq: int = 50
    verbose: bool = True
    log_tensorboard: bool = False


@dataclass
class FullConfig:
    """Complete configuration for HMAPPO training."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    phases: TrainingPhasesConfig = field(default_factory=TrainingPhasesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = "cpu"
    seed: int = 42
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "env": {
                "N": self.env.N,
                "X": self.env.X,
                "Z_init": self.env.Z_init,
                "Z_min": self.env.Z_min,
                "Z_max": self.env.Z_max,
                "max_steps": self.env.max_steps,
                "local_obs_size": self.env.local_obs_size,
                "manager_decision_freq": self.env.manager_decision_freq,
            },
            "manager": {
                "hidden_dims": self.manager.hidden_dims,
                "lr": self.manager.lr,
                "gamma": self.manager.gamma,
                "gae_lambda": self.manager.gae_lambda,
                "clip_epsilon": self.manager.clip_epsilon,
                "value_coef": self.manager.value_coef,
                "entropy_coef": self.manager.entropy_coef,
                "max_grad_norm": self.manager.max_grad_norm,
                "n_epochs": self.manager.n_epochs,
                "batch_size": self.manager.batch_size,
            },
            "worker": {
                "actor_hidden_dims": self.worker.actor_hidden_dims,
                "critic_hidden_dims": self.worker.critic_hidden_dims,
                "lr_actor": self.worker.lr_actor,
                "lr_critic": self.worker.lr_critic,
                "gamma": self.worker.gamma,
                "gae_lambda": self.worker.gae_lambda,
                "clip_epsilon": self.worker.clip_epsilon,
                "value_coef": self.worker.value_coef,
                "entropy_coef": self.worker.entropy_coef,
                "max_grad_norm": self.worker.max_grad_norm,
                "n_epochs": self.worker.n_epochs,
                "batch_size": self.worker.batch_size,
            },
            "phases": {
                "pretrain_episodes": self.phases.pretrain_episodes,
                "pretrain_Z": self.phases.pretrain_Z,
                "manager_episodes": self.phases.manager_episodes,
                "finetune_episodes": self.phases.finetune_episodes,
                "finetune_lr_scale": self.phases.finetune_lr_scale,
                "finetune_max_steps": self.phases.finetune_max_steps,
            },
            "logging": {
                "log_dir": self.logging.log_dir,
                "experiment_name": self.logging.experiment_name,
                "save_freq": self.logging.save_freq,
                "verbose": self.logging.verbose,
            },
            "device": self.device,
            "seed": self.seed,
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FullConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Update env config
        for key, value in data.get("env", {}).items():
            if hasattr(config.env, key):
                setattr(config.env, key, value)
        
        # Update manager config
        for key, value in data.get("manager", {}).items():
            if hasattr(config.manager, key):
                setattr(config.manager, key, value)
        
        # Update worker config
        for key, value in data.get("worker", {}).items():
            if hasattr(config.worker, key):
                setattr(config.worker, key, value)
        
        # Update phases config
        for key, value in data.get("phases", {}).items():
            if hasattr(config.phases, key):
                setattr(config.phases, key, value)
        
        # Update logging config
        for key, value in data.get("logging", {}).items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)
        
        # Update top-level
        config.device = data.get("device", "cpu")
        config.seed = data.get("seed", 42)
        
        return config


# Predefined configurations for different scenarios
def get_small_config() -> FullConfig:
    """Configuration for small-scale testing."""
    config = FullConfig()
    config.env.N = 4
    config.env.X = 3
    config.env.max_steps = 200
    config.phases.pretrain_episodes = 100
    config.phases.manager_episodes = 50
    config.phases.finetune_episodes = 50
    return config


def get_medium_config() -> FullConfig:
    """Configuration for medium-scale training."""
    config = FullConfig()
    config.env.N = 10
    config.env.X = 5
    config.env.max_steps = 500
    config.phases.pretrain_episodes = 500
    config.phases.manager_episodes = 300
    config.phases.finetune_episodes = 200
    return config


def get_large_config() -> FullConfig:
    """Configuration for large-scale training."""
    config = FullConfig()
    config.env.N = 20
    config.env.X = 8
    config.env.max_steps = 1000
    config.phases.pretrain_episodes = 1000
    config.phases.manager_episodes = 500
    config.phases.finetune_episodes = 300
    return config


# Example usage
if __name__ == "__main__":
    # Create and display configuration
    config = get_medium_config()
    
    print("HMAPPO Configuration")
    print("=" * 50)
    print(f"\nEnvironment:")
    print(f"  Shelves (N): {config.env.N}")
    print(f"  Robots (X): {config.env.X}")
    print(f"  Initial Z: {config.env.Z_init}")
    print(f"  Max steps: {config.env.max_steps}")
    
    print(f"\nManager Agent:")
    print(f"  Hidden dims: {config.manager.hidden_dims}")
    print(f"  Learning rate: {config.manager.lr}")
    
    print(f"\nWorker Agent:")
    print(f"  Actor hidden dims: {config.worker.actor_hidden_dims}")
    print(f"  Critic hidden dims: {config.worker.critic_hidden_dims}")
    print(f"  Learning rate (actor): {config.worker.lr_actor}")
    
    print(f"\nTraining Phases:")
    print(f"  Pretrain episodes: {config.phases.pretrain_episodes}")
    print(f"  Manager episodes: {config.phases.manager_episodes}")
    print(f"  Finetune episodes: {config.phases.finetune_episodes}")
    
    # Save and load test
    config.save("/tmp/test_config.json")
    loaded_config = FullConfig.load("/tmp/test_config.json")
    print(f"\nConfig saved and loaded successfully!")
