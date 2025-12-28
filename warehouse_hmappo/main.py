#!/usr/bin/env python3
"""
Main entry point for HMAPPO Warehouse Robot Optimization.

Usage:
    python -m warehouse_hmappo.main --config small --device cpu
    python -m warehouse_hmappo.main --config medium --device cuda
    python -m warehouse_hmappo.main --N 10 --X 4 --episodes 1000

This implements the Hierarchical Multi-Agent PPO (HMAPPO) algorithm for
optimizing warehouse robot operations as described in the thesis.
"""

import argparse
import os
import sys
import numpy as np
import torch
import random

from .config import (
    FullConfig,
    get_small_config,
    get_medium_config,
    get_large_config
)
from .training.trainer import HMAPPOTrainer, TrainingConfig


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HMAPPO for Smart Warehouse Robot Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", type=str, default="small",
        choices=["small", "medium", "large", "custom"],
        help="Predefined configuration to use"
    )
    parser.add_argument(
        "--config-file", type=str, default=None,
        help="Path to custom configuration JSON file"
    )
    
    # Environment overrides
    parser.add_argument("--N", type=int, default=None, help="Number of shelves")
    parser.add_argument("--X", type=int, default=None, help="Number of robots")
    parser.add_argument("--Z", type=int, default=None, help="Initial aisle width")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    
    # Training overrides
    parser.add_argument(
        "--pretrain-episodes", type=int, default=None,
        help="Number of worker pretraining episodes"
    )
    parser.add_argument(
        "--manager-episodes", type=int, default=None,
        help="Number of manager training episodes"
    )
    parser.add_argument(
        "--finetune-episodes", type=int, default=None,
        help="Number of fine-tuning episodes"
    )
    
    # System
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    # Modes
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation instead of training"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for evaluation"
    )
    
    return parser.parse_args()


def build_config(args) -> TrainingConfig:
    """Build training configuration from arguments."""
    # Start with predefined config
    if args.config == "small":
        base_config = get_small_config()
    elif args.config == "medium":
        base_config = get_medium_config()
    elif args.config == "large":
        base_config = get_large_config()
    elif args.config_file:
        base_config = FullConfig.load(args.config_file)
    else:
        base_config = FullConfig()
    
    # Apply overrides
    if args.N is not None:
        base_config.env.N = args.N
    if args.X is not None:
        base_config.env.X = args.X
    if args.Z is not None:
        base_config.env.Z_init = args.Z
    if args.max_steps is not None:
        base_config.env.max_steps = args.max_steps
    if args.pretrain_episodes is not None:
        base_config.phases.pretrain_episodes = args.pretrain_episodes
    if args.manager_episodes is not None:
        base_config.phases.manager_episodes = args.manager_episodes
    if args.finetune_episodes is not None:
        base_config.phases.finetune_episodes = args.finetune_episodes
    
    # System settings
    base_config.device = args.device
    base_config.seed = args.seed
    base_config.logging.log_dir = args.log_dir
    base_config.logging.experiment_name = args.exp_name
    base_config.logging.verbose = not args.quiet
    
    # Convert to TrainingConfig for trainer
    training_config = TrainingConfig(
        N=base_config.env.N,
        X=base_config.env.X,
        Z_init=base_config.env.Z_init,
        max_steps=base_config.env.max_steps,
        local_obs_size=base_config.env.local_obs_size,
        pretrain_episodes=base_config.phases.pretrain_episodes,
        manager_episodes=base_config.phases.manager_episodes,
        finetune_episodes=base_config.phases.finetune_episodes,
        manager_lr=base_config.manager.lr,
        manager_gamma=base_config.manager.gamma,
        manager_hidden_dims=base_config.manager.hidden_dims,
        manager_decision_freq=base_config.env.manager_decision_freq,
        worker_lr_actor=base_config.worker.lr_actor,
        worker_lr_critic=base_config.worker.lr_critic,
        worker_gamma=base_config.worker.gamma,
        worker_hidden_dims=base_config.worker.actor_hidden_dims,
        clip_epsilon=base_config.manager.clip_epsilon,
        value_coef=base_config.manager.value_coef,
        entropy_coef=base_config.manager.entropy_coef,
        max_grad_norm=base_config.manager.max_grad_norm,
        n_epochs=base_config.manager.n_epochs,
        batch_size=base_config.worker.batch_size,
        gae_lambda=base_config.manager.gae_lambda,
        finetune_lr_scale=base_config.phases.finetune_lr_scale,
        log_dir=base_config.logging.log_dir,
        save_freq=base_config.logging.save_freq,
        verbose=base_config.logging.verbose,
        device=base_config.device
    )
    
    return training_config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Build config
    config = build_config(args)
    
    # Print banner
    print("=" * 60)
    print(" HMAPPO: Hierarchical Multi-Agent PPO")
    print(" Smart Warehouse Robot Optimization")
    print("=" * 60)
    print(f"\nConfiguration: {args.config}")
    print(f"Device: {config.device}")
    print(f"Seed: {args.seed}")
    print()
    
    # Create trainer
    trainer = HMAPPOTrainer(config)
    
    if args.evaluate:
        # Evaluation mode
        if args.checkpoint:
            trainer.load_models(args.checkpoint)
        
        print("Running evaluation...")
        metrics = trainer.evaluate(n_episodes=10)
        
        print("\nEvaluation Results:")
        print(f"  Total Reward: {metrics['total_reward']:.2f}")
        print(f"  Deliveries: {metrics['deliveries']:.2f}")
        print(f"  Collisions: {metrics['collisions']:.2f}")
        print(f"  Throughput: {metrics['throughput']:.4f}")
        print(f"  Avg Travel Time: {metrics['avg_travel_time']:.2f}")
    else:
        # Training mode
        print("Starting training...")
        print(f"  Pretrain episodes: {config.pretrain_episodes}")
        print(f"  Manager episodes: {config.manager_episodes}")
        print(f"  Finetune episodes: {config.finetune_episodes}")
        print()
        
        trainer.train()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
