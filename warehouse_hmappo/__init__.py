"""
Hierarchical Multi-Agent PPO (HMAPPO) for Smart Warehouse Robot Optimization
============================================================================

This package implements a two-level hierarchical reinforcement learning system
for optimizing warehouse robot operations:

Level 1 - Manager Agent (PPO):
    - Optimizes global parameters (aisle width Z, traffic flow policies)
    - Uses compressed state vectors (not full grid)
    - Episodic reward based on throughput and collision metrics

Level 2 - Worker Agents (MAPPO):
    - Individual robot control with shared policy
    - Local observations (7x7 or 9x9 grid around robot)
    - Centralized training, decentralized execution

Training Strategy:
    1. Pretrain Workers with fixed large Z
    2. Train Manager with frozen Workers
    3. Joint fine-tuning with small learning rate
"""

__version__ = "1.0.0"
__author__ = "Warehouse Robot Optimization Team"

from .envs.warehouse_env import WarehouseEnv
from .envs.warehouse_layout import WarehouseLayout
from .agents.robot_agent import RobotAgent
from .agents.manager_agent import ManagerAgent
from .agents.worker_agent import WorkerAgent
from .training.trainer import HMAPPOTrainer

__all__ = [
    "WarehouseEnv",
    "WarehouseLayout", 
    "RobotAgent",
    "ManagerAgent",
    "WorkerAgent",
    "HMAPPOTrainer",
]
