"""
TrainingLogger - Logging utilities for HMAPPO training.

This module provides logging functionality for tracking training metrics,
saving checkpoints, and visualizing results.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


class TrainingLogger:
    """
    Logger for tracking and saving training metrics.
    
    Features:
    - Tracks metrics for both Manager and Worker agents
    - Saves metrics to JSON files
    - Provides summary statistics
    - Supports tensorboard-style logging (optional)
    
    Attributes:
        log_dir (str): Directory for saving logs
        experiment_name (str): Name of current experiment
        metrics (Dict): Dictionary storing all metrics
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        save_freq: int = 100,
        verbose: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for saving logs
            experiment_name: Name of experiment (auto-generated if None)
            save_freq: How often to save metrics (in episodes)
            verbose: Whether to print progress
        """
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.verbose = verbose
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"hmappo_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Metrics storage
        self.metrics: Dict[str, List] = defaultdict(list)
        self.episode_metrics: Dict[str, Any] = {}
        
        # Timing
        self.start_time = time.time()
        self.episode_start_time = time.time()
        
        # Episode counter
        self.current_episode = 0
        self.current_step = 0
        
        # Best metrics for checkpointing
        self.best_reward = float('-inf')
        self.best_throughput = 0.0
        
        if self.verbose:
            print(f"Logging to: {self.experiment_dir}")
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (uses current_step if None)
        """
        if step is None:
            step = self.current_step
        
        self.metrics[name].append({
            'step': step,
            'value': value,
            'time': time.time() - self.start_time
        })
        
        self.episode_metrics[name] = value
    
    def log_dict(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric name -> value
            step: Step number
        """
        for name, value in metrics_dict.items():
            self.log_scalar(name, value, step)
    
    def log_episode_start(self):
        """Mark the start of an episode."""
        self.episode_start_time = time.time()
        self.episode_metrics = {}
    
    def log_episode_end(
        self,
        episode: int,
        total_steps: int,
        rewards: Dict[str, float],
        metrics: Dict[str, Any],
        extra_info: Optional[Dict] = None
    ):
        """
        Log end-of-episode statistics.
        
        Args:
            episode: Episode number
            total_steps: Total steps in episode
            rewards: Dictionary of reward values
            metrics: Environment metrics
            extra_info: Additional information to log
        """
        self.current_episode = episode
        self.current_step += total_steps
        
        episode_time = time.time() - self.episode_start_time
        
        # Log core metrics
        self.log_scalar("episode", episode)
        self.log_scalar("total_steps", total_steps)
        self.log_scalar("episode_time", episode_time)
        self.log_scalar("steps_per_second", total_steps / max(episode_time, 1e-6))
        
        # Log rewards
        for name, value in rewards.items():
            self.log_scalar(f"reward/{name}", value)
        
        # Log environment metrics
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.log_scalar(f"env/{name}", value)
        else:
            # Handle EpisodeMetrics dataclass
            self.log_scalar("env/throughput", metrics.throughput)
            self.log_scalar("env/collision_rate", metrics.collision_rate)
            self.log_scalar("env/deadlock_ratio", metrics.deadlock_ratio)
            self.log_scalar("env/avg_travel_time", metrics.avg_travel_time)
            self.log_scalar("env/total_deliveries", metrics.total_deliveries)
            self.log_scalar("env/total_collisions", metrics.total_collisions)
        
        # Log extra info
        if extra_info:
            for name, value in extra_info.items():
                if isinstance(value, (int, float)):
                    self.log_scalar(f"info/{name}", value)
        
        # Check for best metrics
        total_reward = sum(rewards.values())
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.log_scalar("best_reward", total_reward)
        
        # Print progress
        if self.verbose:
            self._print_progress(episode, total_steps, rewards, metrics)
        
        # Save periodically
        if episode % self.save_freq == 0:
            self.save_metrics()
    
    def log_training_update(
        self,
        agent_type: str,
        update_metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics from a training update.
        
        Args:
            agent_type: "manager" or "worker"
            update_metrics: Dictionary of training metrics
            step: Step number
        """
        for name, value in update_metrics.items():
            self.log_scalar(f"{agent_type}/{name}", value, step)
    
    def _print_progress(
        self,
        episode: int,
        total_steps: int,
        rewards: Dict[str, float],
        metrics: Any
    ):
        """Print training progress to console."""
        elapsed = time.time() - self.start_time
        
        # Format metrics
        if hasattr(metrics, 'throughput'):
            throughput = metrics.throughput
            collisions = metrics.total_collisions
            deliveries = metrics.total_deliveries
        else:
            throughput = metrics.get('throughput', 0)
            collisions = metrics.get('total_collisions', 0)
            deliveries = metrics.get('total_deliveries', 0)
        
        total_reward = sum(rewards.values())
        
        print(f"Episode {episode:5d} | "
              f"Steps: {total_steps:5d} | "
              f"Reward: {total_reward:8.2f} | "
              f"Throughput: {throughput:.4f} | "
              f"Deliveries: {deliveries:4d} | "
              f"Collisions: {collisions:4d} | "
              f"Time: {elapsed:.0f}s")
    
    def save_metrics(self, filename: Optional[str] = None):
        """
        Save all metrics to JSON file.
        
        Args:
            filename: Custom filename (auto-generated if None)
        """
        if filename is None:
            filename = "metrics.json"
        
        filepath = os.path.join(self.experiment_dir, filename)
        
        # Convert metrics to serializable format
        metrics_data = {}
        for name, values in self.metrics.items():
            metrics_data[name] = values
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        if self.verbose:
            print(f"Metrics saved to: {filepath}")
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        filepath = os.path.join(self.experiment_dir, "config.json")
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                vals = [v['value'] for v in values]
                summary[name] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'last': vals[-1]
                }
        
        return summary
    
    def get_recent_metrics(self, n: int = 10) -> Dict[str, float]:
        """
        Get average of recent metric values.
        
        Args:
            n: Number of recent values to average
            
        Returns:
            Dictionary of metric name -> recent average
        """
        recent = {}
        
        for name, values in self.metrics.items():
            if values:
                recent_vals = [v['value'] for v in values[-n:]]
                recent[name] = float(np.mean(recent_vals))
        
        return recent
    
    def log_phase_start(self, phase_name: str):
        """
        Log the start of a training phase.
        
        Args:
            phase_name: Name of the phase (e.g., "pretrain_worker")
        """
        self.log_scalar(f"phase/{phase_name}_start", self.current_episode)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting phase: {phase_name}")
            print(f"{'='*60}\n")
    
    def log_phase_end(self, phase_name: str):
        """
        Log the end of a training phase.
        
        Args:
            phase_name: Name of the phase
        """
        self.log_scalar(f"phase/{phase_name}_end", self.current_episode)
        
        if self.verbose:
            summary = self.get_summary()
            print(f"\n{'='*60}")
            print(f"Phase completed: {phase_name}")
            print(f"Episodes: {self.current_episode}")
            print(f"Total time: {time.time() - self.start_time:.0f}s")
            print(f"{'='*60}\n")
    
    def close(self):
        """Clean up and save final metrics."""
        self.save_metrics()
        
        # Save summary
        summary = self.get_summary()
        summary_path = os.path.join(self.experiment_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"\nTraining complete!")
            print(f"Total episodes: {self.current_episode}")
            print(f"Total time: {time.time() - self.start_time:.0f}s")
            print(f"Logs saved to: {self.experiment_dir}")


class MetricsAggregator:
    """
    Aggregates metrics across multiple episodes for reporting.
    
    Useful for computing rolling averages and tracking trends.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize aggregator.
        
        Args:
            window_size: Window size for rolling statistics
        """
        self.window_size = window_size
        self.buffers: Dict[str, List[float]] = defaultdict(list)
    
    def add(self, name: str, value: float):
        """Add a value to the buffer."""
        self.buffers[name].append(value)
        
        # Keep buffer size limited
        if len(self.buffers[name]) > self.window_size * 2:
            self.buffers[name] = self.buffers[name][-self.window_size:]
    
    def get_rolling_mean(self, name: str) -> float:
        """Get rolling mean for a metric."""
        values = self.buffers.get(name, [])
        if not values:
            return 0.0
        return float(np.mean(values[-self.window_size:]))
    
    def get_rolling_std(self, name: str) -> float:
        """Get rolling std for a metric."""
        values = self.buffers.get(name, [])
        if len(values) < 2:
            return 0.0
        return float(np.std(values[-self.window_size:]))
    
    def get_all_rolling_means(self) -> Dict[str, float]:
        """Get rolling means for all metrics."""
        return {name: self.get_rolling_mean(name) for name in self.buffers}
    
    def reset(self):
        """Reset all buffers."""
        self.buffers.clear()


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = TrainingLogger(
        log_dir="./test_logs",
        experiment_name="test_experiment",
        verbose=True
    )
    
    # Simulate training
    for episode in range(5):
        logger.log_episode_start()
        
        # Simulate some training
        total_steps = np.random.randint(100, 200)
        rewards = {
            "manager": np.random.randn() * 10,
            "worker_mean": np.random.randn() * 5
        }
        
        # Create mock metrics
        class MockMetrics:
            throughput = np.random.random() * 0.1
            collision_rate = np.random.random() * 0.05
            deadlock_ratio = np.random.random() * 0.01
            avg_travel_time = np.random.random() * 100
            total_deliveries = np.random.randint(10, 50)
            total_collisions = np.random.randint(0, 10)
        
        logger.log_episode_end(
            episode=episode,
            total_steps=total_steps,
            rewards=rewards,
            metrics=MockMetrics()
        )
    
    # Get summary
    summary = logger.get_summary()
    print("\nSummary:")
    for name, stats in list(summary.items())[:5]:
        print(f"  {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    # Clean up
    logger.close()
    
    # Clean up test directory
    import shutil
    shutil.rmtree("./test_logs", ignore_errors=True)
    
    print("\nLogger test completed!")
