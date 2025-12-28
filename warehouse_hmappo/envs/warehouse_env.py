"""
WarehouseEnv - Main simulation environment for warehouse robot optimization.

This environment implements the warehouse simulation where:
- Multiple robots navigate to pick up items and deliver to sorting areas
- Manager agent optimizes global parameters (Z, traffic flow)
- Worker agents control individual robot movements

The environment provides:
- State observations for both Manager (compressed vector) and Workers (local grid)
- Step-by-step simulation with collision detection and deadlock detection
- Reward computation for both hierarchical levels
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import IntEnum
from dataclasses import dataclass, field
import time

from .warehouse_layout import WarehouseLayout


class RobotAction(IntEnum):
    """Actions available to Worker agents (robots)."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    WAIT = 4


class ManagerAction(IntEnum):
    """Actions available to Manager agent."""
    KEEP_Z = 0  # Keep current Z
    INCREASE_Z = 1  # Increase Z by 1
    DECREASE_Z = 2  # Decrease Z by 1 (minimum 2)


class RobotState(IntEnum):
    """Robot operational states."""
    IDLE = 0
    MOVING_TO_PICKUP = 1
    PICKING = 2
    MOVING_TO_DELIVERY = 3
    DELIVERING = 4


@dataclass
class Robot:
    """Represents a single robot in the warehouse."""
    id: int
    item_type: int  # Which item type this robot handles
    row: int
    col: int
    state: RobotState = RobotState.IDLE
    current_task: Optional[Dict] = None
    steps_since_task: int = 0
    total_deliveries: int = 0
    collision_count: int = 0
    wait_count: int = 0


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    total_steps: int = 0
    total_deliveries: int = 0
    total_collisions: int = 0
    total_deadlocks: int = 0
    travel_times: List[int] = field(default_factory=list)
    
    @property
    def avg_travel_time(self) -> float:
        if not self.travel_times:
            return 0.0
        return np.mean(self.travel_times)
    
    @property
    def throughput(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_deliveries / self.total_steps
    
    @property
    def collision_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_collisions / self.total_steps
    
    @property
    def deadlock_ratio(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_deadlocks / self.total_steps


class WarehouseEnv:
    """
    Main warehouse simulation environment.
    
    Implements a two-level hierarchical RL environment:
    - Level 1 (Manager): Optimizes Z and traffic policies episodically
    - Level 2 (Workers): Control individual robots step-by-step
    
    Attributes:
        N (int): Number of shelves
        X (int): Number of item types / robots
        Z (int): Current aisle width
        layout (WarehouseLayout): Physical warehouse layout
        robots (List[Robot]): List of robot objects
        metrics (EpisodeMetrics): Current episode metrics
    """
    
    def __init__(
        self,
        N: int = 10,
        X: int = 4,
        Z: int = 3,
        max_steps: int = 1000,
        local_obs_size: int = 7,
        manager_decision_freq: int = 100,
    ):
        """
        Initialize the warehouse environment.
        
        Args:
            N: Number of shelves
            X: Number of item types (= number of robots)
            Z: Initial aisle width (>= 2)
            max_steps: Maximum steps per episode
            local_obs_size: Size of local observation grid for workers (7 or 9)
            manager_decision_freq: How often Manager makes decisions (in steps)
        """
        self.N = N
        self.X = X
        self.Z = Z
        self.max_steps = max_steps
        self.local_obs_size = local_obs_size
        self.manager_decision_freq = manager_decision_freq
        
        # Validate local_obs_size
        if local_obs_size not in [7, 9]:
            raise ValueError("local_obs_size must be 7 or 9")
        
        # Initialize layout
        self.layout = WarehouseLayout(N, X, Z)
        
        # Initialize robots
        self.robots: List[Robot] = []
        self._init_robots()
        
        # Episode state
        self.current_step = 0
        self.metrics = EpisodeMetrics()
        self.done = False
        
        # Manager state
        self.manager_obs_dim = 7  # Compressed state vector dimension
        self.manager_action_dim = 3  # KEEP, INCREASE, DECREASE Z
        
        # Worker state
        self.worker_obs_dim = self._calculate_worker_obs_dim()
        self.worker_action_dim = 5  # UP, DOWN, LEFT, RIGHT, WAIT
        
        # Deadlock detection
        self.position_history: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(X)}
        self.deadlock_threshold = 10  # Steps without progress
        
    def _init_robots(self):
        """Initialize robots at their starting positions (near sorting areas)."""
        self.robots = []
        
        for i in range(self.X):
            # Start robots near their corresponding sorting positions
            sorting_pos = self.layout.get_sorting_position(i)
            if sorting_pos:
                # Start one row above the sorting position
                start_row = max(0, sorting_pos[0] - 2)
                start_col = sorting_pos[1]
                
                # Ensure position is walkable
                if not self.layout.is_walkable(start_row, start_col):
                    start_row = sorting_pos[0]
            else:
                start_row, start_col = self.layout.Z, self.layout.Z + i * 2
            
            robot = Robot(
                id=i,
                item_type=i,
                row=start_row,
                col=start_col,
                state=RobotState.IDLE
            )
            self.robots.append(robot)
            
    def _calculate_worker_obs_dim(self) -> int:
        """Calculate the observation dimension for worker agents."""
        # Local grid: local_obs_size x local_obs_size
        grid_dim = self.local_obs_size * self.local_obs_size
        
        # Additional features:
        # - Direction to target (2: dx, dy normalized)
        # - Distance to target (1: normalized)
        # - Current Z (1: normalized)
        # - Robot state (1: encoded)
        # - Nearby robot info (8: positions of up to 4 neighbors, 2 each)
        additional_dim = 2 + 1 + 1 + 1 + 8
        
        return grid_dim + additional_dim
    
    def reset(self, new_Z: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment for a new episode.
        
        Args:
            new_Z: Optional new aisle width to use
            
        Returns:
            Dictionary with 'manager_obs' and 'worker_obs' keys
        """
        # Update Z if specified
        if new_Z is not None and new_Z >= 2:
            self.Z = new_Z
            self.layout.update_aisle_width(new_Z)
        
        # Reset robots
        self._init_robots()
        
        # Assign initial tasks to robots
        for robot in self.robots:
            self._assign_task(robot)
        
        # Reset episode state
        self.current_step = 0
        self.metrics = EpisodeMetrics()
        self.done = False
        
        # Reset position history for deadlock detection
        self.position_history = {i: [] for i in range(self.X)}
        
        return self._get_observations()
    
    def _assign_task(self, robot: Robot):
        """Assign a pickup-delivery task to a robot."""
        # Get a task for items of this robot's type
        task = None
        attempts = 0
        max_attempts = 50
        
        while task is None and attempts < max_attempts:
            candidate = self.layout.get_random_pickup_task()
            if candidate and candidate["item_type"] == robot.item_type:
                task = candidate
            attempts += 1
        
        if task:
            robot.current_task = task
            robot.state = RobotState.MOVING_TO_PICKUP
            robot.steps_since_task = 0
        else:
            # No task available, stay idle
            robot.current_task = None
            robot.state = RobotState.IDLE
    
    def step(self, worker_actions: np.ndarray) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute one timestep in the environment.
        
        Args:
            worker_actions: Array of actions for each robot (shape: [X,])
            
        Returns:
            observations: Dict with 'manager_obs' and 'worker_obs'
            rewards: Dict with 'manager_reward' and 'worker_rewards'
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # Store previous positions for reward calculation
        prev_positions = [(r.row, r.col) for r in self.robots]
        prev_distances = [self._get_distance_to_target(r) for r in self.robots]
        
        # Execute actions and handle collisions
        collisions, deadlocks = self._execute_actions(worker_actions)
        
        # Update metrics
        self.metrics.total_steps += 1
        self.metrics.total_collisions += collisions
        self.metrics.total_deadlocks += deadlocks
        
        # Calculate rewards
        worker_rewards = self._calculate_worker_rewards(
            prev_distances, collisions > 0, deadlocks > 0
        )
        
        # Check if episode is done
        self.done = self.current_step >= self.max_steps
        
        # Get observations
        observations = self._get_observations()
        
        # Calculate Manager reward if at decision point or episode end
        manager_reward = 0.0
        if self.done or self.current_step % self.manager_decision_freq == 0:
            manager_reward = self._calculate_manager_reward()
        
        rewards = {
            "manager_reward": manager_reward,
            "worker_rewards": worker_rewards
        }
        
        info = {
            "step": self.current_step,
            "collisions": collisions,
            "deadlocks": deadlocks,
            "deliveries": sum(r.total_deliveries for r in self.robots),
            "metrics": self.metrics
        }
        
        return observations, rewards, self.done, info
    
    def manager_step(self, action: int) -> bool:
        """
        Execute Manager's action (modify Z or traffic policy).
        
        Args:
            action: Manager action (0=keep, 1=increase, 2=decrease Z)
            
        Returns:
            bool: Whether the action was successfully applied
        """
        old_Z = self.Z
        
        if action == ManagerAction.INCREASE_Z:
            new_Z = self.Z + 1
        elif action == ManagerAction.DECREASE_Z:
            new_Z = max(2, self.Z - 1)
        else:
            return True  # Keep current Z
        
        if new_Z != old_Z:
            # Update layout with new Z
            success = self.layout.update_aisle_width(new_Z)
            if success:
                self.Z = new_Z
                # Reposition robots to valid positions
                self._reposition_robots()
                return True
        
        return True
    
    def _reposition_robots(self):
        """Reposition robots to valid positions after layout change."""
        for robot in self.robots:
            if not self.layout.is_walkable(robot.row, robot.col):
                # Find nearest walkable position
                for dr in range(-5, 6):
                    for dc in range(-5, 6):
                        new_row = robot.row + dr
                        new_col = robot.col + dc
                        if self.layout.is_walkable(new_row, new_col):
                            robot.row = new_row
                            robot.col = new_col
                            break
                    else:
                        continue
                    break
    
    def _execute_actions(self, actions: np.ndarray) -> Tuple[int, int]:
        """
        Execute robot actions with collision detection.
        
        Returns:
            Tuple of (collision_count, deadlock_count)
        """
        # Direction vectors for actions
        directions = {
            RobotAction.UP: (-1, 0),
            RobotAction.DOWN: (1, 0),
            RobotAction.LEFT: (0, -1),
            RobotAction.RIGHT: (0, 1),
            RobotAction.WAIT: (0, 0)
        }
        
        # Calculate intended positions
        intended_positions = []
        for i, robot in enumerate(self.robots):
            action = RobotAction(actions[i])
            dr, dc = directions[action]
            new_row = robot.row + dr
            new_col = robot.col + dc
            
            # Check if new position is valid
            if self.layout.is_walkable(new_row, new_col):
                intended_positions.append((new_row, new_col))
            else:
                # Stay in place if move is invalid
                intended_positions.append((robot.row, robot.col))
        
        # Detect collisions (two robots at same position)
        collision_count = 0
        final_positions = list(intended_positions)
        
        # Check for vertex collisions (same position)
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                if intended_positions[i] == intended_positions[j]:
                    collision_count += 1
                    # Both robots stay in place
                    final_positions[i] = (self.robots[i].row, self.robots[i].col)
                    final_positions[j] = (self.robots[j].row, self.robots[j].col)
                    self.robots[i].collision_count += 1
                    self.robots[j].collision_count += 1
        
        # Check for edge collisions (robots swapping positions)
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                pos_i = (self.robots[i].row, self.robots[i].col)
                pos_j = (self.robots[j].row, self.robots[j].col)
                
                if intended_positions[i] == pos_j and intended_positions[j] == pos_i:
                    collision_count += 1
                    # Both robots stay in place
                    final_positions[i] = pos_i
                    final_positions[j] = pos_j
        
        # Apply movements
        for i, robot in enumerate(self.robots):
            robot.row, robot.col = final_positions[i]
            robot.steps_since_task += 1
            
            # Update position history for deadlock detection
            self.position_history[i].append((robot.row, robot.col))
            if len(self.position_history[i]) > self.deadlock_threshold:
                self.position_history[i].pop(0)
        
        # Detect deadlocks
        deadlock_count = self._detect_deadlocks()
        
        # Check task completion
        self._check_task_completion()
        
        return collision_count, deadlock_count
    
    def _detect_deadlocks(self) -> int:
        """Detect robots stuck in deadlock situations."""
        deadlock_count = 0
        
        for i, robot in enumerate(self.robots):
            history = self.position_history[i]
            
            if len(history) >= self.deadlock_threshold:
                # Check if robot has been in same small area
                unique_positions = set(history)
                if len(unique_positions) <= 3:  # Stuck in 3 or fewer positions
                    deadlock_count += 1
        
        return deadlock_count
    
    def _check_task_completion(self):
        """Check if any robots have completed their tasks."""
        for robot in self.robots:
            if robot.current_task is None:
                self._assign_task(robot)
                continue
            
            current_pos = (robot.row, robot.col)
            
            if robot.state == RobotState.MOVING_TO_PICKUP:
                # Check if reached pickup point
                if current_pos == robot.current_task["pickup_point"]:
                    robot.state = RobotState.PICKING
                    
            elif robot.state == RobotState.PICKING:
                # Instant picking, then move to delivery
                robot.state = RobotState.MOVING_TO_DELIVERY
                
            elif robot.state == RobotState.MOVING_TO_DELIVERY:
                # Check if reached delivery point
                if current_pos == robot.current_task["delivery_point"]:
                    robot.state = RobotState.DELIVERING
                    
            elif robot.state == RobotState.DELIVERING:
                # Complete delivery
                robot.total_deliveries += 1
                self.metrics.total_deliveries += 1
                self.metrics.travel_times.append(robot.steps_since_task)
                
                # Assign new task
                self._assign_task(robot)
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        return {
            "manager_obs": self._get_manager_observation(),
            "worker_obs": self._get_worker_observations()
        }
    
    def _get_manager_observation(self) -> np.ndarray:
        """
        Get compressed state observation for Manager agent.
        
        Manager observation is a compressed vector (NOT full grid):
        - avg_travel_time: Average travel time of completed deliveries
        - robot_density_mean: Mean robot density across grid regions
        - robot_density_std: Std of robot density
        - collision_rate: Collisions per step
        - deadlock_ratio: Deadlocks per step
        - throughput: Deliveries per step
        - current_Z: Normalized aisle width
        """
        # Calculate robot density statistics
        grid_h, grid_w = self.layout.get_dimensions()
        region_size = 5
        densities = []
        
        for r in range(0, grid_h, region_size):
            for c in range(0, grid_w, region_size):
                count = sum(1 for robot in self.robots 
                           if r <= robot.row < r + region_size 
                           and c <= robot.col < c + region_size)
                densities.append(count)
        
        robot_density_mean = np.mean(densities) if densities else 0.0
        robot_density_std = np.std(densities) if densities else 0.0
        
        # Normalize values
        max_Z = 10  # Assumed maximum Z for normalization
        max_travel_time = self.max_steps
        
        obs = np.array([
            self.metrics.avg_travel_time / max_travel_time,  # Normalized
            robot_density_mean / self.X,  # Normalized by max possible
            robot_density_std / self.X,
            self.metrics.collision_rate,  # Already a rate
            self.metrics.deadlock_ratio,  # Already a rate
            self.metrics.throughput * 100,  # Scale up for better learning
            self.Z / max_Z,  # Normalized Z
        ], dtype=np.float32)
        
        return obs
    
    def _get_worker_observations(self) -> np.ndarray:
        """
        Get local observations for all Worker agents.
        
        Each worker observes:
        - Local grid around the robot (local_obs_size x local_obs_size)
        - Direction to target (dx, dy normalized)
        - Distance to target (normalized)
        - Current Z (normalized)
        - Robot state (encoded)
        - Nearby robot positions (up to 4 neighbors)
        """
        observations = []
        
        for robot in self.robots:
            obs = self._get_single_worker_observation(robot)
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
    
    def _get_single_worker_observation(self, robot: Robot) -> np.ndarray:
        """Get observation for a single worker."""
        grid_h, grid_w = self.layout.get_dimensions()
        half_size = self.local_obs_size // 2
        
        # Extract local grid around robot
        local_grid = np.zeros((self.local_obs_size, self.local_obs_size), dtype=np.float32)
        
        for i in range(self.local_obs_size):
            for j in range(self.local_obs_size):
                grid_row = robot.row - half_size + i
                grid_col = robot.col - half_size + j
                
                if 0 <= grid_row < grid_h and 0 <= grid_col < grid_w:
                    cell_value = self.layout.grid[grid_row, grid_col]
                    # Normalize: 0=empty, 0.5=shelf, 1.0=sorting
                    local_grid[i, j] = cell_value / 2.0
                else:
                    local_grid[i, j] = 0.5  # Out of bounds treated as obstacle
        
        # Mark other robots in local grid
        for other in self.robots:
            if other.id != robot.id:
                rel_row = other.row - robot.row + half_size
                rel_col = other.col - robot.col + half_size
                if 0 <= rel_row < self.local_obs_size and 0 <= rel_col < self.local_obs_size:
                    local_grid[rel_row, rel_col] = 1.0  # Mark as obstacle
        
        # Flatten local grid
        grid_flat = local_grid.flatten()
        
        # Calculate direction and distance to target
        if robot.current_task:
            if robot.state == RobotState.MOVING_TO_PICKUP:
                target = robot.current_task["pickup_point"]
            else:
                target = robot.current_task["delivery_point"]
            
            dx = target[1] - robot.col
            dy = target[0] - robot.row
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            
            # Normalize direction
            dx_norm = dx / dist
            dy_norm = dy / dist
            
            # Normalize distance
            max_dist = np.sqrt(grid_h**2 + grid_w**2)
            dist_norm = dist / max_dist
        else:
            dx_norm, dy_norm, dist_norm = 0.0, 0.0, 0.0
        
        # Nearby robot information (positions of up to 4 nearest robots)
        nearby_info = np.zeros(8, dtype=np.float32)
        distances = []
        
        for other in self.robots:
            if other.id != robot.id:
                d = abs(other.row - robot.row) + abs(other.col - robot.col)
                distances.append((d, other))
        
        distances.sort(key=lambda x: x[0])
        
        for i, (_, other) in enumerate(distances[:4]):
            nearby_info[i*2] = (other.row - robot.row) / grid_h
            nearby_info[i*2 + 1] = (other.col - robot.col) / grid_w
        
        # Combine all features
        additional_features = np.array([
            dx_norm,
            dy_norm,
            dist_norm,
            self.Z / 10.0,  # Normalized Z (manager signal)
            robot.state.value / 4.0,  # Normalized state
        ] + list(nearby_info), dtype=np.float32)
        
        return np.concatenate([grid_flat, additional_features])
    
    def _get_distance_to_target(self, robot: Robot) -> float:
        """Get Manhattan distance from robot to its target."""
        if robot.current_task is None:
            return 0.0
        
        if robot.state == RobotState.MOVING_TO_PICKUP:
            target = robot.current_task["pickup_point"]
        else:
            target = robot.current_task["delivery_point"]
        
        return abs(robot.row - target[0]) + abs(robot.col - target[1])
    
    def _calculate_worker_rewards(
        self,
        prev_distances: List[float],
        had_collision: bool,
        had_deadlock: bool
    ) -> np.ndarray:
        """
        Calculate rewards for all worker agents.
        
        Reward shaping:
        - Progress reward: +0.1 * distance_reduction
        - Delivery reward: +10 for successful delivery
        - Step penalty: -0.01 per step
        - Collision penalty: -1.0
        - Deadlock penalty: -5.0
        """
        rewards = np.zeros(self.X, dtype=np.float32)
        
        for i, robot in enumerate(self.robots):
            reward = 0.0
            
            # Progress reward
            current_dist = self._get_distance_to_target(robot)
            progress = prev_distances[i] - current_dist
            reward += 0.1 * progress
            
            # Step penalty
            reward -= 0.01
            
            # Delivery reward (check if just completed)
            if robot.state == RobotState.MOVING_TO_PICKUP and robot.steps_since_task == 1:
                # Just started new task after delivery
                reward += 10.0
            
            # Collision penalty
            if had_collision:
                reward -= 1.0
            
            # Deadlock penalty
            if had_deadlock:
                reward -= 5.0
            
            rewards[i] = reward
        
        return rewards
    
    def _calculate_manager_reward(self) -> float:
        """
        Calculate reward for Manager agent based on episode metrics.
        
        Reward components:
        - Throughput bonus: +100 * throughput
        - Travel time penalty: -0.1 * avg_travel_time
        - Collision penalty: -10 * collision_rate
        - Deadlock penalty: -50 * deadlock_ratio
        """
        reward = 0.0
        
        # Throughput bonus
        reward += 100 * self.metrics.throughput
        
        # Travel time penalty (normalized)
        reward -= 0.1 * self.metrics.avg_travel_time
        
        # Collision penalty
        reward -= 10 * self.metrics.collision_rate
        
        # Deadlock penalty (very strong)
        reward -= 50 * self.metrics.deadlock_ratio
        
        return reward
    
    def get_robot_positions(self) -> List[Tuple[int, int]]:
        """Get current positions of all robots."""
        return [(r.row, r.col) for r in self.robots]
    
    def render(self, mode: str = "text") -> Optional[str]:
        """
        Render the current state of the environment.
        
        Args:
            mode: "text" for ASCII representation
            
        Returns:
            String representation if mode is "text"
        """
        if mode != "text":
            return None
        
        grid_h, grid_w = self.layout.get_dimensions()
        display = np.full((grid_h, grid_w), '.', dtype=str)
        
        # Mark shelves
        for i in range(grid_h):
            for j in range(grid_w):
                if self.layout.grid[i, j] == WarehouseLayout.SHELF:
                    display[i, j] = '#'
                elif self.layout.grid[i, j] == WarehouseLayout.SORTING:
                    display[i, j] = 'S'
        
        # Mark robots
        for robot in self.robots:
            display[robot.row, robot.col] = str(robot.id)
        
        # Convert to string
        lines = [''.join(row) for row in display]
        result = '\n'.join(lines)
        result += f"\n\nStep: {self.current_step}, Z: {self.Z}"
        result += f"\nDeliveries: {self.metrics.total_deliveries}"
        result += f"\nCollisions: {self.metrics.total_collisions}"
        
        return result


# Example usage
if __name__ == "__main__":
    # Create environment
    env = WarehouseEnv(N=4, X=3, Z=3, max_steps=100)
    
    print("Environment created:")
    print(f"Grid size: {env.layout.get_dimensions()}")
    print(f"Number of robots: {env.X}")
    print(f"Worker observation dim: {env.worker_obs_dim}")
    print(f"Manager observation dim: {env.manager_obs_dim}")
    
    # Reset and get initial observations
    obs = env.reset()
    print(f"\nInitial manager obs shape: {obs['manager_obs'].shape}")
    print(f"Initial worker obs shape: {obs['worker_obs'].shape}")
    
    # Run a few random steps
    print("\nRunning random actions...")
    for step in range(10):
        actions = np.random.randint(0, 5, size=env.X)
        obs, rewards, done, info = env.step(actions)
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(env.render())
    
    print("\nFinal metrics:")
    print(f"Deliveries: {info['metrics'].total_deliveries}")
    print(f"Collisions: {info['metrics'].total_collisions}")
