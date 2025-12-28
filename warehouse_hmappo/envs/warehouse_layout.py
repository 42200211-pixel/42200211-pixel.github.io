"""
WarehouseLayout - Generates and manages warehouse grid layout.

This module creates the physical layout of the warehouse based on:
- N: Number of shelves
- Z: Aisle width (must be >= 2)
- X: Number of item types (also determines number of robots and sorting positions)

Layout Structure:
    - Shelves are arranged in rows with aisles of width Z between them
    - Each shelf has 20 slots, each slot holds 50 items of random type
    - Sorting area is located 10 lines below the shelf area
    - X sorting positions correspond to X item types
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Shelf:
    """Represents a shelf in the warehouse."""
    id: int
    row: int  # Row position in grid
    col_start: int  # Starting column
    col_end: int  # Ending column
    slots: np.ndarray  # Shape: (20, 50) - 20 slots, 50 items each


@dataclass
class SortingPosition:
    """Represents a sorting position in the sorting area."""
    id: int
    item_type: int  # Which item type this position handles
    row: int
    col: int


class WarehouseLayout:
    """
    Generates and manages the warehouse layout.
    
    The warehouse consists of:
    1. Shelf area: N shelves arranged with Z-width aisles between them
    2. Sorting area: X positions, 10 lines below the shelf area
    
    Grid Cell Types:
        0: Empty (walkable)
        1: Shelf (non-walkable)
        2: Sorting position (walkable, destination)
        3: Robot (dynamic)
    
    Attributes:
        N (int): Number of shelves
        X (int): Number of item types / robots / sorting positions
        Z (int): Aisle width (>= 2)
        grid (np.ndarray): 2D grid representation of the warehouse
        shelves (List[Shelf]): List of shelf objects
        sorting_positions (List[SortingPosition]): List of sorting positions
    """
    
    # Cell type constants
    EMPTY = 0
    SHELF = 1
    SORTING = 2
    ROBOT = 3
    
    def __init__(self, N: int, X: int, Z: int = 3):
        """
        Initialize warehouse layout.
        
        Args:
            N: Number of shelves
            X: Number of item types (= number of robots = number of sorting positions)
            Z: Aisle width (minimum 2)
        
        Raises:
            ValueError: If Z < 2 or N/X are invalid
        """
        if Z < 2:
            raise ValueError(f"Aisle width Z must be >= 2, got {Z}")
        if N < 1:
            raise ValueError(f"Number of shelves N must be >= 1, got {N}")
        if X < 1:
            raise ValueError(f"Number of item types X must be >= 1, got {X}")
        
        self.N = N
        self.X = X
        self.Z = Z
        
        # Generate layout components
        self.shelves: List[Shelf] = []
        self.sorting_positions: List[SortingPosition] = []
        
        # Calculate grid dimensions and generate layout
        self._calculate_dimensions()
        self._generate_grid()
        self._generate_shelves()
        self._generate_sorting_area()
        
    def _calculate_dimensions(self):
        """Calculate grid dimensions based on warehouse parameters."""
        # Shelf configuration: each shelf spans 20 columns (20 slots)
        self.shelf_width = 20
        
        # Shelves are stacked vertically with aisles between them
        # Each shelf row takes 2 cells (for access from both sides)
        self.shelf_height = 2
        
        # Calculate how many shelf rows we need
        # Arrange shelves in a grid pattern
        self.shelves_per_row = max(1, int(np.sqrt(self.N)))
        self.shelf_rows = int(np.ceil(self.N / self.shelves_per_row))
        
        # Grid width: shelves + aisles between them + border aisles
        self.grid_width = (self.shelf_width * self.shelves_per_row + 
                          self.Z * (self.shelves_per_row + 1))
        
        # Grid height: shelf area + gap + sorting area
        # Shelf area: shelf_rows * (shelf_height + Z)
        self.shelf_area_height = self.shelf_rows * (self.shelf_height + self.Z) + self.Z
        
        # Sorting area is 10 lines below shelf area
        self.gap_to_sorting = 10
        self.sorting_area_height = 3  # Some space for sorting operations
        
        self.grid_height = self.shelf_area_height + self.gap_to_sorting + self.sorting_area_height
        
    def _generate_grid(self):
        """Generate the base grid with all cells empty."""
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        
    def _generate_shelves(self):
        """Generate shelf positions and initialize shelf objects."""
        shelf_id = 0
        
        for row_idx in range(self.shelf_rows):
            for col_idx in range(self.shelves_per_row):
                if shelf_id >= self.N:
                    break
                
                # Calculate shelf position in grid
                # Row position: Z (top aisle) + row_idx * (shelf_height + Z)
                grid_row = self.Z + row_idx * (self.shelf_height + self.Z)
                
                # Column position: Z (left aisle) + col_idx * (shelf_width + Z)
                grid_col_start = self.Z + col_idx * (self.shelf_width + self.Z)
                grid_col_end = grid_col_start + self.shelf_width
                
                # Mark shelf cells in grid
                for r in range(grid_row, grid_row + self.shelf_height):
                    for c in range(grid_col_start, grid_col_end):
                        self.grid[r, c] = self.SHELF
                
                # Create shelf object with item data
                # Each slot has 50 items, randomly assigned types from 0 to X-1
                slots = np.random.randint(0, self.X, size=(20, 50))
                
                shelf = Shelf(
                    id=shelf_id,
                    row=grid_row,
                    col_start=grid_col_start,
                    col_end=grid_col_end,
                    slots=slots
                )
                self.shelves.append(shelf)
                shelf_id += 1
                
    def _generate_sorting_area(self):
        """Generate sorting positions in the sorting area."""
        # Sorting area row
        sorting_row = self.shelf_area_height + self.gap_to_sorting
        
        # Distribute X sorting positions evenly across the width
        spacing = self.grid_width // (self.X + 1)
        
        for i in range(self.X):
            col = spacing * (i + 1)
            col = min(col, self.grid_width - 1)  # Ensure within bounds
            
            # Mark in grid
            self.grid[sorting_row, col] = self.SORTING
            
            # Create sorting position object
            pos = SortingPosition(
                id=i,
                item_type=i,
                row=sorting_row,
                col=col
            )
            self.sorting_positions.append(pos)
            
    def update_aisle_width(self, new_Z: int) -> bool:
        """
        Update aisle width Z and regenerate layout.
        
        This is called by the Manager agent when optimizing Z.
        
        Args:
            new_Z: New aisle width (must be >= 2)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if new_Z < 2:
            return False
        
        # Store old values for potential rollback
        old_Z = self.Z
        
        try:
            self.Z = new_Z
            self.shelves = []
            self.sorting_positions = []
            
            self._calculate_dimensions()
            self._generate_grid()
            self._generate_shelves()
            self._generate_sorting_area()
            
            return True
        except (ValueError, RuntimeError, IndexError) as e:
            # Rollback on specific expected errors during layout generation
            self.Z = old_Z
            self.shelves = []
            self.sorting_positions = []
            self._calculate_dimensions()
            self._generate_grid()
            self._generate_shelves()
            self._generate_sorting_area()
            return False
            
    def get_shelf_access_points(self, shelf_id: int) -> List[Tuple[int, int]]:
        """
        Get accessible points around a shelf where robots can pick items.
        
        Args:
            shelf_id: The shelf ID
            
        Returns:
            List of (row, col) tuples representing access points
        """
        if shelf_id >= len(self.shelves):
            return []
        
        shelf = self.shelves[shelf_id]
        access_points = []
        
        # Access from above the shelf
        row_above = shelf.row - 1
        if row_above >= 0:
            for c in range(shelf.col_start, shelf.col_end):
                if self.grid[row_above, c] == self.EMPTY:
                    access_points.append((row_above, c))
        
        # Access from below the shelf
        row_below = shelf.row + self.shelf_height
        if row_below < self.grid_height:
            for c in range(shelf.col_start, shelf.col_end):
                if self.grid[row_below, c] == self.EMPTY:
                    access_points.append((row_below, c))
        
        return access_points
        
    def get_sorting_position(self, item_type: int) -> Optional[Tuple[int, int]]:
        """
        Get the sorting position for a given item type.
        
        Args:
            item_type: The type of item (0 to X-1)
            
        Returns:
            (row, col) tuple or None if not found
        """
        for pos in self.sorting_positions:
            if pos.item_type == item_type:
                return (pos.row, pos.col)
        return None
    
    def is_walkable(self, row: int, col: int) -> bool:
        """Check if a cell is walkable (not a shelf)."""
        if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
            return self.grid[row, col] != self.SHELF
        return False
    
    def get_random_pickup_task(self) -> Optional[Dict]:
        """
        Generate a random pickup task.
        
        Returns:
            Dictionary with task details or None if no valid task
        """
        if not self.shelves:
            return None
        
        # Select random shelf
        shelf = np.random.choice(self.shelves)
        
        # Select random slot and item
        slot_idx = np.random.randint(0, 20)
        item_idx = np.random.randint(0, 50)
        item_type = shelf.slots[slot_idx, item_idx]
        
        # Get access point for this shelf
        access_points = self.get_shelf_access_points(shelf.id)
        if not access_points:
            return None
        
        pickup_point = access_points[np.random.randint(0, len(access_points))]
        
        # Get sorting position for this item type
        sorting_pos = self.get_sorting_position(int(item_type))
        if sorting_pos is None:
            return None
        
        return {
            "shelf_id": shelf.id,
            "slot_idx": slot_idx,
            "item_idx": item_idx,
            "item_type": int(item_type),
            "pickup_point": pickup_point,
            "delivery_point": sorting_pos
        }
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Return grid dimensions as (height, width)."""
        return (self.grid_height, self.grid_width)
    
    def __repr__(self) -> str:
        return (f"WarehouseLayout(N={self.N}, X={self.X}, Z={self.Z}, "
                f"grid_size={self.grid_height}x{self.grid_width})")


# Example usage and visualization
if __name__ == "__main__":
    # Create a sample warehouse
    layout = WarehouseLayout(N=4, X=3, Z=3)
    print(layout)
    print(f"Grid shape: {layout.grid.shape}")
    print(f"Number of shelves: {len(layout.shelves)}")
    print(f"Number of sorting positions: {len(layout.sorting_positions)}")
    
    # Visualize grid
    print("\nWarehouse Grid (0=empty, 1=shelf, 2=sorting):")
    print(layout.grid)
