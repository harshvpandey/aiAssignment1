"""
Coding Assignment-1: Artificial Intelligence
Dynamic Goal-Based Agent for Warehouse Logistics Optimization
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
from collections import deque
import time

class WarehouseEnvironment:
    """
    Represents the warehouse grid environment with packages, drop-off points, and obstacles.
    This class handles the representation of the warehouse as a grid.
    """
    # Define constants for cell types
    EMPTY = 0
    OBSTACLE = 1
    ROBOT = 2
    PACKAGE = 3
    DROPOFF = 4
    ROBOT_ON_DROPOFF = 5
    ROBOT_WITH_PACKAGE = 6
    
    # Display characters for visualization in console
    DISPLAY_SYMBOLS = {
        EMPTY: '.',
        OBSTACLE: '█',
        ROBOT: 'R',
        PACKAGE: 'P',
        DROPOFF: 'D',
        ROBOT_ON_DROPOFF: '*',
        ROBOT_WITH_PACKAGE: 'C'  # 'C' for carrying
    }
    
    # Colors for matplotlib visualization
    CELL_COLORS = {
        EMPTY: 'white',
        OBSTACLE: 'black',
        ROBOT: 'blue',
        PACKAGE: 'green',
        DROPOFF: 'red',
        ROBOT_ON_DROPOFF: 'purple',
        ROBOT_WITH_PACKAGE: 'orange'
    }
    
    def __init__(self, n, m, num_packages, num_obstacles, seed_value=42):
        """
        Initialize the warehouse environment with given dimensions and parameters.
        
        Args:
            n (int): Number of rows in the grid
            m (int): Number of columns in the grid
            num_packages (int): Number of packages to place
            num_obstacles (int): Number of obstacles to place
            seed_value (int): Random seed for reproducibility
        """
        # Set random seed for reproducibility
        self.seed = seed_value
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize grid parameters
        self.n = n  # Number of rows
        self.m = m  # Number of columns
        self.num_packages = num_packages
        self.num_obstacles = num_obstacles
        
        # Create empty grid
        self.grid = np.zeros((n, m), dtype=int)
        
        # Initialize positions
        self.robot_pos = None
        self.package_positions = []
        self.dropoff_positions = []
        self.obstacle_positions = []
        
        # Setup the environment
        self._setup_environment()
    
    def _setup_environment(self):
        """
        Set up the warehouse environment by placing the robot, packages,
        drop-off points, and obstacles randomly on the grid.
        """
        # Start with all positions as empty
        occupied_positions = set()
        
        # Place the robot at a fixed position (loading dock) - top-left corner
        self.robot_pos = (0, 0)
        self.grid[self.robot_pos] = self.ROBOT
        occupied_positions.add(self.robot_pos)
        
        # Place packages randomly
        for _ in range(self.num_packages):
            while True:
                pos = (random.randint(0, self.n - 1), random.randint(0, self.m - 1))
                if pos not in occupied_positions:
                    self.package_positions.append(pos)
                    occupied_positions.add(pos)
                    self.grid[pos] = self.PACKAGE
                    break
        
        # Place drop-off points randomly
        for _ in range(self.num_packages):
            while True:
                pos = (random.randint(0, self.n - 1), random.randint(0, self.m - 1))
                if pos not in occupied_positions:
                    self.dropoff_positions.append(pos)
                    occupied_positions.add(pos)
                    self.grid[pos] = self.DROPOFF
                    break
        
        # Place obstacles randomly
        for _ in range(self.num_obstacles):
            while True:
                pos = (random.randint(0, self.n - 1), random.randint(0, self.m - 1))
                if pos not in occupied_positions:
                    self.obstacle_positions.append(pos)
                    occupied_positions.add(pos)
                    self.grid[pos] = self.OBSTACLE
                    break
    
    def display_grid(self):
        """Display the current state of the warehouse grid in the console."""
        print(f"Warehouse Grid ({self.n}x{self.m}):")
        for i in range(self.n):
            row = []
            for j in range(self.m):
                cell_type = self.grid[i, j]
                row.append(self.DISPLAY_SYMBOLS[cell_type])
            print(' '.join(row))
    
    def visualize_grid(self, title="Warehouse Configuration"):
        """
        Visualize the warehouse grid using matplotlib.
        
        Args:
            title (str): Title of the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create a custom colormap for the grid cells
        colors = []
        for i in range(7):  # Number of cell types
            if i == self.EMPTY:
                colors.append([1, 1, 1])  # white
            elif i == self.OBSTACLE:
                colors.append([0, 0, 0])  # black
            elif i == self.ROBOT:
                colors.append([0, 0, 1])  # blue
            elif i == self.PACKAGE:
                colors.append([0, 0.8, 0])  # green
            elif i == self.DROPOFF:
                colors.append([1, 0, 0])  # red
            elif i == self.ROBOT_ON_DROPOFF:
                colors.append([0.5, 0, 0.5])  # purple
            elif i == self.ROBOT_WITH_PACKAGE:
                colors.append([1, 0.5, 0])  # orange
        
        # Create a grid image with proper colors
        plt.imshow(self.grid, cmap=plt.cm.colors.ListedColormap(colors))
        
        # Add cell labels
        for i in range(self.n):
            for j in range(self.m):
                cell_type = self.grid[i, j]
                plt.text(j, i, self.DISPLAY_SYMBOLS[cell_type], 
                         ha='center', va='center', 
                         color='white' if cell_type in [self.OBSTACLE, self.ROBOT, self.ROBOT_WITH_PACKAGE, self.ROBOT_ON_DROPOFF] else 'black')
        
        # Add legend elements
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[self.EMPTY], label='Empty'),
            plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[self.OBSTACLE], label='Obstacle'),
            plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[self.ROBOT], label='Robot'),
            plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[self.PACKAGE], label='Package'),
            plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[self.DROPOFF], label='Drop-off'),
            plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[self.ROBOT_ON_DROPOFF], label='Robot on Drop-off'),
            plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[self.ROBOT_WITH_PACKAGE], label='Robot with Package'),
        ]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def update_robot_position(self, new_pos, carrying_package=False):
        """
        Update the robot's position on the grid.
        
        Args:
            new_pos (tuple): New position (row, col) of the robot
            carrying_package (bool): Whether the robot is carrying a package
        """
        # Clear the old position
        row, col = self.robot_pos
        if self.grid[row, col] == self.ROBOT_ON_DROPOFF:
            self.grid[row, col] = self.DROPOFF
        elif self.grid[row, col] == self.ROBOT or self.grid[row, col] == self.ROBOT_WITH_PACKAGE:
            self.grid[row, col] = self.EMPTY
        
        # Update to the new position
        self.robot_pos = new_pos
        row, col = new_pos
        
        # Set the cell type based on whether robot is on a drop-off and/or carrying a package
        if self.grid[row, col] == self.DROPOFF:
            self.grid[row, col] = self.ROBOT_ON_DROPOFF
        else:
            self.grid[row, col] = self.ROBOT_WITH_PACKAGE if carrying_package else self.ROBOT


class WarehouseAgent:
    """
    A goal-based agent for warehouse logistics optimization. This agent can plan
    and execute paths to pick up packages and deliver them to drop-off points.
    """
    
    def __init__(self, warehouse):
        """
        Initialize the agent with a warehouse environment.
        
        Args:
            warehouse (WarehouseEnvironment): The warehouse environment
        """
        self.warehouse = warehouse
        self.current_pos = warehouse.robot_pos
        self.carrying_package = False
        self.current_package_idx = None
        
        # Copy the initial package and dropoff positions
        self.remaining_packages = warehouse.package_positions.copy()
        self.dropoff_positions = warehouse.dropoff_positions.copy()
        
        # For tracking performance
        self.movement_cost = 0
        self.delivery_reward = 0
        self.obstacle_penalty = 0
        self.packages_delivered = 0
        
        # For storing the path history
        self.path_history = []
    
    def find_path(self, start, goal, algorithm="BFS"):
        """
        Find a path from start to goal using the specified search algorithm.
        
        Args:
            start (tuple): Starting position (row, col)
            goal (tuple): Goal position (row, col)
            algorithm (str): Search algorithm to use: 'BFS', 'DFS', or 'UCS'
        
        Returns:
            list: List of positions forming the path, or None if no path found
        """
        if algorithm == "BFS":
            return self._breadth_first_search(start, goal)
        elif algorithm == "DFS":
            return self._depth_first_search(start, goal)
        elif algorithm == "UCS":
            return self._uniform_cost_search(start, goal)
        else:
            raise ValueError(f"Unknown search algorithm: {algorithm}")
    
    def _breadth_first_search(self, start, goal):
        """
        Find the shortest path using Breadth-First Search.
        
        Args:
            start (tuple): Starting position (row, col)
            goal (tuple): Goal position (row, col)
        
        Returns:
            list: List of positions forming the path, or None if no path found
        """
        # Initialize queue with start position and empty path
        queue = deque([(start, [])])
        # Keep track of visited positions
        visited = {start}
        
        # Direction vectors: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while queue:
            (row, col), path = queue.popleft()
            
            # Check if reached goal
            if (row, col) == goal:
                return path + [(row, col)]
            
            # Try each direction
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                new_pos = (new_row, new_col)
                
                # Check if the move is valid
                if (0 <= new_row < self.warehouse.n and 
                    0 <= new_col < self.warehouse.m and 
                    self.warehouse.grid[new_row, new_col] != self.warehouse.OBSTACLE and
                    new_pos not in visited):
                    
                    visited.add(new_pos)
                    queue.append((new_pos, path + [(row, col)]))
        
        # No path found
        return None
    
    def _depth_first_search(self, start, goal):
        """
        Find a path using Depth-First Search.
        
        Args:
            start (tuple): Starting position (row, col)
            goal (tuple): Goal position (row, col)
        
        Returns:
            list: List of positions forming the path, or None if no path found
        """
        # Initialize stack with start position and its path
        stack = [(start, [start])]
        # Keep track of visited positions
        visited = set()
        
        # Direction vectors: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while stack:
            (row, col), path = stack.pop()
            
            # Check if reached goal
            if (row, col) == goal:
                return path
            
            # If not visited
            if (row, col) not in visited:
                visited.add((row, col))
                
                # Try each direction
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    new_pos = (new_row, new_col)
                    
                    # Check if the move is valid
                    if (0 <= new_row < self.warehouse.n and 
                        0 <= new_col < self.warehouse.m and 
                        self.warehouse.grid[new_row, new_col] != self.warehouse.OBSTACLE and
                        new_pos not in visited):
                        
                        stack.append((new_pos, path + [new_pos]))
        
        # No path found
        return None
    
    def _uniform_cost_search(self, start, goal):
        """
        Find the optimal path using Uniform Cost Search.
        
        Args:
            start (tuple): Starting position (row, col)
            goal (tuple): Goal position (row, col)
        
        Returns:
            list: List of positions forming the path, or None if no path found
        """
        # Priority queue: (cost, position, path)
        pq = [(0, start, [])]
        # Keep track of visited positions and their costs
        visited = {}
        
        # Direction vectors: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while pq:
            cost, (row, col), path = heapq.heappop(pq)
            
            # Check if reached goal
            if (row, col) == goal:
                return path + [(row, col)]
            
            # If position not visited or found a better path
            if (row, col) not in visited or cost < visited[(row, col)]:
                visited[(row, col)] = cost
                
                # Try each direction
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    new_pos = (new_row, new_col)
                    
                    # Check if the move is valid
                    if (0 <= new_row < self.warehouse.n and 
                        0 <= new_col < self.warehouse.m and 
                        self.warehouse.grid[new_row, new_col] != self.warehouse.OBSTACLE):
                        
                        # Each move costs 1 unit
                        new_cost = cost + 1
                        
                        heapq.heappush(pq, (new_cost, new_pos, path + [(row, col)]))
        
        # No path found
        return None
    
    def move_along_path(self, path):
        """
        Move the robot along the given path.
        
        Args:
            path (list): List of positions forming the path
        
        Returns:
            int: Number of movements made
        """
        if not path:
            return 0
        
        movements = 0
        
        # Skip the first position as it's the current position
        for position in path[1:]:
            # Update the robot's position in the environment
            self.warehouse.update_robot_position(position, self.carrying_package)
            
            # Update the agent's current position
            self.current_pos = position
            
            # Check if hit an obstacle
            if position in self.warehouse.obstacle_positions:
                self.obstacle_penalty -= 5
            
            # Record movement
            movements += 1
            self.movement_cost += 1
            self.path_history.append(position)
        
        return movements
    
    def pickup_package(self):
        """
        Attempt to pick up a package at the current position.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Can only pick up if at a package location and not already carrying
        if not self.carrying_package and self.current_pos in self.remaining_packages:
            # Find the package index
            package_idx = self.remaining_packages.index(self.current_pos)
            
            # Update state
            self.carrying_package = True
            self.current_package_idx = package_idx
            
            # Update warehouse grid
            self.warehouse.update_robot_position(self.current_pos, True)
            
            # Remove from remaining packages
            self.remaining_packages.remove(self.current_pos)
            
            return True
        
        return False
    
    def deliver_package(self):
        """
        Attempt to deliver the package at the current position.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Can only deliver if carrying a package and at the correct dropoff location
        if self.carrying_package:
            # Check if at the correct dropoff location
            correct_dropoff = self.dropoff_positions[self.current_package_idx]
            if self.current_pos == correct_dropoff:
                # Update state
                self.carrying_package = False
                self.packages_delivered += 1
                self.delivery_reward += 10
                
                # Update warehouse grid
                self.warehouse.update_robot_position(self.current_pos, False)
                
                # Mark this dropoff as used
                self.dropoff_positions[self.current_package_idx] = None
                
                return True
        
        return False
    
    def plan_and_execute_deliveries(self, search_algorithm="BFS"):
        """
        Plan and execute the optimal sequence to deliver all packages.
        
        Args:
            search_algorithm (str): Search algorithm to use
            
        Returns:
            dict: Performance metrics and path history
        """
        print(f"\nExecuting deliveries using {search_algorithm} algorithm...")
        start_time = time.time()
        
        # Continue until all packages are delivered
        while self.remaining_packages:
            if not self.carrying_package:
                # Find the closest package
                closest_package = None
                shortest_path = None
                
                for package in self.remaining_packages:
                    path = self.find_path(self.current_pos, package, search_algorithm)
                    if path and (shortest_path is None or len(path) < len(shortest_path)):
                        shortest_path = path
                        closest_package = package
                
                if not shortest_path:
                    print("Error: No path to any package found.")
                    break
                
                # Move to the package
                print(f"Moving to package at {closest_package}...")
                self.move_along_path(shortest_path)
                
                # Pick up the package
                success = self.pickup_package()
                print(f"Picked up package: {success}")
            
            else:
                # Find the path to the correct dropoff
                dropoff = self.dropoff_positions[self.current_package_idx]
                path = self.find_path(self.current_pos, dropoff, search_algorithm)
                
                if not path:
                    print(f"Error: No path to dropoff point {dropoff} found.")
                    break
                
                # Move to the dropoff
                print(f"Moving to dropoff at {dropoff}...")
                self.move_along_path(path)
                
                # Deliver the package
                success = self.deliver_package()
                print(f"Delivered package: {success}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate final score
        final_score = self.delivery_reward - self.movement_cost + self.obstacle_penalty
        
        return {
            "path_taken": self.path_history,
            "movements": self.movement_cost,
            "rewards": self.delivery_reward,
            "penalties": self.obstacle_penalty,
            "packages_delivered": self.packages_delivered,
            "final_score": final_score,
            "execution_time": execution_time
        }


def q1_setup_warehouse():
    """
    Q1: Represent the warehouse as an N×M matrix. Place the packages,
    drop-off points, and obstacles randomly. Display the initial warehouse configuration.
    """
    # Set parameters within the specified ranges
    n = 8  # Number of rows (between 5 and 10)
    m = 8  # Number of columns (between 5 and 10)
    num_packages = 3  # Number of packages (between 2 and 6)
    num_obstacles = 5  # Number of obstacles (between 1 and 10)
    seed = 42  # Random seed for reproducibility
    
    # Create the warehouse environment
    warehouse = WarehouseEnvironment(n, m, num_packages, num_obstacles, seed)
    
    # Display information about the warehouse configuration
    print(f"Warehouse Configuration ({n}×{m}):")
    print(f"Number of packages: {num_packages}")
    print(f"Number of obstacles: {num_obstacles}")
    print(f"Robot starting position: {warehouse.robot_pos}")
    print(f"Package positions: {warehouse.package_positions}")
    print(f"Drop-off positions: {warehouse.dropoff_positions}")
    print(f"Obstacle positions: {warehouse.obstacle_positions}")
    
    # Display the initial grid
    print("\nInitial warehouse grid:")
    warehouse.display_grid()
    
    # Visualize the grid
    warehouse.visualize_grid("Initial Warehouse Configuration")
    
    return warehouse


def q2_implement_agent(warehouse, search_algorithm="BFS"):
    """
    Q2: Implement a goal-based agent that can identify all goals, plan a sequence of actions,
    use a search algorithm to find optimal paths, deliver all packages, and calculate the total cost.
    """
    # Create the agent
    agent = WarehouseAgent(warehouse)
    
    # Plan and execute the deliveries
    results = agent.plan_and_execute_deliveries(search_algorithm)
    
    # Display the final state of the warehouse
    print("\nFinal warehouse state:")
    warehouse.display_grid()
    
    return results


def q3_evaluate_performance(warehouse, algorithms=None):
    """
    Q3: Choose a random seed value for reproducibility. Show the chosen path, total cost and rewards,
    and final score based on penalties, movement costs, and successful deliveries.
    """
    if algorithms is None:
        algorithms = ["BFS", "DFS", "UCS"]
    
    results = {}
    
    for algorithm in algorithms:
        # Create a fresh copy of the warehouse for each algorithm
        warehouse_copy = WarehouseEnvironment(
            warehouse.n, warehouse.m, 
            warehouse.num_packages, warehouse.num_obstacles,
            warehouse.seed
        )
        
        # Run the agent with this algorithm
        print(f"\n===== Running with {algorithm} algorithm =====")
        result = q2_implement_agent(warehouse_copy, algorithm)
        
        # Store the results
        results[algorithm] = result
        
        # Display the results
        print(f"\nResults for {algorithm}:")
        print(f"Total movements: {result['movements']}")
        print(f"Total rewards: {result['rewards']}")
        print(f"Total penalties: {result['penalties']}")
        print(f"Packages delivered: {result['packages_delivered']}")
        print(f"Final score: {result['final_score']}")
        print(f"Execution time: {result['execution_time']:.4f} seconds")
        
        # Visualize the final state
        warehouse_copy.visualize_grid(f"Final Warehouse State ({algorithm})")
    
    # Compare the algorithms
    print("\n===== Algorithm Comparison =====")
    print(f"{'Algorithm':<10} | {'Score':<10} | {'Movements':<10} | {'Time (s)':<10}")
    print("-" * 45)
    for algorithm, result in results.items():
        print(f"{algorithm:<10} | {result['final_score']:<10} | {result['movements']:<10} | {result['execution_time']:<10.4f}")
    
    return results


def main():
    """Main function to run the warehouse logistics optimization."""
    print("=== Dynamic Goal-Based Agent for Warehouse Logistics Optimization ===\n")
    
    # Q1: Setup the warehouse environment
    warehouse = q1_setup_warehouse()
    
    # Q2 & Q3: Run the agent with different search algorithms and evaluate performance
    algorithms = ["BFS", "DFS", "UCS"]
    q3_evaluate_performance(warehouse, algorithms)


if __name__ == "__main__":
    main()