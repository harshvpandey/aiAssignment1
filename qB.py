"""
Problem B: Finding Optimal Meetup Place using Search Algorithms

This solution implements a system to find the optimal meetup point for two friends
located in different cities in India (Mumbai and Prayagraj) using various search algorithms.
"""

import heapq
import time
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class City:
    """Represents a city/taluka with its location and connections"""
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.neighbors = []  # List of (neighbor_city, distance) tuples
    
    def add_neighbor(self, neighbor, distance):
        """Add a neighboring city with distance"""
        self.neighbors.append((neighbor, distance))
    
    def __str__(self):
        return f"{self.name} ({self.latitude}, {self.longitude})"
    
    def __repr__(self):
        return self.__str__()


class IndiaMap:
    """Represents a map of Indian cities/talukas with connections"""
    def __init__(self):
        self.cities = {}  # name -> City object
        
    def add_city(self, name, latitude, longitude):
        """Add a city to the map"""
        self.cities[name] = City(name, latitude, longitude)
        return self.cities[name]
    
    def add_connection(self, city1_name, city2_name, distance=None):
        """Add a bidirectional connection between two cities"""
        city1 = self.cities[city1_name]
        city2 = self.cities[city2_name]
        
        # Calculate straight-line distance if not provided
        if distance is None:
            distance = self.calculate_distance(city1, city2)
        
        # Add bidirectional connections
        city1.add_neighbor(city2, distance)
        city2.add_neighbor(city1, distance)
    
    def get_city(self, name):
        """Get a city by name"""
        return self.cities.get(name)
    
    @staticmethod
    def calculate_distance(city1, city2):
        """Calculate the Haversine distance between two cities in kilometers"""
        # Earth's radius in kilometers
        R = 6371.0
        
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(city1.latitude)
        lon1 = math.radians(city1.longitude)
        lat2 = math.radians(city2.latitude)
        lon2 = math.radians(city2.longitude)
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance
    
    def visualize(self, path=None):
        """Visualize the map and optionally a path using NetworkX and Matplotlib"""
        G = nx.Graph()
        
        # Add nodes
        for name, city in self.cities.items():
            G.add_node(name, pos=(city.longitude, city.latitude))
        
        # Add edges
        for name, city in self.cities.items():
            for neighbor, distance in city.neighbors:
                G.add_edge(name, neighbor.name, weight=distance)
        
        # Get positions for plotting
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Draw the path if provided
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='red')
            
            # Highlight start and end cities
            start_nodes = [path[0]]
            end_nodes = [path[-1]]
            nx.draw_networkx_nodes(G, pos, nodelist=start_nodes, node_size=200, node_color='green')
            nx.draw_networkx_nodes(G, pos, nodelist=end_nodes, node_size=200, node_color='red')
        
        plt.title("Map of India with Cities and Connections")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class MeetupProblem:
    """
    Represents the problem of finding an optimal meetup point
    for two friends located in different cities.
    """
    def __init__(self, map_obj, city1_name, city2_name):
        self.map = map_obj
        self.city1 = self.map.get_city(city1_name)
        self.city2 = self.map.get_city(city2_name)
        
        if not self.city1:
            raise ValueError(f"City not found: {city1_name}")
        if not self.city2:
            raise ValueError(f"City not found: {city2_name}")
            
        print(f"Initialized meetup problem with friends in {self.city1.name} and {self.city2.name}")
    
    def straight_line_heuristic(self, state1, state2):
        """
        Heuristic function based on straight-line distance between
        the current positions of the two friends.
        """
        city1 = self.map.get_city(state1)
        city2 = self.map.get_city(state2)
        return self.map.calculate_distance(city1, city2)
    
    def road_distance_heuristic(self, state1, state2):
        """
        Heuristic function based on road distance between
        the current positions of the two friends.
        This is a more realistic but potentially less admissible heuristic.
        """
        # Since we don't have actual road distances, we'll multiply
        # the straight-line distance by a factor to approximate it
        return self.straight_line_heuristic(state1, state2) * 1.3
    
    def train_distance_heuristic(self, state1, state2):
        """
        Heuristic function based on train route distance between
        the current positions of the two friends.
        This takes into account that trains don't go in straight lines.
        """
        # Again, since we don't have actual train distances, we'll approximate
        return self.straight_line_heuristic(state1, state2) * 1.5
    
    def greedy_best_first_search(self, heuristic_func):
        """
        Greedy Best-First Search to find the optimal meetup point.
        
        Args:
            heuristic_func: Function that calculates heuristic between two cities
            
        Returns:
            tuple: (meetup_city, path1, path2, metrics)
        """
        start_time = time.time()
        nodes_generated = 0
        
        # Priority queue: (heuristic, (your_city, friend_city), (your_path, friend_path))
        frontier = [(heuristic_func(self.city1.name, self.city2.name),
                    (self.city1.name, self.city2.name),
                    ([self.city1.name], [self.city2.name]))]
        
        # Keep track of visited states to avoid cycles
        visited = set([(self.city1.name, self.city2.name)])
        
        while frontier:
            _, (your_city, friend_city), (your_path, friend_path) = heapq.heappop(frontier)
            
            # Check if meetup achieved
            if your_city == friend_city:
                end_time = time.time()
                return (your_city, your_path, friend_path, {
                    'nodes_generated': nodes_generated,
                    'time_taken': end_time - start_time,
                    'path_length_you': len(your_path),
                    'path_length_friend': len(friend_path)
                })
            
            # Generate successors
            for your_next, your_dist in self.map.get_city(your_city).neighbors:
                for friend_next, friend_dist in self.map.get_city(friend_city).neighbors:
                    next_state = (your_next.name, friend_next.name)
                    
                    if next_state not in visited:
                        visited.add(next_state)
                        nodes_generated += 1
                        
                        # Time to move is the maximum of the two distances
                        # (one person has to wait for the other)
                        move_time = max(your_dist * 2, friend_dist * 2)
                        
                        # Calculate heuristic for the next state
                        h = heuristic_func(your_next.name, friend_next.name)
                        
                        # Add to frontier with priority based on heuristic only (greedy)
                        heapq.heappush(frontier, 
                                     (h, 
                                      next_state, 
                                      (your_path + [your_next.name], 
                                       friend_path + [friend_next.name])))
        
        # No solution found
        end_time = time.time()
        return (None, None, None, {
            'nodes_generated': nodes_generated,
            'time_taken': end_time - start_time
        })
    
    def a_star_search(self, heuristic_func):
        """
        A* Search to find the optimal meetup point.
        
        Args:
            heuristic_func: Function that calculates heuristic between two cities
            
        Returns:
            tuple: (meetup_city, path1, path2, metrics)
        """
        start_time = time.time()
        nodes_generated = 0
        
        # Priority queue: (f=g+h, (your_city, friend_city), g, (your_path, friend_path))
        # g is the cost so far
        h = heuristic_func(self.city1.name, self.city2.name)
        frontier = [(h, (self.city1.name, self.city2.name), 0, 
                     ([self.city1.name], [self.city2.name]))]
        
        # Keep track of visited states and their costs
        visited = {}
        visited[(self.city1.name, self.city2.name)] = 0
        
        while frontier:
            _, (your_city, friend_city), g, (your_path, friend_path) = heapq.heappop(frontier)
            
            # Check if meetup achieved
            if your_city == friend_city:
                end_time = time.time()
                return (your_city, your_path, friend_path, {
                    'nodes_generated': nodes_generated,
                    'time_taken': end_time - start_time,
                    'path_length_you': len(your_path),
                    'path_length_friend': len(friend_path),
                    'total_cost': g
                })
            
            # If we've found a better path to this state, skip
            if g > visited.get((your_city, friend_city), float('inf')):
                continue
                
            # Generate successors
            for your_next, your_dist in self.map.get_city(your_city).neighbors:
                for friend_next, friend_dist in self.map.get_city(friend_city).neighbors:
                    next_state = (your_next.name, friend_next.name)
                    
                    # Time to move is the maximum of the two distances
                    # (one person has to wait for the other)
                    move_cost = max(your_dist * 2, friend_dist * 2)
                    new_g = g + move_cost
                    
                    # If we found a better path to this state
                    if new_g < visited.get(next_state, float('inf')):
                        visited[next_state] = new_g
                        nodes_generated += 1
                        
                        # Calculate heuristic for the next state
                        h = heuristic_func(your_next.name, friend_next.name)
                        
                        # Add to frontier with priority based on f = g + h
                        heapq.heappush(frontier, 
                                     (new_g + h, 
                                      next_state, 
                                      new_g, 
                                      (your_path + [your_next.name], 
                                       friend_path + [friend_next.name])))
        
        # No solution found
        end_time = time.time()
        return (None, None, None, {
            'nodes_generated': nodes_generated,
            'time_taken': end_time - start_time
        })


def create_india_map():
    """
    Create a map of India with major cities/talukas and their connections.
    Focuses on Mumbai to Prayagraj (UP) with cities in between.
    """
    india = IndiaMap()
    
    # Add cities with their coordinates (latitude, longitude)
    # Starting with Mumbai and Prayagraj (Allahabad)
    mumbai = india.add_city("Mumbai", 19.0760, 72.8777)
    prayagraj = india.add_city("Prayagraj", 25.4358, 81.8463)
    
    # Add important cities between Mumbai and Prayagraj
    # Maharashtra
    nashik = india.add_city("Nashik", 19.9975, 73.7898)
    aurangabad = india.add_city("Aurangabad", 19.8762, 75.3433)
    nagpur = india.add_city("Nagpur", 21.1458, 79.0882)
    
    # Madhya Pradesh
    jabalpur = india.add_city("Jabalpur", 23.1815, 79.9864)
    bhopal = india.add_city("Bhopal", 23.2599, 77.4126)
    indore = india.add_city("Indore", 22.7196, 75.8577)
    
    # Uttar Pradesh
    jhansi = india.add_city("Jhansi", 25.4484, 78.5685)
    kanpur = india.add_city("Kanpur", 26.4499, 80.3319)
    lucknow = india.add_city("Lucknow", 26.8467, 80.9462)
    varanasi = india.add_city("Varanasi", 25.3176, 82.9739)
    
    # Additional cities to create a more connected network
    pune = india.add_city("Pune", 18.5204, 73.8567)
    surat = india.add_city("Surat", 21.1702, 72.8311)
    ahmedabad = india.add_city("Ahmedabad", 23.0225, 72.5714)
    ujjain = india.add_city("Ujjain", 23.1765, 75.7885)
    gwalior = india.add_city("Gwalior", 26.2183, 78.1828)
    
    # Add connections between cities
    # Maharashtra connections
    india.add_connection("Mumbai", "Pune")
    india.add_connection("Mumbai", "Nashik")
    india.add_connection("Mumbai", "Surat")
    india.add_connection("Pune", "Nashik")
    india.add_connection("Pune", "Aurangabad")
    india.add_connection("Nashik", "Aurangabad")
    india.add_connection("Aurangabad", "Nagpur")
    
    # Gujarat connections
    india.add_connection("Surat", "Ahmedabad")
    india.add_connection("Ahmedabad", "Indore")
    
    # Madhya Pradesh connections
    india.add_connection("Indore", "Ujjain")
    india.add_connection("Indore", "Bhopal")
    india.add_connection("Ujjain", "Bhopal")
    india.add_connection("Bhopal", "Jabalpur")
    india.add_connection("Nagpur", "Jabalpur")
    india.add_connection("Bhopal", "Gwalior")
    
    # Uttar Pradesh connections
    india.add_connection("Gwalior", "Jhansi")
    india.add_connection("Jhansi", "Kanpur")
    india.add_connection("Kanpur", "Lucknow")
    india.add_connection("Kanpur", "Prayagraj")
    india.add_connection("Lucknow", "Prayagraj")
    india.add_connection("Prayagraj", "Varanasi")
    india.add_connection("Jabalpur", "Prayagraj")
    
    # Add a few more connections to create alternative routes
    india.add_connection("Nagpur", "Bhopal")
    india.add_connection("Jabalpur", "Jhansi")
    india.add_connection("Indore", "Nagpur")
    india.add_connection("Ahmedabad", "Ujjain")
    india.add_connection("Gwalior", "Kanpur")
    
    return india


def main():
    # Create the map of India
    print("Creating map of India...")
    india_map = create_india_map()
    
    # Visualize the map
    print("Visualizing the map of India...")
    india_map.visualize()
    
    # Q1: Formulate the search problem
    print("\nQ1: Formulating the search problem...")
    print("You are in Mumbai and your friend is in Prayagraj (UP).")
    print("You both want to meet at a common location as quickly as possible.")
    print("The amount of time needed to move between cities is equal to the straight line distance × 2.")
    print("On each turn, the friend that arrives first must wait until the other one arrives.")
    
    # Create the meetup problem
    meetup = MeetupProblem(india_map, "Mumbai", "Prayagraj")
    
    # Q2: Implement search strategies
    print("\nQ2: Implementing search strategies...")
    
    # Greedy Best-First Search
    print("\nRunning Greedy Best-First Search...")
    gbfs_meetup, gbfs_path_you, gbfs_path_friend, gbfs_metrics = meetup.greedy_best_first_search(
        meetup.straight_line_heuristic
    )
    
    if gbfs_meetup:
        print(f"Meetup point found: {gbfs_meetup}")
        print(f"Your path: {' -> '.join(gbfs_path_you)}")
        print(f"Friend's path: {' -> '.join(gbfs_path_friend)}")
        print(f"Metrics: {gbfs_metrics}")
        
        # Visualize the paths
        combined_path = list(set(gbfs_path_you + gbfs_path_friend))
        india_map.visualize(combined_path)
    else:
        print("No meetup point found using Greedy Best-First Search.")
    
    # A* Search
    print("\nRunning A* Search...")
    astar_meetup, astar_path_you, astar_path_friend, astar_metrics = meetup.a_star_search(
        meetup.straight_line_heuristic
    )
    
    if astar_meetup:
        print(f"Meetup point found: {astar_meetup}")
        print(f"Your path: {' -> '.join(astar_path_you)}")
        print(f"Friend's path: {' -> '.join(astar_path_friend)}")
        print(f"Metrics: {astar_metrics}")
        
        # Visualize the paths
        combined_path = list(set(astar_path_you + astar_path_friend))
        india_map.visualize(combined_path)
    else:
        print("No meetup point found using A* Search.")
    
    # Q3: Change the heuristic function
    print("\nQ3: Changing the heuristic function...")
    
    # Use road distance heuristic
    print("\nRunning A* Search with road distance heuristic...")
    road_meetup, road_path_you, road_path_friend, road_metrics = meetup.a_star_search(
        meetup.road_distance_heuristic
    )
    
    if road_meetup:
        print(f"Meetup point found: {road_meetup}")
        print(f"Metrics: {road_metrics}")
    
    # Use train distance heuristic
    print("\nRunning A* Search with train distance heuristic...")
    train_meetup, train_path_you, train_path_friend, train_metrics = meetup.a_star_search(
        meetup.train_distance_heuristic
    )
    
    if train_meetup:
        print(f"Meetup point found: {train_meetup}")
        print(f"Metrics: {train_metrics}")
    
    # Compare results
    print("\nComparison of different heuristics:")
    print(f"{'Heuristic':<20} | {'Meetup Point':<15} | {'Nodes Generated':<15} | {'Time (ms)':<10} | {'Total Cost':<10}")
    print("-" * 80)
    
    # Straight-line distance
    print(f"{'Straight-line':<20} | {astar_meetup:<15} | {astar_metrics['nodes_generated']:<15} | {astar_metrics['time_taken']*1000:.2f} | {astar_metrics.get('total_cost', 'N/A')}")
    
    # Road distance
    print(f"{'Road distance':<20} | {road_meetup:<15} | {road_metrics['nodes_generated']:<15} | {road_metrics['time_taken']*1000:.2f} | {road_metrics.get('total_cost', 'N/A')}")
    
    # Train distance
    print(f"{'Train distance':<20} | {train_meetup:<15} | {train_metrics['nodes_generated']:<15} | {train_metrics['time_taken']*1000:.2f} | {train_metrics.get('total_cost', 'N/A')}")


if __name__ == "__main__":
    main()