"""
Deep Q-Learning with Q-Matrix Transfer Learning for Fire Evacuation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from pathlib import Path
import math
import random
from typing import Dict, Tuple, List, Optional, Set
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

# Exact parameters from the paper's table
REWARD_EXIT = 10.0        # Reward for reaching exit
REWARD_MOVE = -1.0        # Penalty for normal movement
FIRE_SPREAD_RATE = 0.01   # δ Fire spread rate
BOTTLENECK_LIMIT = 10     # B Bottleneck limit
ACTION_UNCERTAINTY = 0.05  # p Action uncertainty
SIGMA = 10.0             # σ Q-matrix noise
TARGET_CRS = "EPSG:32610"


class BaseEvacuationEnv(gym.Env):
    """Base evacuation environment - shared base class for pretraining and main environment"""
    
    def __init__(self, mesh_dir: str = "outputs", max_nodes: int = 50):
        super().__init__()
        self.mesh_dir = Path(mesh_dir)
        self.max_nodes = max_nodes
        
        # Load and build graph - ensure pretraining and main env use same graph
        self._load_graph()
        
        # Action space: n² possible actions
        self.action_space = spaces.Discrete(self.n_nodes * self.n_nodes)
        # Observation space
        self.observation_space = spaces.Discrete(self.n_nodes)
        
    def _load_graph(self):
        """Load graph structure - shared by pretraining and main environment"""
        from evac_multi import load_inputs, build_graph, nearest_node
        
        self.edges, self.nodes_gdf, self.shelters_gdf = load_inputs(self.mesh_dir)
        
        # Build base graph without fire
        empty_fire = unary_union([])
        full_G = build_graph(self.edges, empty_fire, alpha=0.0)
        
        # Extract subgraph
        if full_G.number_of_nodes() > self.max_nodes:
            self.G = self._extract_connected_subgraph(full_G)
        else:
            self.G = full_G
        
        self.nodes_list = list(self.G.nodes())
        self.n_nodes = len(self.nodes_list)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes_list)}
        
        # Build adjacency matrix
        self.adjacency_matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        for u, v in self.G.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                i = self.node_to_idx[u]
                j = self.node_to_idx[v]
                self.adjacency_matrix[i, j] = 1
                # Undirected graph needs bidirectional edges
                if not self.G.is_directed():
                    self.adjacency_matrix[j, i] = 1
        
        # Mark exit nodes
        self.exit_indices = set()
        for _, shelter in self.shelters_gdf.iterrows():
            from evac_multi import nearest_node
            shelter_node = nearest_node(shelter.geometry, self.nodes_list)
            if shelter_node in self.node_to_idx:
                idx = self.node_to_idx[shelter_node]
                self.exit_indices.add(idx)
        
        # Ensure at least one exit
        if not self.exit_indices:
            self.exit_indices = {self.n_nodes - 1}
            
        print(f"Graph structure: {self.n_nodes} nodes, {len(self.exit_indices)} exits")
        
    def _extract_connected_subgraph(self, full_G):
        """Extract connected subgraph"""
        from evac_multi import nearest_node
        
        # Start from shelters
        shelter_nodes = []
        for _, shelter in self.shelters_gdf.iterrows():
            shelter_node = nearest_node(shelter.geometry, list(full_G.nodes()))
            if shelter_node in full_G:
                shelter_nodes.append(shelter_node)
        
        if not shelter_nodes:
            # If no shelters, choose node with highest degree
            degrees = dict(full_G.degree())
            shelter_nodes = [max(degrees, key=degrees.get)]
        
        # BFS to build subgraph
        visited = set()
        queue = deque(shelter_nodes)
        for node in shelter_nodes:
            visited.add(node)
        
        while queue and len(visited) < self.max_nodes:
            current = queue.popleft()
            for neighbor in full_G.neighbors(current):
                if neighbor not in visited and len(visited) < self.max_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        subgraph = full_G.subgraph(visited).copy()
        
        # Ensure connectivity
        if not nx.is_weakly_connected(subgraph):
            largest_cc = max(nx.weakly_connected_components(subgraph), key=len)
            subgraph = subgraph.subgraph(largest_cc).copy()
        
        return subgraph


class PretrainingEnv(BaseEvacuationEnv):
    """Pretraining environment - learn shortest paths, inherit base environment to ensure consistent graph structure"""
    
    def reset(self, seed=None):
        """Reset - random starting position"""
        super().reset(seed=seed)
        # Choose randomly from non-exit nodes
        non_exit_nodes = [i for i in range(self.n_nodes) if i not in self.exit_indices]
        if non_exit_nodes:
            self.current_node = random.choice(non_exit_nodes)
        else:
            self.current_node = 0
        self.steps = 0
        return self.current_node, {}
    
    def step(self, action):
        """Execute action - Algorithm 2"""
        self.steps += 1
        
        # Parse action
        from_idx = action // self.n_nodes
        to_idx = action % self.n_nodes
        
        # Verify: must start from current node
        if from_idx != self.current_node:
            return self.current_node, -10, False, False, {}
        
        # Check if reached exit
        if to_idx in self.exit_indices and self.adjacency_matrix[from_idx, to_idx] == 1:
            return to_idx, 1, True, False, {"reached_exit": True}
        
        # Check move legality
        if self.adjacency_matrix[from_idx, to_idx] == 0:
            return self.current_node, -10, False, False, {"illegal_move": True}
        
        # Execute legal move
        self.current_node = to_idx
        reward = -1
        
        # Check timeout
        truncated = self.steps >= 100
        
        return self.current_node, reward, False, truncated, {}


class FireEvacuationEnv(BaseEvacuationEnv):
    """Fire evacuation environment - inherit base environment, add fire dynamics"""
    
    def __init__(self, mesh_dir: str = "outputs", timestep: int = None,
                 alpha: float = 3.0, max_steps: int = 1000, max_nodes: int = 50,
                 n_people_per_room: int = 10):
        super().__init__(mesh_dir, max_nodes)
        
        self.timestep = timestep
        self.alpha = alpha
        self.max_steps = max_steps
        self.n_people_per_room = n_people_per_room
        self.current_step = 0
        
        # Load fire data
        self._load_fire_data()
        
        # State space: number of people in each room
        self.observation_space = spaces.Box(
            low=0, high=100, 
            shape=(self.n_nodes,), dtype=np.float32
        )
        
        self.reset()
    
    def _load_fire_data(self):
        """Load fire data"""
        if self.timestep is not None:
            fire_file = self.mesh_dir / f"fires_t{self.timestep}.geojson"
        else:
            fire_file = self.mesh_dir / "fires.geojson"
        
        if fire_file.exists():
            self.fires_gdf = gpd.read_file(fire_file).to_crs(TARGET_CRS)
            self.fire_union = unary_union(self.fires_gdf.geometry)
        else:
            self.fires_gdf = gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
            self.fire_union = Point(0, 0).buffer(0)
        
        self._init_fire_degrees()
    
    def _init_fire_degrees(self):
        """Initialize fire degree vector D"""
        self.fire_degrees = np.zeros(self.n_nodes)
        self.fire_spread_delta = np.zeros(self.n_nodes)
        
        for idx, node in enumerate(self.nodes_list):
            node_pt = Point(node[0], node[1])
            
            if len(self.fires_gdf) > 0:
                # In fire
                if self.fire_union.contains(node_pt):
                    self.fire_degrees[idx] = 1.0
                    self.fire_spread_delta[idx] = 0.0
                else:
                    # Calculate distance to fire
                    min_dist = node_pt.distance(self.fire_union)
                    
                    # Set fire degree based on distance
                    if min_dist < 50:
                        self.fire_degrees[idx] = 0.5
                        self.fire_spread_delta[idx] = FIRE_SPREAD_RATE * 2
                    elif min_dist < 100:
                        self.fire_degrees[idx] = 0.2
                        self.fire_spread_delta[idx] = FIRE_SPREAD_RATE * 1.5
                    elif min_dist < 200:
                        self.fire_degrees[idx] = 0.1
                        self.fire_spread_delta[idx] = FIRE_SPREAD_RATE
                    else:
                        self.fire_degrees[idx] = 0.0
                        self.fire_spread_delta[idx] = FIRE_SPREAD_RATE * 0.5
    
    def _update_fire_degrees(self):
        """Update fire degrees - Formula (23)"""
        self.fire_degrees = np.clip(
            self.fire_degrees + self.fire_spread_delta, 0, 1.0
        )
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize people distribution
        self.people_in_rooms = np.ones(self.n_nodes, dtype=np.float32) * self.n_people_per_room
        
        # No people at exits
        for idx in self.exit_indices:
            self.people_in_rooms[idx] = 0
        
        self.initial_total_people = int(np.sum(self.people_in_rooms))
        self.total_evacuated = 0
        
        # Reset fire
        self._init_fire_degrees()
        
        return self.people_in_rooms.copy(), {}
    
    def step(self, action):
        """Execute action - Algorithm 1"""
        self.current_step += 1
        
        # Parse action
        from_idx = action // self.n_nodes  # vi
        to_idx = action % self.n_nodes     # vj
        
        # Basic validation
        if from_idx >= self.n_nodes or to_idx >= self.n_nodes:
            return self._get_state(), -100, False, False, self._get_info()
        
        # No people to move
        if self.people_in_rooms[from_idx] <= 0:
            return self._get_state(), -1, False, False, self._get_info()
        
        # Action uncertainty - key feature from paper
        if random.random() < ACTION_UNCERTAINTY:
            # Action fails, but time continues
            self._update_fire_degrees()
            reward = -1  # Time penalty
            return self._get_state(), reward, False, False, self._get_info()
        
        # Calculate reward
        reward = 0
        
        # Case 1: Successful evacuation to exit
        if to_idx in self.exit_indices and self.adjacency_matrix[from_idx, to_idx] == 1:
            reward = REWARD_EXIT
            self.people_in_rooms[from_idx] -= 1
            self.total_evacuated += 1
            
        # Case 2: Illegal move
        elif self.adjacency_matrix[from_idx, to_idx] == 0:
            reward = -20  # Heavy penalty
            
        # Case 3: Bottleneck
        elif self.people_in_rooms[to_idx] >= BOTTLENECK_LIMIT and to_idx not in self.exit_indices:
            reward = -5
            
        # Case 4: Normal move
        else:
            # Paper formula: r = -[d(vj)]^t
            d_vj = self.fire_degrees[to_idx]
            if d_vj > 0:
                reward = -(d_vj ** self.current_step)
            else:
                reward = REWARD_MOVE
            
            # Execute move
            self.people_in_rooms[from_idx] -= 1
            self.people_in_rooms[to_idx] += 1
        
        # Update fire
        self._update_fire_degrees()
        
        # Check termination
        non_exit_people = sum(self.people_in_rooms[i] for i in range(self.n_nodes) 
                             if i not in self.exit_indices)
        terminated = non_exit_people == 0
        truncated = self.current_step >= self.max_steps
        
        # Extra reward for complete evacuation
        if terminated:
            reward += 50
        
        return self._get_state(), reward, terminated, truncated, self._get_info()
    
    def _get_state(self):
        return self.people_in_rooms.copy()
    
    def _get_info(self):
        non_exit_people = sum(self.people_in_rooms[i] for i in range(self.n_nodes) 
                             if i not in self.exit_indices)
        return {
            "total_evacuated": self.total_evacuated,
            "people_remaining": int(non_exit_people),
            "initial_people": self.initial_total_people,
            "evacuation_rate": self.total_evacuated / max(1, self.initial_total_people),
            "current_step": self.current_step
        }


class QMatrixLearner:
    """Q-Learning - ensure sufficient learning"""
    
    def __init__(self, env: PretrainingEnv):
        self.env = env
        self.q_matrix = np.zeros((env.n_nodes, env.n_nodes))
        
        # Initialize Q values - give small negative values to reachable paths
        for i in range(env.n_nodes):
            for j in range(env.n_nodes):
                if env.adjacency_matrix[i, j] == 1:
                    self.q_matrix[i, j] = -0.1
                else:
                    self.q_matrix[i, j] = -100  # Illegal paths
        
        # Learning parameters
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def train(self, episodes: int = 3000):
        """Train Q-matrix - increase training episodes to ensure convergence"""
        print(f"Starting Q-Learning training ({episodes} episodes)...")
        
        success_count = 0
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0
            
            while not done and steps < 100:
                # ε-greedy strategy
                if random.random() < self.epsilon:
                    # Explore: only choose legal actions
                    valid_actions = []
                    for to_idx in range(self.env.n_nodes):
                        if self.env.adjacency_matrix[state, to_idx] == 1:
                            valid_actions.append(state * self.env.n_nodes + to_idx)
                    
                    if valid_actions:
                        action = random.choice(valid_actions)
                    else:
                        action = state * self.env.n_nodes + state
                else:
                    # Exploit: choose best action
                    # Only consider actions from current position
                    q_values = self.q_matrix[state, :].copy()
                    # Mask illegal actions
                    for j in range(self.env.n_nodes):
                        if self.env.adjacency_matrix[state, j] == 0:
                            q_values[j] = -float('inf')
                    
                    to_idx = np.argmax(q_values)
                    action = state * self.env.n_nodes + to_idx
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Q-learning update
                from_idx = action // self.env.n_nodes
                to_idx = action % self.env.n_nodes
                
                if from_idx == state:  # Ensure action is legal
                    old_q = self.q_matrix[from_idx, to_idx]
                    
                    if terminated and reward > 0:  # Reached exit
                        target = reward
                        success_count += 1
                    else:
                        # Calculate max Q value for next state
                        next_q_values = self.q_matrix[next_state, :].copy()
                        # Only consider legal actions
                        for j in range(self.env.n_nodes):
                            if self.env.adjacency_matrix[next_state, j] == 0:
                                next_q_values[j] = -float('inf')
                        
                        max_next_q = np.max(next_q_values) if np.max(next_q_values) > -float('inf') else 0
                        target = reward + self.gamma * max_next_q
                    
                    # Update Q value
                    self.q_matrix[from_idx, to_idx] = old_q + self.lr * (target - old_q)
                
                state = next_state
                steps += 1
            
            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if episode % 200 == 0:
                valid_q_values = self.q_matrix[self.q_matrix > -100]
                if len(valid_q_values) > 0:
                    print(f"  Episode {episode}, Q-matrix range: [{valid_q_values.min():.2f}, {valid_q_values.max():.2f}], "
                          f"Epsilon: {self.epsilon:.3f}")
        
        # Add conditional noise - key from paper
        self._add_conditional_noise()
        
        print(f"Q-Learning training complete!")
        return self.q_matrix
    
    def _add_conditional_noise(self):
        """Add conditional noise - paper formula"""
        for i in range(self.env.n_nodes):
            for j in range(self.env.n_nodes):
                if self.q_matrix[i, j] > -100:  # Only add noise to valid paths
                    if self.q_matrix[i, j] <= 0:
                        self.q_matrix[i, j] += SIGMA
                    else:
                        self.q_matrix[i, j] -= SIGMA


class DuelingDQN(nn.Module):
    """Dueling DQN - deeper network"""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        
        # Deeper feature extraction
        self.features = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Value function
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage function
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    
    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Dueling formula
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class QMatrixDQN:
    """Q-Matrix pretrained DQN"""
    
    def __init__(self, state_size: int, action_size: int, adjacency_matrix: np.ndarray):
        self.state_size = state_size
        self.action_size = action_size
        self.adjacency_matrix = adjacency_matrix
        
        # Network parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        
        # Optimizer - use smaller learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        
        # Experience replay
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        
        # Double DQN
        self.use_double = True
        
        # Other parameters
        self.gamma = 0.9
        self.target_update_freq = 200
        self.steps = 0
        
        # Record best performance
        self.best_evacuation_rate = 0.0
    
    def pretrain_with_qmatrix(self, q_matrix: np.ndarray, episodes: int = 2000):
        """Pretraining method"""
        print("\nStarting Q-Matrix pretraining DQN...")
        
        # Generate diverse training data
        all_states = []
        all_targets = []
        
        # Generate multiple state configurations for each node
        for _ in range(episodes // self.state_size):
            for node_idx in range(self.state_size):
                # Randomly generate people distribution
                state = np.random.randint(0, 15, size=self.state_size).astype(np.float32)
                # Ensure current node has people
                state[node_idx] = max(1, state[node_idx])
                
                # Generate target Q values
                target_q = np.full(self.action_size, -100.0)
                
                for action_idx in range(self.action_size):
                    from_idx = action_idx // self.state_size
                    to_idx = action_idx % self.state_size
                    
                    # Only use Q-matrix values for actions from current node
                    if from_idx == node_idx and from_idx < q_matrix.shape[0] and to_idx < q_matrix.shape[1]:
                        target_q[action_idx] = q_matrix[from_idx, to_idx]
                
                all_states.append(state)
                all_targets.append(target_q)
        
        # Convert to numpy arrays
        all_states_np = np.array(all_states)
        all_targets_np = np.array(all_targets)
        
        # Pretrain network
        best_loss = float('inf')
        for epoch in range(1000):
            indices = np.random.permutation(len(all_states_np))
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_states = torch.FloatTensor(all_states_np[batch_indices]).to(self.device)
                batch_targets = torch.FloatTensor(all_targets_np[batch_indices]).to(self.device)
                
                predicted = self.q_network(batch_states)
                loss = F.mse_loss(predicted, batch_targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if epoch % 200 == 0:
                print(f"  Pretrain epoch {epoch}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
        
        # Sync target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        print("Q-Matrix pretraining complete!")
    
    def get_legal_actions(self, state_idx):
        """Get legal actions - based on adjacency matrix"""
        legal_actions = []
        for to_idx in range(self.state_size):
            if self.adjacency_matrix[state_idx, to_idx] == 1:
                action = state_idx * self.state_size + to_idx
                legal_actions.append(action)
        return legal_actions
    
    def act(self, state, training=True):
        """Choose action - consider action legality"""
        # Find rooms with people
        rooms_with_people = np.where(state > 0)[0]
        
        if len(rooms_with_people) == 0:
            return 0  # No people to move
        
        if training and random.random() < self.epsilon:
            # Explore: randomly choose a room with people, then choose legal action
            from_idx = random.choice(rooms_with_people)
            legal_actions = self.get_legal_actions(from_idx)
            if legal_actions:
                return random.choice(legal_actions)
            else:
                return from_idx * self.state_size + from_idx
        else:
            # Exploit: choose best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Mask illegal actions
            masked_q_values = np.full_like(q_values, -float('inf'))
            
            for from_idx in rooms_with_people:
                for to_idx in range(self.state_size):
                    if self.adjacency_matrix[from_idx, to_idx] == 1:
                        action = from_idx * self.state_size + to_idx
                        masked_q_values[action] = q_values[action]
            
            if np.all(masked_q_values == -float('inf')):
                # No legal actions, choose randomly
                from_idx = random.choice(rooms_with_people)
                return from_idx * self.state_size + from_idx
            
            return np.argmax(masked_q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Experience replay training"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        # First convert to numpy arrays, then create tensors
        states_np = np.array([e[0] for e in batch])
        next_states_np = np.array([e[3] for e in batch])
        
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        if self.use_double:
            # Double DQN
            next_actions = self.q_network(next_states).max(1)[1]
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        else:
            next_q = self.target_network(next_states).max(1)[0]
        
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_rate': self.best_evacuation_rate
        }, path)
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.best_evacuation_rate = checkpoint.get('best_rate', 0.0)


def train_qmp_dqn(mesh_dir: str = "outputs", episodes: int = 3000,
                  use_double: bool = True, use_dueling: bool = True,
                  max_nodes: int = 50):
    """Training process"""
    
    print("="*70)
    print("Fire Evacuation RL Training - Q-Matrix Transfer Learning")
    print("="*70)
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize log file
    log_file = output_dir / "log.txt"
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write("="*70 + "\n")
    
    # Step 1: Q-Learning pretraining
    print("\nUsing single environment for training...")
    print("="*70)
    print("Step 1: Q-Learning Pretraining") 
    print("="*70)
    
    pretrain_env = PretrainingEnv(mesh_dir, max_nodes)
    q_learner = QMatrixLearner(pretrain_env)
    q_matrix = q_learner.train(episodes=3000)  # Train for 3000 episodes
    
    # Step 2: Create main environment
    print("\n" + "="*70)
    print("Step 2: Create Fire Evacuation Environment")
    print("="*70)
    
    env = FireEvacuationEnv(mesh_dir, max_nodes=max_nodes)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State space size: {state_size}")
    print(f"Action space size: {action_size}")
    print(f"Initial total people: {env.initial_total_people}")
    print(f"Number of exits: {len(env.exit_indices)}")
    
    # Verify environment consistency
    assert pretrain_env.n_nodes == env.n_nodes, "Pretraining and main environment node counts don't match!"
    assert np.array_equal(pretrain_env.adjacency_matrix, env.adjacency_matrix), "Adjacency matrices don't match!"
    
    # Step 3: Pretrain DQN
    print("\n" + "="*70)
    print("Step 3: Q-Matrix Pretraining DQN")
    print("="*70)
    
    agent = QMatrixDQN(state_size, action_size, env.adjacency_matrix)
    agent.pretrain_with_qmatrix(q_matrix, episodes=2000)
    
    # Step 4: Main training
    print("\n" + "="*70)
    print("Step 4: Fire Evacuation Environment Training")
    print("="*70)
    print()
    
    scores = deque(maxlen=100)
    evacuation_rates = deque(maxlen=100)
    best_score = -float('inf')
    best_evacuated = 0
    best_evacuation_rate = 0.0
    
    # Data for plotting
    all_scores = []
    all_evacuation_rates = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            state = next_state
            total_reward += reward
        
        agent.update_epsilon()
        scores.append(total_reward)
        evacuation_rates.append(info['evacuation_rate'])
        
        # Record all data for plotting
        all_scores.append(total_reward)
        all_evacuation_rates.append(info['evacuation_rate'])
        
        # Save best model
        if info['evacuation_rate'] > best_evacuation_rate:
            best_evacuation_rate = info['evacuation_rate']
            best_evacuated = info['total_evacuated']
            best_score = total_reward
            agent.best_evacuation_rate = best_evacuation_rate
            agent.save_model(output_dir / "best_model.pth")
            
            log_msg = f"\nEpisode {episode}: New best model! Evacuation rate: {best_evacuation_rate:.2%}, Evacuated: {best_evacuated}/{env.initial_total_people}\n"
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg)
        
        # Update charts and logs every 10 episodes
        if episode % 10 == 0:
            avg_score = np.mean(scores) if scores else 0
            avg_evac_rate = np.mean(evacuation_rates) if evacuation_rates else 0
            
            log_msg = f"Episode {episode}, Avg score: {avg_score:.2f}, Best score: {best_score:.2f}, " \
                     f"Epsilon: {agent.epsilon:.3f}, Current evac: {info['total_evacuated']}/{env.initial_total_people}, " \
                     f"Best evac: {best_evacuated}, Evac rate: {info['evacuation_rate']:.2%}, " \
                     f"Best rate: {best_evacuation_rate:.2%}, Avg rate: {avg_evac_rate:.2%}\n"
            
            print(log_msg.strip())
            with open(log_file, 'a') as f:
                f.write(log_msg)
            
            # Update charts
            if len(all_scores) > 1:
                # Reward curve
                plt.figure(figsize=(10, 6))
                plt.plot(all_scores, alpha=0.3, color='blue', label='Raw Scores')
                
                # Calculate smoothed curve
                window = min(100, len(all_scores) // 10, 10)
                if len(all_scores) >= window:
                    smoothed_scores = np.convolve(all_scores, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(all_scores)), smoothed_scores, 
                            color='red', linewidth=2, label=f'{window}-Episode Moving Average')
                
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Training Progress - Reward')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'reward_curve.png', dpi=100, bbox_inches='tight')
                plt.close()
                
                # Evacuation rate curve
                plt.figure(figsize=(10, 6))
                plt.plot(all_evacuation_rates, alpha=0.3, color='green', label='Raw Evacuation Rate')
                
                if len(all_evacuation_rates) >= window:
                    smoothed_rates = np.convolve(all_evacuation_rates, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(all_evacuation_rates)), smoothed_rates,
                            color='darkgreen', linewidth=2, label=f'{window}-Episode Moving Average')
                
                plt.xlabel('Episode')
                plt.ylabel('Evacuation Rate')
                plt.title('Training Progress - Evacuation Rate')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.05)
                plt.tight_layout()
                plt.savefig(output_dir / 'evacuation_rate_curve.png', dpi=100, bbox_inches='tight')
                plt.close()
    
    print("\nTraining complete!")
    print(f"Best evacuation rate: {best_evacuation_rate:.2%}")
    
    agent.save_model(output_dir / "final_model.pth")
    
    # Write final log
    with open(log_file, 'a') as f:
        f.write("\n" + "="*70 + "\n")
        f.write(f"Training completed at {datetime.now()}\n")
        f.write(f"Best evacuation rate: {best_evacuation_rate:.2%}\n")
        f.write(f"Best evacuated: {best_evacuated}/{env.initial_total_people}\n")
    
    return agent