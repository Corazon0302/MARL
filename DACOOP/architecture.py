
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


# Create discretized parameter pairs (λ, η) for APF-A
lambda_values = np.linspace(0.0, APF_LAMBDA_MAX, APF_LAMBDA_COUNT)
action_pairs = []
for lambda_val in lambda_values:
    for eta_val in APF_ETA_VALUES:
        action_pairs.append((lambda_val, eta_val))
ACTION_SIZE = len(action_pairs)  # 24 pairs (8 lambda × 3 eta)

class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(EmbeddingNetwork, self).__init__()
        self.embedding = nn.Linear(2, embedding_size)
        
    def forward(self, x):
        return F.relu(self.embedding(x))

class KeyNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(KeyNetwork, self).__init__()
        self.key = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, x):
        return F.relu(self.key(x))

class AttentionNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionNetwork, self).__init__()
        # Input: [local_obs (6) + mean_embedding + key]
        # local_obs = [evader_distance, evader_angle, obstacle1_distance, obstacle1_angle, obstacle2_distance, obstacle2_angle]
        self.attention = nn.Linear(6 + embedding_size * 2, 1)
        
    def forward(self, local_obs, mean_embedding, key):
        x = torch.cat([local_obs, mean_embedding, key], dim=1)
        return self.attention(x)

class DuelingNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(DuelingNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        # State value stream
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        # Action advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_size)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        # Compute state value V(s)
        value = self.value_stream(features)
        # Compute advantages A(s,a)
        advantages = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values, value  # Return both Q-values and state value

class APFA:
    def __init__(self):
        self.rho = APF_RHO  # Influence range of obstacles
        
    def compute_forces(self, agent_pos, target_pos, agent_vel, obstacles_info, neighbor_info, lambda_val, eta_val, attention_scores):
        # Attractive force toward evader
        F_att = self._compute_attraction(agent_pos, target_pos)

        # Repulsive force from obstacles
        F_rep = self._compute_repulsion(obstacles_info, eta_val)
        
        # Inter-robot force with attention weights
        F_in = self._compute_inter_robot_force(agent_pos, neighbor_info, attention_scores, lambda_val)
        
        # Combine forces
        total_force = F_att + F_rep + F_in
        
        # Check if wall-following behavior is needed
        if np.dot(F_att + F_rep, F_att) < 0:
            total_force = self._wall_following(F_att, F_rep, F_in, agent_vel)
            
        return total_force
    
    def _compute_attraction(self, agent_pos, target_pos):
        direction = target_pos - agent_pos
        distance = np.linalg.norm(direction) + 1e-6
        return direction / distance
    
    def _compute_repulsion(self, obstacles_info, eta_val):  
        F_rep = np.zeros(2, dtype=np.float32)     
        for i, obstacle in enumerate(obstacles_info):
            distance, angle, res_pos = obstacle
            if distance <= self.rho:
                force_magnitude = - (eta_val * (1.0 / distance - 1.0 / self.rho) / (distance**3)) * res_pos
                F_rep += force_magnitude

        return F_rep
    
    def _compute_inter_robot_force(self, agent_pos, neighbor_info, attention_scores, lambda_val):
        F_in = np.zeros(2, dtype=np.float32)
        for i, (distance, angle, pos_x, pos_y) in enumerate(neighbor_info):
            direction = np.array([pos_x, pos_y])
            # Compute force component according to equation (12) in the paper
            force_component = attention_scores[i] * (0.5 - lambda_val / distance) * direction / distance
            F_in += force_component
            
        return F_in
    
    def _wall_following(self, F_att, F_rep, F_in, agent_vel):
        agent_heading = agent_vel / (np.linalg.norm(agent_vel) + 1e-6)
        # Compute two possible wall-following directions
        rotate_matrix = np.array([[0, -1], [1, 0]])  # 90-degree rotation matrix
        n1 = np.dot(rotate_matrix, F_rep)
        n2 = -n1
        
        # Choose the best following direction
        if np.linalg.norm(F_in) < 0.3:  # When intrinsic coordination force is small
            # Choose the direction with a smaller angle to the current heading
            if np.dot(n1, agent_heading) > np.dot(n2, agent_heading):
                follow_dir = n1
            else:
                follow_dir = n2
        else:
            # Choose the direction consistent with the intrinsic coordination force
            if np.dot(n1, F_in) > 0:
                follow_dir = n1
            else:
                follow_dir = n2
        
        # Increase repulsion force if too close to obstacles
        if np.linalg.norm(F_rep) > 3.0:
            follow_dir = follow_dir + F_rep
            
        return follow_dir / (np.linalg.norm(follow_dir) + 1e-6)

def extract_local_observation(obs):
    evader_rel_pos = obs[12:14]
    evader_distance = np.linalg.norm(evader_rel_pos)
    evader_angle = np.degrees(np.arctan2(evader_rel_pos[1], evader_rel_pos[0]))
    
    obstacle1_rel_pos = obs[6:8]
    obstacle1_distance = np.linalg.norm(obstacle1_rel_pos)
    obstacle1_angle = np.degrees(np.arctan2(obstacle1_rel_pos[1], obstacle1_rel_pos[0]))
    obstacle2_rel_pos = obs[8:10]
    obstacle2_distance = np.linalg.norm(obstacle2_rel_pos)
    obstacle2_angle = np.degrees(np.arctan2(obstacle2_rel_pos[1], obstacle2_rel_pos[0]))
    obstacle_distance = np.array([obstacle1_distance, obstacle2_distance])
    obstacle_angle = np.array([obstacle1_angle, obstacle2_angle])
    
    return evader_distance, evader_angle, obstacle_distance, obstacle_angle

def extract_neighbor_info(obs):
    neighbor_info = []
    rel_pos1 = obs[8:10]
    rel_pos2 = obs[10:12]
    distance1 = np.linalg.norm(rel_pos1)
    distance2 = np.linalg.norm(rel_pos2)
    angle1 = np.degrees(np.arctan2(rel_pos1[1], rel_pos1[0]))
    angle2 = np.degrees(np.arctan2(rel_pos2[1], rel_pos2[0]))
    neighbor_info.append([distance1, angle1, rel_pos1[0], rel_pos1[1]])
    neighbor_info.append([distance2, angle2, rel_pos2[0], rel_pos2[1]])
    return neighbor_info

# Define Experience namedtuple
from collections import namedtuple
Experience = namedtuple('Experience', 
                        ['state', 'neighbor_states', 'action', 'reward', 
                         'next_state', 'next_neighbor_states', 'done'])

