import numpy as np
import torch
from mpe2 import simple_tag_v3
import supersuit as ss
from collections import namedtuple

# Constants and Hyperparameters
BUFFER_SIZE = 100000 
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.01              # Soft update parameter
LR_ACTOR = 0.0001       # Actor learning rate
LR_CRITIC = 0.0003      # Critic learning rate
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 64
NUM_EPISODES = 10000
MAX_STEPS = 100
START_TRAINING = 1000   # Steps before training starts
UPDATE_EVERY = 5      # Update frequency
NOISE_STD = 0.1         # Exploration noise
ATTENTION_HEADS = 4     # Number of attention heads
CLIP_GRAD = 0.5         # Gradient clipping
KL_WEIGHT = 0.05        # KL divergence weight
SEED = 42

# APF-A Parameters
APF_LAMBDA_MAX = 1.0    # Maximum lambda value
APF_LAMBDA_COUNT = 8    # Number of discretized lambda values
APF_ETA_VALUES = [0.5, 1.0, 2.0]  # Eta candidates
APF_RHO = 0.05          # Influence range of obstacles

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_env(seed=SEED, render_mode=None):
    """Create and configure the MPE environment"""
    env = simple_tag_v3.parallel_env(
        num_good=1,              # 1 prey (evader)
        num_adversaries=3,       # 3 predators (pursuers)
        num_obstacles=2,         # 2 obstacles
        max_cycles=MAX_STEPS,    # maximum steps
        continuous_actions=True, # continuous action space
        dynamic_rescaling=True,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    # Normalize observations to improve learning stability
    env = ss.observation_lambda_v0(env, lambda obs, obs_space, agent: obs / 10.0)
    return env

def extract_agent_position(obs):
    """Extract agent position from observation"""
    return obs[2:4]

def extract_agent_velocity(obs):
    """Extract agent velocity from observation"""
    return obs[0:2]

def extract_evader_position(obs):
    """Extract relative position of evader from observation"""
    # This assumes evader relative position is at indices 4-6
    # Adjust based on actual observation structure
    return obs[4:6]

def extract_obstacle_position(obs):
    """Extract relative position of nearest obstacle from observation"""
    # This assumes obstacle relative position is at indices 6-8
    # Adjust based on actual observation structure
    return obs[6:8]

def convert_force_to_action(force_vector):
    """
    Convert a 2D force vector to a 5D discrete action
    [no_action, left, right, down, up]
    """
    action = np.zeros(5)
    force_magnitude = np.linalg.norm(force_vector)
    
    if force_magnitude < 1e-6:
        # No significant force, choose no_action
        action[0] = 1.0
        return action
    
    # Normalize force vector
    normalized_force = force_vector / force_magnitude
    fx, fy = normalized_force
    
    # Determine primary movement direction
    if abs(fx) > abs(fy):  # Horizontal movement dominates
        if fx < 0:         # Move left
            action[1] = 1.0
        else:              # Move right
            action[2] = 1.0
    else:                  # Vertical movement dominates
        if fy < 0:         # Move down
            action[3] = 1.0
        else:              # Move up
            action[4] = 1.0
            
    return action

# Experience namedtuple for replay buffer
Experience = namedtuple('Experience', 
                        ['state', 'neighbor_states', 'action', 'reward', 
                         'next_state', 'next_neighbor_states', 'done'])
