import numpy as np
import torch
from mpe2 import simple_tag_v3
import supersuit as ss
from collections import namedtuple

# Constants and Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.01              # Soft update parameter
LR_ACTOR = 0.0001       # Actor learning rate
LR_CRITIC = 0.0003      # Critic learning rate
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 64
NUM_EPISODES = 500
MAX_STEPS = 80
START_TRAINING = 1000   # Steps before training starts
UPDATE_EVERY = 10      # Update frequency
NOISE_STD = 0.1         # Exploration noise
ATTENTION_HEADS = 4     # Number of attention heads
CLIP_GRAD = 0.5         # Gradient clipping
KL_WEIGHT = 0.05        # KL divergence weight
SEED = 43

# APF-A Parameters
APF_LAMBDA_MAX = 0.2    # Maximum lambda value
APF_LAMBDA_COUNT = 8    # Number of discretized lambda values
APF_ETA_VALUES = [0.5, 1.0, 2.0]  # Eta candidates
APF_RHO = 0.05       # Influence range of obstacles

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

# Experience namedtuple for replay buffer
Experience = namedtuple('Experience', 
                        ['state', 'neighbor_states', 'action', 'reward', 
                         'next_state', 'next_neighbor_states', 'done'])
