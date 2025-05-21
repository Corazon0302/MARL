import os
import random
import numpy as np
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import datetime
from collections import deque, namedtuple
import mpe2
from mpe2 import simple_tag_v3
import supersuit as ss
import matplotlib
from architecture import *
from agent import DACOOPAgent

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Configure matplotlib
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK JP', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.use('Agg')

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        
    def add(self, state, neighbor_states, action, reward, next_state, next_neighbor_states, done):
        self.memory.append(self.experience(state, neighbor_states, action, reward, next_state, next_neighbor_states, done))
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.FloatTensor(np.vstack([e.state for e in experiences if e is not None])).to(device)
        
        # Process neighbor states (creating a list of tensors for each neighbor)
        neighbor_states = []
        for i in range(len(experiences[0].neighbor_states)):
            neighbor_i_states = torch.FloatTensor(np.vstack([
                e.neighbor_states[i] for e in experiences if e is not None
            ])).to(device)
            neighbor_states.append(neighbor_i_states)
            
        actions = torch.FloatTensor(np.vstack([e.action for e in experiences if e is not None])).to(device)
        rewards = torch.FloatTensor(np.vstack([e.reward for e in experiences if e is not None])).to(device)
        next_states = torch.FloatTensor(np.vstack([e.next_state for e in experiences if e is not None])).to(device)
        
        # Process next neighbor states
        next_neighbor_states = []
        for i in range(len(experiences[0].next_neighbor_states)):
            next_neighbor_i_states = torch.FloatTensor(np.vstack([
                e.next_neighbor_states[i] for e in experiences if e is not None
            ])).to(device)
            next_neighbor_states.append(next_neighbor_i_states)
            
        dones = torch.FloatTensor(np.vstack([e.done for e in experiences if e is not None])).to(device)
        
        return (states, neighbor_states, actions, rewards, next_states, next_neighbor_states, dones)
        
    def __len__(self):
        return len(self.memory)

class DACOOPTrainer:
    def __init__(self, env, predator_only=True):
        self.env = env
        self.predator_only = predator_only
        
        self.agents = env.possible_agents
        self.predator_agents = [a for a in self.agents if "adversary" in a]
        self.prey_agents = [a for a in self.agents if "adversary" not in a]
        
        self.dacoop_agents: dict[str, DACOOPAgent] = {}
        for agent_id in self.agents:
            obs_size = env.observation_space(agent_id).shape[0]
            act_size = env.action_space(agent_id).shape[0]            
            if "adversary" in agent_id or not predator_only:
                self.dacoop_agents[agent_id] = DACOOPAgent(agent_id, obs_size, act_size)

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
    def get_actions(self, obs, add_noise=True):
        actions = {}
        predator_obs = {a: obs[a] for a in self.predator_agents if a in obs}
        prey_obs = {a: obs[a] for a in self.prey_agents if a in obs}
        
        for agent_id in predator_obs:
            neighbor_obs = []
            
            for other_id in predator_obs:
                if other_id != agent_id:
                    other_obs = predator_obs[other_id]
                    neighbor_obs.append(other_obs)
            for id in prey_obs:
                prey_ob = prey_obs[id]

            apf_action, lambda_val, eta_val, attention_scores = self.dacoop_agents[agent_id].act(
                predator_obs[agent_id], 
                neighbor_obs, 
                prey_ob,
                add_noise=add_noise
            )
            actions[agent_id] = np.array(apf_action, dtype=np.float32)
        
        for agent_id in prey_obs:
            if agent_id in self.dacoop_agents:
                neighbor_obs = []
                apf_action, _, _, _ = self.dacoop_agents[agent_id].act(prey_obs[agent_id], neighbor_obs, add_noise)
                norm = np.linalg.norm(apf_action)
                actions[agent_id] = np.array(apf_action, dtype=np.float32)
            else:
                actions[agent_id] = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                
        return actions
            
    def update(self):
        experiences = self.buffer.sample()
        losses = {}
        for agent_id, agent in self.dacoop_agents.items():
            total_loss, q_loss, kl_loss = agent.update(experiences, self.dacoop_agents)
            losses[agent_id] = {
                'total': total_loss,
                'q': q_loss,
                'kl': kl_loss
            }
                    
        return losses     
       
    def save_models(self, path='models'):
        os.makedirs(path, exist_ok=True)
        for agent_id, agent in self.dacoop_agents.items():
            agent.save(path)
            
    def load_models(self, path='models'):
        for agent_id, agent in self.dacoop_agents.items():
            agent.load(path)

def train_dacoop(env, trainer:DACOOPTrainer, n_episodes=NUM_EPISODES):
    episode_rewards = []
    avg_rewards = []
    predator_agents = [a for a in env.possible_agents if "adversary" in a]
    
    for episode in range(1, n_episodes+1):
        # observations, _ = env.reset(seed=np.random.randint(1000))
        print(f"Episode {episode}/{n_episodes}")
        observations, _ = env.reset(seed=SEED)
        episode_reward = 0
        step_count = 0
            
        done = {agent_id: False for agent_id in env.possible_agents}
        episode_done = False
        
        # 在一个episode中，控制所有agent移动直到任务完成或达到最大步数
        while not episode_done and step_count < MAX_STEPS:
            actions = trainer.get_actions(observations, add_noise=(len(trainer.buffer) < START_TRAINING))
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            
            dones = {agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in env.possible_agents}
            episode_done = any(dones.values())
            
            for agent_id in predator_agents:
                if agent_id in trainer.dacoop_agents:
                    neighbor_obs = []
                    neighbor_next_obs = []
                    
                    for other_id in predator_agents:
                        if other_id != agent_id:
                            neighbor_obs.append(observations[other_id])
                            neighbor_next_obs.append(next_observations[other_id])
                    
                    trainer.buffer.add(
                        observations[agent_id],
                        neighbor_obs,
                        actions[agent_id], 
                        rewards[agent_id],
                        next_observations[agent_id],
                        neighbor_next_obs,
                        dones[agent_id]
                    )
            
            if len(trainer.buffer) >= BATCH_SIZE and step_count % UPDATE_EVERY == 0:
                trainer.update()
            observations = next_observations
            
            predator_reward = sum(rewards[agent_id] for agent_id in predator_agents)
            episode_reward += predator_reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
    
        if episode % 100 == 0:
            print(f"Episode {episode}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Buffer: {len(trainer.buffer)}")

            # if episode % 1000 == 0:
            #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            #     save_path = f"models/dacoop_{timestamp}_ep{episode}"
            #     os.makedirs(save_path, exist_ok=True)
            #     trainer.save_models(save_path)

            #     plt.figure(figsize=(10, 6))
            #     plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
            #     plt.plot(avg_rewards, label='Avg Reward (100 ep)', linewidth=2)
            #     plt.xlabel('Episode')
            #     plt.ylabel('Reward')
            #     plt.title('DACOOP Training Progress')
            #     plt.legend()
            #     plt.grid(True, alpha=0.3)
            #     plt.savefig(f"{save_path}/learning_curve.png")
            #     plt.close()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = f"models/dacoop_{timestamp}_final"
    os.makedirs(final_path, exist_ok=True)
    trainer.save_models(final_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    plt.plot(avg_rewards, label='Avg Reward (100 ep)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DACOOP Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{final_path}/learning_curve.png")
    plt.close()
    
    return episode_rewards, avg_rewards, final_path

def evaluate_dacoop(env, trainer, n_episodes=10, render=False):
    """Evaluate trained DACOOP agents"""
    evaluation_rewards = []
    predator_agents = [a for a in env.possible_agents if "adversary" in a]
    
    for episode in range(n_episodes):
        # observations, _ = env.reset(seed=episode)
        observations, _ = env.reset(seed=SEED)
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            # Get actions without exploration noise
            actions = trainer.get_actions(observations, add_noise=False)
            
            # Step environment
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            dones = {a: terminations[a] or truncations[a] for a in env.possible_agents}
            done = any(dones.values())
            
            # Render if requested
            if render:
                env.render()
                
            # Update for next step
            observations = next_observations
            predator_reward = sum(rewards[agent_id] for agent_id in predator_agents)
            episode_reward += predator_reward
            step += 1
            
        evaluation_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(evaluation_rewards)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    return evaluation_rewards, avg_reward

if __name__ == "__main__":
    env = make_env()
    trainer = DACOOPTrainer(env, predator_only=True)
    episode_rewards, avg_rewards, model_path = train_dacoop(env, trainer)
    eval_env = make_env(render_mode="human")
    eval_rewards, avg_eval_reward = evaluate_dacoop(eval_env, trainer)