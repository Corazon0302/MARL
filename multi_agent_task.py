import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mpe2 import simple_tag_v3
import supersuit as ss
import random
from collections import deque
import time

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 使用CPU还是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BUFFER_SIZE = 100000     # 经验回放缓冲区大小
BATCH_SIZE = 64          # 批次大小
GAMMA = 0.95             # 折扣因子
TAU = 0.01               # 目标网络软更新系数
LR_ACTOR = 0.0001        # Actor学习率
LR_CRITIC = 0.001        # Critic学习率
HIDDEN_SIZE = 64         # 隐藏层大小
NUM_EPISODES = 20000     # 训练回合数
MAX_STEPS = 25           # 每回合最大步数
START_TRAINING = 1000    # 开始训练前收集的步数
UPDATE_EVERY = 100       # 更新网络的频率
NOISE = 0.2              # 探索噪声

# 创建环境
def make_env(render_mode=None):
    env = simple_tag_v3.parallel_env(
        num_good=1,              # 1个逃避者
        num_adversaries=3,       # 3个追捕者
        num_obstacles=2,         # 2个障碍物
        max_cycles=MAX_STEPS,    # 最大步数
        continuous_actions=True, # 使用连续动作空间
        render_mode=render_mode
    )
    # 对观察空间进行归一化
    env = ss.observation_lambda_v0(env, lambda obs, obs_space, agent: obs / 10.0)
    return env

# Actor网络
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_size),
            nn.Sigmoid()
        )
        
    def forward(self, state):
        return self.sequential(state)

# Critic网络
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(state_size + action_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
            nn.ReLU()
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.sequential(x)

# MADDPG智能体
class MADDPGAgent:
    def __init__(self, agent_id, obs_size, act_size, num_agents):
        self.agent_id = agent_id
        self.obs_size = obs_size
        self.act_size = act_size
        
        # 全局状态和动作尺寸
        self.global_obs_size = sum(obs_size.values())
        self.global_act_size = sum(act_size.values())
        
        # Actor网络 (局部观察 -> 动作)
        self.actor = Actor(obs_size[agent_id], act_size[agent_id]).to(device)
        self.actor_target = Actor(obs_size[agent_id], act_size[agent_id]).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        # Critic网络 (全局状态和动作 -> Q值)
        self.critic = Critic(self.global_obs_size, self.global_act_size).to(device)
        self.critic_target = Critic(self.global_obs_size, self.global_act_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # 初始化目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
    
    def act(self, obs, add_noise=True):
        obs = torch.FloatTensor(obs).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        self.actor.train()
        
        if add_noise:
            action += NOISE * np.random.randn(action.shape[0])
        return np.clip(action, 0, 1)
    
    def update(self, experiences, all_agents):
        obs, actions, rewards, next_obs, dones = experiences

        all_obs = torch.cat([obs[agent] for agent in sorted(obs.keys())], dim=1)
        all_actions = torch.cat([actions[agent] for agent in sorted(actions.keys())], dim=1)
        all_next_obs = torch.cat([next_obs[agent] for agent in sorted(next_obs.keys())], dim=1)
        
        # 计算下一个状态的目标动作
        next_actions = []
        for agent_id, agent in all_agents.items():
            next_act = agent.actor_target(next_obs[agent_id])
            next_actions.append(next_act)
        all_next_actions = torch.cat(next_actions, dim=1)
        
        # 计算目标Q值
        with torch.no_grad():
            q_next = self.critic_target(all_next_obs, all_next_actions)
            q_target = rewards[self.agent_id] + GAMMA * q_next * (1 - dones[self.agent_id])
            
        # 更新Critic
        # Loss(θ_i) = E[(y_i - Q_i(x, a_1, ..., a_N | θ_i))^2]
        # y_i = r_i + γ * Q'_i(x', a'_1, ..., a'_N | θ'_i) * (1 - done_i) 并且 a'_j = μ'_j(o'_j | φ'_j) 是所有智能体 j 在下一个状态 x' 下由其目标 Actor 网络选择的动作
        q_current = self.critic(all_obs, all_actions)
        critic_loss = F.mse_loss(q_current, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        
        # 更新Actor
        pred_actions = []
        for agent_id, agent in all_agents.items():
            if agent_id == self.agent_id:
                pred_act = self.actor(obs[agent_id])
            else:
                pred_act = agent.actor(obs[agent_id]).detach()
            pred_actions.append(pred_act)
            
        all_pred_actions = torch.cat(pred_actions, dim=1)
        actor_loss = -self.critic(all_obs, all_pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)
        
        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """软更新模型参数: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def hard_update(self, target_model, local_model):
        """硬更新模型参数: θ_target = θ_local"""
        target_model.load_state_dict(local_model.state_dict())
        
    def save(self, path):
        torch.save(self.actor.state_dict(), f'{path}/actor_{self.agent_id}.pth')
        torch.save(self.critic.state_dict(), f'{path}/critic_{self.agent_id}.pth')
        
    def load(self, path):
        self.actor.load_state_dict(torch.load(f'{path}/actor_{self.agent_id}.pth'))
        self.critic.load_state_dict(torch.load(f'{path}/critic_{self.agent_id}.pth'))
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)

        states = {agent: [] for agent in batch[0][0].keys()}
        actions = {agent: [] for agent in batch[0][1].keys()}
        rewards = {agent: [] for agent in batch[0][2].keys()}
        next_states = {agent: [] for agent in batch[0][3].keys()}
        dones = {agent: [] for agent in batch[0][4].keys()}
        
        for state, action, reward, next_state, done in batch:
            for agent in state.keys():
                states[agent].append(state[agent])
                actions[agent].append(action[agent])
                rewards[agent].append([reward[agent]])
                next_states[agent].append(next_state[agent])
                dones[agent].append([float(done[agent])])

        states = {agent: torch.FloatTensor(np.vstack(states[agent])).to(device) for agent in states.keys()}
        actions = {agent: torch.FloatTensor(np.vstack(actions[agent])).to(device) for agent in actions.keys()}
        rewards = {agent: torch.FloatTensor(np.vstack(rewards[agent])).to(device) for agent in rewards.keys()}
        next_states = {agent: torch.FloatTensor(np.vstack(next_states[agent])).to(device) for agent in next_states.keys()}
        dones = {agent: torch.FloatTensor(np.vstack(dones[agent])).to(device) for agent in dones.keys()}
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

class MADDPG:
    def __init__(self, env):
        self.env = env
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
        
        # 获取观察和动作空间
        obs_size = {}
        act_size = {}
        for agent in self.agents:
            obs_size[agent] = env.observation_space(agent).shape[0]
            act_size[agent] = env.action_space(agent).shape[0]
        
        self.maddpg_agents = {}
        for agent_id in self.agents:
            self.maddpg_agents[agent_id] = MADDPGAgent(agent_id, obs_size, act_size, self.num_agents)
            
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
    def get_actions(self, observations, add_noise=True):
        actions = {}
        for agent_id, agent in self.maddpg_agents.items():
            actions[agent_id] = agent.act(np.expand_dims(observations[agent_id], 0), add_noise)[0]
        return actions
    
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        
        experiences = self.buffer.sample()
        
        for agent_id, agent in self.maddpg_agents.items():
            agent.update(experiences, self.maddpg_agents)
            
    def save(self, path='models'):
        os.makedirs(path, exist_ok=True)
        for agent_id, agent in self.maddpg_agents.items():
            agent.save(path)
            
    def load(self, path='models'):
        for agent_id, agent in self.maddpg_agents.items():
            agent.load(path)

def train_maddpg(env, maddpg: MADDPG, n_episodes=NUM_EPISODES, max_steps=MAX_STEPS, print_every=100):
    total_scores = []
    avg_scores = []
    predator_agents = [agent for agent in env.possible_agents if "adversary" in agent]

    for episode in range(1, n_episodes+1):
        observations, _ = env.reset()
        episode_score = 0
        
        for step in range(max_steps):
            actions = maddpg.get_actions(observations, add_noise=True)
            
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # 处理完成状态
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.possible_agents}

            maddpg.buffer.add(observations, actions, rewards, next_observations, dones)
            
            # 如果缓冲区足够大，则更新网络
            if len(maddpg.buffer) > START_TRAINING and step % UPDATE_EVERY == 0:
                maddpg.update()

            observations = next_observations

            predator_score = sum(rewards[agent] for agent in predator_agents)
            episode_score += predator_score

            if any(dones.values()):
                break
                
        # 记录得分
        total_scores.append(episode_score)
        avg_score = np.mean(total_scores[-100:]) if len(total_scores) >= 100 else np.mean(total_scores)
        avg_scores.append(avg_score)

        if episode % print_every == 0:
            print(f'Episode {episode}/{n_episodes} | Average Score: {avg_score:.2f}')
        if episode % 1000 == 0:
            maddpg.save(f'models_episode_{episode}')

    maddpg.save() 
    return total_scores, avg_scores

# 评估函数
def evaluate_maddpg(env, maddpg, n_episodes=10, render=False):
    scores = []
    predator_agents = [agent for agent in env.possible_agents if "adversary" in agent]
    
    for episode in range(n_episodes):
        observations, _ = env.reset()
        episode_score = 0
        
        for step in range(MAX_STEPS):
            if render:
                env.render()
                time.sleep(0.1)
                
            actions = maddpg.get_actions(observations, add_noise=False)
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # 处理完成状态
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.possible_agents}
            # 更新状态
            observations = next_observations
            
            # 更新得分
            predator_score = sum(rewards[agent] for agent in predator_agents)
            episode_score += predator_score
            
            # 如果回合结束，跳出循环
            if any(dones.values()):
                break

        scores.append(episode_score)
        print(f'Evaluation Episode {episode+1}: Score = {episode_score:.2f}')
    avg_score = np.mean(scores)
    print(f'Evaluation over {n_episodes} episodes: Average Score = {avg_score:.2f}')
    
    return avg_score

def plot_scores(scores, avg_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label='Score')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label='Average Score')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('MADDPG Training Progress')
    plt.legend()
    plt.savefig('maddpg_training.png')
    plt.show()

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    env = make_env()
    maddpg = MADDPG(env)

    train_mode = input("Train (t) or Evaluate (e)? ").lower().startswith('t')
    
    if train_mode:
        print("Starting training...")
        # n_episodes = int(input("Number of episodes (default 20000): ") or "20000")
        n_episodes = 20000
        scores, avg_scores = train_maddpg(env, maddpg, n_episodes=n_episodes)
        
        plot_scores(scores, avg_scores)
        print("\nEvaluating trained agents...")
        eval_env = make_env(render_mode="human")
        evaluate_maddpg(eval_env, maddpg, n_episodes=3, render=True)
        eval_env.close()
    else:
        print("Loading trained models...")
        maddpg.load()
        print("Evaluating trained agents...")
        eval_env = make_env(render_mode="human")
        evaluate_maddpg(eval_env, maddpg, n_episodes=5, render=True)
        eval_env.close()
    
    # 关闭环境
    env.close()