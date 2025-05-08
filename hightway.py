# 安装必要的依赖
# !pip install torch numpy matplotlib gym pettingzoo[all] stable-baselines3 highway-env supersuit

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import gymnasium as gym
import pettingzoo
from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3
from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.conversions import parallel_wrapper_fn
import supersuit as ss
import highway_env
from gymnasium.wrappers import FlattenObservation

# 设置随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, full_state_dim, full_action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(full_state_dim + full_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义MADDPG智能体
class MADDPGAgent:
    def __init__(self, agent_id, state_dim, action_dim, n_agents, hidden_dim=256, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01, max_action=1.0):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        
        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # 初始化Critic网络（使用全局信息）
        full_state_dim = state_dim * n_agents
        full_action_dim = action_dim * n_agents
        self.critic = Critic(full_state_dim, full_action_dim, hidden_dim).to(device)
        self.critic_target = Critic(full_state_dim, full_action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def select_action(self, state, add_noise=True, noise_scale=0.1):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        
        if add_noise:
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            action = np.clip(action, -self.actor.max_action, self.actor.max_action)
            
        return action
    
    def update(self, memory, batch_size, n_agents, agents):
        # 从经验回放中采样
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        
        # 为当前智能体提取状态和动作
        agent_state = states[:, self.agent_id]
        agent_action = actions[:, self.agent_id]
        agent_reward = rewards[:, self.agent_id]
        agent_next_state = next_states[:, self.agent_id]
        agent_done = dones[:, self.agent_id]
        
        # 计算当前Q值
        current_Q = self.critic(states.reshape(batch_size, -1), actions.reshape(batch_size, -1))
        
        # 计算目标动作
        next_actions = torch.zeros(batch_size, n_agents, self.action_dim).to(device)
        for i, agent in enumerate(agents):
            next_agent_state = next_states[:, i]
            next_actions[:, i] = agent.actor_target(next_agent_state)
        
        # 计算目标Q值
        target_Q = self.critic_target(next_states.reshape(batch_size, -1), next_actions.reshape(batch_size, -1))
        target_Q = agent_reward.reshape(-1, 1) + (1 - agent_done.reshape(-1, 1)) * self.gamma * target_Q
        
        # 更新Critic
        critic_loss = F.mse_loss(current_Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        predicted_actions = torch.zeros(batch_size, n_agents, self.action_dim).to(device)
        for i, agent in enumerate(agents):
            if i == self.agent_id:
                predicted_actions[:, i] = self.actor(agent_state)
            else:
                predicted_actions[:, i] = actions[:, i]
        
        actor_loss = -self.critic(states.reshape(batch_size, -1), predicted_actions.reshape(batch_size, -1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f'{directory}/{name}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{name}_critic.pth')
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f'{directory}/{name}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{name}_critic.pth'))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, n_agents):
        self.capacity = capacity
        self.counter = 0
        
        self.states = torch.zeros((capacity, n_agents, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((capacity, n_agents, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((capacity, n_agents, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((capacity, n_agents, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((capacity, n_agents, 1), dtype=torch.float32).to(device)
        
    def add(self, state, action, reward, next_state, done):
        idx = self.counter % self.capacity
        
        self.states[idx] = torch.FloatTensor(state).to(device)
        self.actions[idx] = torch.FloatTensor(action).to(device)
        self.rewards[idx] = torch.FloatTensor(reward).reshape(self.rewards[idx].shape).to(device)
        self.next_states[idx] = torch.FloatTensor(next_state).to(device)
        self.dones[idx] = torch.FloatTensor(done).reshape(self.dones[idx].shape).to(device)
        
        self.counter += 1
        
    def sample(self, batch_size):
        max_idx = min(self.counter, self.capacity)
        indices = torch.randint(0, max_idx, (batch_size,))
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return min(self.counter, self.capacity)

# 环境适配器
class MAHighwayEnvAdapter:
    def __init__(self, num_agents=4):
        # 创建Highway环境
        env = gym.make('highway-v0')
        env.configure({
            'observation': {
                'type': 'Kinematics',
                'vehicles_count': 6,
                'features': ['presence', 'x', 'y', 'vx', 'vy', 'heading'],
            },
            'action': {
                'type': 'ContinuousAction'
            },
            'simulation_frequency': 15,
            'policy_frequency': 5,
            'lanes_count': 3,
            'vehicles_count': num_agents + 5,  # 多几辆后台车
            'controlled_vehicles': num_agents,
            'initial_lane_id': None,
            'duration': 50,  # 单次episode时长
            'vehicles_density': 1.5,
        })
        env.reset()
        self.env = env
        
        # 处理观察空间和动作空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6 * 6,)  # 每个车观察6辆车，每辆车6个特征
        )
        
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,)  # 转向和加速度
        )
        
        self.num_agents = num_agents
        self.agents = [f"vehicle_{i}" for i in range(num_agents)]
        
    def reset(self):
        obs = self.env.reset()[0]
        
        # 将obs重组为每个智能体的观察
        agent_observations = []
        
        for i in range(self.num_agents):
            agent_obs = obs.reshape(6, 6)  # 重新整形为6辆车×6特征
            agent_observations.append(agent_obs.flatten())
            
        return agent_observations
    
    def step(self, actions):
        # 将多个智能体的动作合并为一个动作列表
        combined_action = np.array(actions)
        
        # 执行步骤
        obs, reward, terminated, truncated, info = self.env.step(combined_action)
        
        # 将单一奖励分解为每个智能体的奖励
        rewards = []
        # 这里我们可以基于车辆位置、速度等为每个智能体分配不同的奖励
        # 简单起见，我们先平均分配总奖励
        individual_reward = reward / self.num_agents
        rewards = [individual_reward] * self.num_agents
        
        # 将obs重组为每个智能体的观察
        agent_observations = []
        agent_dones = []
        
        for i in range(self.num_agents):
            agent_obs = obs.reshape(6, 6)  # 重新整形为6辆车×6特征
            agent_observations.append(agent_obs.flatten())
            agent_dones.append(terminated or truncated)
            
        return agent_observations, rewards, agent_dones, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

# 训练MADDPG
def train_maddpg(env_fn, n_episodes=20000, max_t=100, print_every=100, 
                 batch_size=128, buffer_size=1e6, gamma=0.99, tau=0.01,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=0):
    
    # 创建环境
    env = env_fn()
    n_agents = env.num_agents
    
    # 获取状态维度和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化智能体
    agents = []
    for i in range(n_agents):
        agent = MADDPGAgent(i, state_dim, action_dim, n_agents, 
                           hidden_dim=256, 
                           lr_actor=lr_actor,
                           lr_critic=lr_critic,
                           gamma=gamma,
                           tau=tau)
        agents.append(agent)
    
    # 初始化经验回放缓冲区
    memory = ReplayBuffer(int(buffer_size), state_dim, action_dim, n_agents)
    
    # 初始化噪声过程
    noise_scale = 0.3
    noise_reduction = 0.9999
    min_noise_scale = 0.01
    
    # 统计结果
    rewards_history = []
    avg_rewards_history = []
    
    # 训练循环
    for episode in range(1, n_episodes+1):
        states = env.reset()
        states = np.stack(states)
        episode_reward = np.zeros(n_agents)
        
        for t in range(max_t):
            # 每个智能体选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[i], add_noise=True, noise_scale=noise_scale)
                actions.append(action)
            
            # 执行动作
            next_states, rewards, dones, _ = env.step(actions)
            next_states = np.stack(next_states)
            dones = np.array(dones).astype(np.float32)
            rewards = np.array(rewards).reshape(n_agents, 1)
            
            # 存储经验
            memory.add(states, actions, rewards, next_states, dones)
            
            # 更新状态和累积奖励
            states = next_states
            episode_reward += rewards.squeeze()
            
            # 如果有足够的经验，训练智能体
            if len(memory) > batch_size:
                for agent_idx, agent in enumerate(agents):
                    critic_loss, actor_loss = agent.update(memory, batch_size, n_agents, agents)
            
            # 检查是否所有智能体都完成
            if np.any(dones):
                break
        
        # 减少噪声
        noise_scale = max(min_noise_scale, noise_scale * noise_reduction)
        
        # 记录奖励
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-print_every:], axis=0)
        avg_rewards_history.append(avg_reward)
        
        # 打印进度
        if episode % print_every == 0:
            print(f'Episode {episode}\tAverage Reward: {avg_reward.mean():.3f}\tNoise Scale: {noise_scale:.3f}')
            
    # 保存模型
    for i, agent in enumerate(agents):
        agent.save("./models", f"maddpg_agent_{i}")
    
    # 关闭环境
    env.close()
    
    return agents, rewards_history, avg_rewards_history

# 测试训练好的智能体
def test_agents(env_fn, agents, n_episodes=3, max_t=1000):
    env = env_fn()
    
    for episode in range(1, n_episodes+1):
        states = env.reset()
        states = np.stack(states)
        episode_reward = np.zeros(len(agents))
        
        for t in range(max_t):
            # 渲染环境
            env.render()
            
            # 每个智能体选择动作（去中心化执行）
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[i], add_noise=False)
                actions.append(action)
            
            # 执行动作
            next_states, rewards, dones, _ = env.step(actions)
            next_states = np.stack(next_states)
            rewards = np.array(rewards).reshape(len(agents), 1)
            
            # 更新状态和累积奖励
            states = next_states
            episode_reward += rewards.squeeze()
            
            # 检查是否所有智能体都完成
            if np.any(dones):
                break
        
        print(f'Episode {episode}\tReward: {episode_reward}\tAverage: {episode_reward.mean():.3f}')
    
    env.close()

# 绘制奖励曲线
def plot_rewards(rewards_history, avg_rewards_history, n_agents):
    plt.figure(figsize=(12, 8))
    
    # 绘制每个智能体的平均奖励
    for i in range(n_agents):
        agent_rewards = [r[i] for r in avg_rewards_history]
        plt.plot(agent_rewards, label=f'Agent {i}')
    
    # 绘制所有智能体的平均奖励
    mean_rewards = [np.mean(r) for r in avg_rewards_history]
    plt.plot(mean_rewards, 'k-', linewidth=2, label='All Agents')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('MADDPG Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('maddpg_training_rewards.png')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 训练智能体
    print("Training MADDPG agents...")
    agents, rewards_history, avg_rewards_history = train_maddpg(MAHighwayEnvAdapter)
    
    # 绘制结果
    plot_rewards(rewards_history, avg_rewards_history, len(agents))
    
    # 测试智能体
    print("\nTesting MADDPG agents...")
    test_agents(MAHighwayEnvAdapter, agents)