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
import imageio
import datetime
# 环境变量设置，解决X11问题
import os
os.environ['SDL_VIDEODRIVER'] = 'x11'  # 使用X11驱动
os.environ['DISPLAY'] = ':0'          # 确保正确的显示设置

# 如果仍有问题，尝试使用下面的设置
# os.environ['QT_X11_NO_MITSHM'] = '1'
# 设置随机种子
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
# 在代码开头添加
import matplotlib
matplotlib.use('Agg') 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK JP', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

# 使用CPU还是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BUFFER_SIZE = 50000    # 经验回放缓冲区大小
BATCH_SIZE = 64          # 批次大小
GAMMA = 0.95             # 折扣因子
TAU = 0.005               # 目标网络软更新系数
LR_ACTOR = 0.0001        # Actor学习率
LR_CRITIC = 0.001        # Critic学习率
HIDDEN_SIZE = 64         # 隐藏层大小
NUM_EPISODES = 20000     # 训练回合数
MAX_STEPS = 100          # 每回合最大步数
START_TRAINING = 200     # 开始训练前收集的步数
UPDATE_EVERY = 50       # 更新网络的频率
NOISE = 0.3              # 探索噪声

# 创建环境
def make_env(seed = SEED, render_mode=None):
    env = simple_tag_v3.parallel_env(
        num_good=1,              # 1个逃避者
        num_adversaries=3,       # 3个追捕者
        num_obstacles=2,         # 2个障碍物
        max_cycles=MAX_STEPS,    # 最大步数
        continuous_actions=True, # 使用连续动作空间
        dynamic_rescaling=True,
        render_mode=render_mode,
    )
    env.reset(seed=SEED)
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
            nn.Tanh()
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
            action += NOISE * np.random.randn(action.shape[1])
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
    def __init__(self, env, train_prey=True):
        self.env = env
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
        self.train_prey = train_prey
        
        # 区分追捕者和逃避者
        self.predator_agents = [agent for agent in self.agents if "adversary" in agent]
        self.prey_agents = [agent for agent in self.agents if "adversary" not in agent]
        
        # 确定要训练的智能体
        if train_prey:
            self.agents_to_train = self.agents  # 所有智能体都训练
        else:
            self.agents_to_train = self.predator_agents  # 只训练追捕者
        
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
            # 如果是逃避者且不训练，则返回固定动作(不动)
            if agent_id in self.prey_agents and not self.train_prey:
                actions[agent_id] = np.ones(env.action_space(agent_id).shape[0], dtype=np.float32) * 0.5
            else:
                actions[agent_id] = agent.act(np.expand_dims(observations[agent_id], 0), add_noise)[0]
        return actions
    
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        experiences = self.buffer.sample()
        
        # 只更新需要训练的智能体
        for agent_id in self.agents_to_train:
            self.maddpg_agents[agent_id].update(experiences, self.maddpg_agents)
            
    def save(self, path='models'):
        os.makedirs(path, exist_ok=True)
        for agent_id, agent in self.maddpg_agents.items():
            agent.save(path)
            
    def load(self, path='models'):
        for agent_id, agent in self.maddpg_agents.items():
            agent.load(path)
            
    def load_custom(self, predator_paths, prey_path):
        """加载自定义的不同模型作为追捕者和逃避者"""
        predator_agents = [agent for agent in self.agents if "adversary" in agent]
        prey_agents = [agent for agent in self.agents if "adversary" not in agent]
        
        # 加载追捕者模型
        for i, agent_id in enumerate(predator_agents):
            if i < len(predator_paths):
                self.maddpg_agents[agent_id].actor.load_state_dict(torch.load(f'{predator_paths[i]}/actor_{agent_id}.pth'))
                self.maddpg_agents[agent_id].critic.load_state_dict(torch.load(f'{predator_paths[i]}/critic_{agent_id}.pth'))
                self.maddpg_agents[agent_id].hard_update(self.maddpg_agents[agent_id].actor_target, self.maddpg_agents[agent_id].actor)
                self.maddpg_agents[agent_id].hard_update(self.maddpg_agents[agent_id].critic_target, self.maddpg_agents[agent_id].critic)
                print(f"已加载追捕者 {agent_id} 模型: {predator_paths[i]}")
        
        # 加载逃避者模型
        for agent_id in prey_agents:
            self.maddpg_agents[agent_id].actor.load_state_dict(torch.load(f'{prey_path}/actor_{agent_id}.pth'))
            self.maddpg_agents[agent_id].critic.load_state_dict(torch.load(f'{prey_path}/critic_{agent_id}.pth'))
            self.maddpg_agents[agent_id].hard_update(self.maddpg_agents[agent_id].actor_target, self.maddpg_agents[agent_id].actor)
            self.maddpg_agents[agent_id].hard_update(self.maddpg_agents[agent_id].critic_target, self.maddpg_agents[agent_id].critic)
            print(f"已加载逃避者 {agent_id} 模型: {prey_path}")

def cal_rewards(rewards, observations, next_observations, actions, predator_agents, train_prey):
    # 获取位置信息
    agent_origin_positions = {agent: observations[agent][2:4] for agent in rewards.keys()}
    agent_velocities = {agent: observations[agent][4:6] for agent in rewards.keys()}  # 获取速度信息
    predator_next_positions = {agent: next_observations[agent][2:4] for agent in predator_agents}
    
    prey_agents = [agent for agent in rewards.keys() if agent not in predator_agents]
    prey_position = {agent: next_observations[agent][2:4] for agent in prey_agents}
    prey_velocity = {agent: next_observations[agent][4:6] for agent in prey_agents}  # 获取猎物速度
    prey_agent = prey_agents[0]
    prey_pos = prey_position[prey_agent]
    prey_vel = prey_velocity[prey_agent]
    
    # 计算距离
    current_distances = {agent: np.sqrt(np.sum((predator_next_positions[agent] - prey_pos)**2)) for agent in predator_agents}
    prev_distances = {agent: np.sqrt(np.sum((agent_origin_positions[agent] - prey_pos)**2)) for agent in predator_agents}
    
    # 计算追捕者相互之间的距离
    predator_distances = {}
    for i, agent_i in enumerate(predator_agents):
        for j, agent_j in enumerate(predator_agents):
            if i < j:
                dist = np.sqrt(np.sum((predator_next_positions[agent_i] - predator_next_positions[agent_j])**2))
                predator_distances[(agent_i, agent_j)] = dist
    
    # 计算追捕者到猎物的方向向量
    directions_to_prey = {agent: (prey_pos - agent_origin_positions[agent]) for agent in predator_agents}
    normalized_directions = {}
    for agent, direction in directions_to_prey.items():
        magnitude = np.sqrt(np.sum(direction**2)) + 1e-10
        normalized_directions[agent] = direction / magnitude
    
    # 预测猎物下一步位置
    prey_next_predicted = prey_pos + prey_vel * 0.8  # 假设猎物继续按当前速度移动
    
    target_rewards = {}
    
    for agent in predator_agents:
        env_reward = rewards[agent] * 0.2
        
        dis = current_distances[agent]
        # 接近时奖励迅速增加
        dis_reward = 15 * np.exp(-2 * dis) - 2
        
        # 距离变化奖励
        dis_change = prev_distances[agent] - current_distances[agent]
        dis_change_reward = dis_change * 50  # 增大系数，强化接近行为
        
        # 大幅增强向猎物方向移动的奖励
        action_vec = predator_next_positions[agent] - agent_origin_positions[agent]
        action_mag = np.sqrt(np.sum(action_vec**2) + 1e-10)
        
        # 计算行动方向与理想方向的一致性
        if action_mag > 1e-6:
            # 朝向当前猎物位置的奖励
            current_alignment = np.sum(action_vec * normalized_directions[agent]) / action_mag
            
            # 朝向预测的猎物位置的奖励
            pred_direction = prey_next_predicted - agent_origin_positions[agent]
            pred_direction_norm = pred_direction / (np.sqrt(np.sum(pred_direction**2)) + 1e-10)
            prediction_alignment = np.sum(action_vec * pred_direction_norm) / action_mag
            
            # 综合当前位置和预测位置的移动奖励
            movement_reward = (current_alignment * 0.7 + prediction_alignment * 0.3) * 12 * min(action_mag, 1.0)
        else:
            # 不移动的惩罚
            movement_reward = -5.0
        
        # 协作奖励
        collaboration_reward = 0
        optimal_distance = 0.4
        
        for other_agent in predator_agents:
            if agent != other_agent:
                pair = tuple(sorted([agent, other_agent]))
                if pair in predator_distances:
                    dist = predator_distances[pair]
                    # 使用高斯函数形式，确保在最佳距离处获得最高奖励
                    collaboration_reward += 4 * np.exp(-6 * (dist - optimal_distance)**2)
        
        # 围捕策略奖励
        # 计算追捕者与猎物之间形成的角度分布
        angles = []
        for other_agent in predator_agents:
            if other_agent != agent:
                vec1 = normalized_directions[agent]
                vec2 = normalized_directions[other_agent]
                cos_angle = np.sum(vec1 * vec2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        # 理想情况：追捕者均匀分布在猎物周围
        if len(angles) >= 2:
            ideal_angle = np.pi * 2 / 3
            angle_diff = np.abs(np.mean(angles) - ideal_angle)
            encirclement_reward = 5 * np.exp(-3 * angle_diff)
        else:
            encirclement_reward = 0
        
        total_reward = env_reward + dis_reward + dis_change_reward + movement_reward + collaboration_reward + encirclement_reward
        
        # print(f"Agent {agent}: env={env_reward:.2f}, dis={dis_reward:.2f}, change={dis_change_reward:.2f}, " 
        #       f"move={movement_reward:.2f}, collab={collaboration_reward:.2f}, encircle={encirclement_reward:.2f}, "
        #       f"total={total_reward:.2f}")
        
        target_rewards[agent] = total_reward
    
    # 逃避者奖励计算
    for agent in prey_agents:
        if train_prey:
            base_reward = rewards[agent] * 0.5
            
            avg_distance = sum(current_distances.values()) / len(current_distances)
            distance_reward = avg_distance * 8
            
            move_mag = np.sqrt(np.sum(prey_vel**2))
            movement_reward = min(move_mag * 5, 10)
            
            # 逃跑方向奖励
            nearest_predator = min(current_distances, key=current_distances.get)
            escape_direction = prey_pos - predator_next_positions[nearest_predator]
            escape_dir_norm = escape_direction / (np.sqrt(np.sum(escape_direction**2)) + 1e-10)
            escape_alignment = np.sum(prey_vel * escape_dir_norm) / (move_mag + 1e-10) if move_mag > 1e-6 else 0
            escape_reward = escape_alignment * 10
            
            target_rewards[agent] = base_reward + distance_reward + movement_reward + escape_reward
        else:
            target_rewards[agent] = 0.0
            
    return target_rewards
def train_maddpg(env, maddpg: MADDPG, n_episodes=NUM_EPISODES, max_steps=MAX_STEPS, print_every=100):
    total_scores = []
    avg_scores = []
    predator_agents = [agent for agent in env.possible_agents if "adversary" in agent]

    for episode in range(1, n_episodes+1):
        observations, _ = env.reset(seed=SEED)  # 使用固定种子
        episode_score = 0
        
        for step in range(max_steps):
            actions = maddpg.get_actions(observations, add_noise=True)
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            # 处理完成状态
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.possible_agents}

            target_rewards = cal_rewards(rewards, observations, next_observations, actions, predator_agents, maddpg.train_prey)
            # print(target_rewards)
            
            maddpg.buffer.add(observations, actions, target_rewards, next_observations, dones)
            
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

    # 创建以时间命名的保存目录
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "predator_only" if not maddpg.train_prey else "full_model"
    save_dir = f"multi_model/{current_time}_{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 保存训练配置
    with open(f"{save_dir}/train_config.txt", "w") as f:
        f.write(f"Train Prey: {maddpg.train_prey}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Seed: {SEED}\n")

    maddpg.save(save_dir)
    print(f"模型已保存至: {save_dir}")
    
    # 保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(total_scores, alpha=0.3, label='单轮得分')
    plt.plot(avg_scores, label='平均得分', linewidth=2)
    plt.ylabel('得分')
    plt.xlabel('回合')
    plt.title('MADDPG 训练曲线' + (' (仅追捕者)' if not maddpg.train_prey else ''))
    plt.legend()
    plt.savefig(f'{save_dir}/training_curve.png')
    
    return total_scores, avg_scores, save_dir

def evaluate_maddpg(env, maddpg, n_episodes=3, video_fps=10, use_trained_prey=True):
    """
    评估MADDPG智能体
    
    参数:
        env: 评估环境
        maddpg: MADDPG模型
        n_episodes: 评估回合数
        video_fps: 视频帧率
        use_trained_prey: 是否使用训练过的逃避者，False则逃避者保持静止
    """
    predator_agents = [agent for agent in env.possible_agents if "adversary" in agent]
    prey_agents = [agent for agent in env.possible_agents if "adversary" not in agent]
    
    # 创建以时间命名的输出目录
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prey_mode = "trained_prey" if use_trained_prey else "static_prey"
    output_dir = f"eval_results/{current_time}_{prey_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    scores = []
    catch_count = 0
    catch_steps = []
    
    print(f"\n===== 开始评估 MADDPG 智能体 =====")
    print(f"逃避者模式: {'使用训练模型' if use_trained_prey else '保持静止'}")
    print(f"结果将保存至: {output_dir}")
    
    try:
        for episode in range(n_episodes):
            observations, _ = env.reset(seed=SEED)  # 使用固定种子
            episode_score = 0
            frames = []
            
            print(f"\n评估回合 {episode+1}/{n_episodes}")
            
            for step in range(MAX_STEPS):
                try:
                    # 渲染并保存画面
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    print(f"渲染警告: {e}")
                
                # 执行动作
                actions = {}
                for agent_id, agent in maddpg.maddpg_agents.items():
                    # 如果是逃避者且设置为不使用训练模型，则保持静止
                    if agent_id in prey_agents and not use_trained_prey:
                        actions[agent_id] = np.ones(env.action_space(agent_id).shape[0], dtype=np.float32) * 0.5
                    else:
                        actions[agent_id] = agent.act(np.expand_dims(observations[agent_id], 0), add_noise=False)[0]
                
                next_observations, rewards, terminations, truncations, _ = env.step(actions)
                # print(next_observations)
                # 处理完成状态
                dones = {agent: terminations[agent] or truncations[agent] for agent in env.possible_agents}
                
                # 更新状态
                observations = next_observations
                
                # 计算回合得分
                predator_score = sum(rewards[agent] for agent in predator_agents)
                episode_score += predator_score
                
                # 显示实时信息
                if step % 5 == 0:
                    print(f"\r步数: {step+1}/{MAX_STEPS} | 当前得分: {episode_score:.2f}", end="", flush=True)
                
                # 判断是否追捕成功
                if any(terminations.values()) and not any(truncations.values()):
                    catch_count += 1
                    catch_steps.append(step + 1)
                    print(f"\n追捕成功! 用时 {step+1} 步")
                
                # 检查回合是否结束
                if any(dones.values()):
                    break
            
            if step == MAX_STEPS - 1:
                print("\n达到最大步数，追捕未成功")
            
            scores.append(episode_score)
            print(f"\n回合 {episode+1} 得分: {episode_score:.2f}")
            
            # 保存视频
            if frames:
                video_path = f"{output_dir}/episode_{episode+1}.mp4"
                try:
                    imageio.mimsave(video_path, frames, fps=video_fps)
                    print(f"视频已保存: {video_path}")
                except Exception as e:
                    print(f"视频保存错误: {e}")
    finally:
        # 确保环境正确关闭
        try:
            env.close()
        except:
            pass
    
    # 计算统计数据
    avg_score = np.mean(scores)
    catch_rate = catch_count / n_episodes * 100
    avg_catch_steps = np.mean(catch_steps) if catch_steps else 0
    
    # 输出总结
    print("\n===== 评估结果汇总 =====")
    print(f"回合数: {n_episodes}")
    print(f"平均得分: {avg_score:.2f}")
    print(f"追捕成功率: {catch_rate:.1f}%")
    if catch_steps:
        print(f"平均追捕时间: {avg_catch_steps:.2f} 步")
    
    # 保存评估结果图（使用安全的绘图方式）
    save_evaluation_plot(output_dir, scores, avg_score)
    
    return output_dir# 单独的函数用于安全地绘制和保存评估结果图表

def save_evaluation_plot(output_dir, scores, avg_score):
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(scores)+1), scores)
        plt.axhline(y=avg_score, color='r', linestyle='-', label=f'平均分: {avg_score:.2f}')
        plt.xlabel('回合')
        plt.ylabel('得分')
        plt.title('评估结果')
        plt.legend()
        plt.savefig(f'{output_dir}/evaluation_scores.png')
        plt.close()  # 确保关闭图表
    except Exception as e:
        print(f"图表保存错误: {e}")

# 列出可用的模型文件夹
def list_model_folders():
    base_dir = "multi_model"
    if not os.path.exists(base_dir):
        print(f"错误: {base_dir} 目录不存在，请先训练模型")
        return []
    
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort(reverse=True)  # 按时间倒序排列，最新的在前面
    
    if not folders:
        print(f"错误: 在 {base_dir} 目录中未找到模型文件夹")
        return []
    
    print("\n可用的模型文件夹:")
    for i, folder in enumerate(folders):
        print(f"{i+1}. {folder}")
    
    return folders

if __name__ == "__main__":
    os.makedirs('multi_model', exist_ok=True)
    os.makedirs('eval_results', exist_ok=True)
    
    env = make_env(seed=SEED)

    print("\n===== MADDPG 多智能体追逃环境 =====")
    print("1. 训练模式 - 全部智能体 (t)")
    print("2. 训练模式 - 仅追捕者 (tp)")
    print("3. 评估模式 - 同一组模型 (e)")
    print("4. 评估模式 - 自定义组合 (c)")
    
    option = input("\n请选择模式 (t/tp/e/c): ").lower().strip()
    
    if option.startswith('t'):
        print("\n===== 训练模式 =====")
        n_episodes = int(input("训练回合数 (默认5000): ") or "5000")
        
        # 确定是否训练逃避者
        train_prey = True
        if option == "tp":
            train_prey = False
            print("仅训练追捕者，逃避者将保持静止")
        else:
            print("训练所有智能体")
        
        # 创建MADDPG实例，指定是否训练逃避者
        maddpg = MADDPG(env, train_prey=train_prey)
        
        print(f"\n开始训练 {n_episodes} 回合...")
        scores, avg_scores, save_dir = train_maddpg(env, maddpg, n_episodes=n_episodes)
        
        print("\n训练完成! 是否要评估训练好的模型?")
        evaluate_option = input("是否评估 (y/n): ").lower().strip()
        
        if evaluate_option.startswith('y'):
            eval_env = make_env(render_mode="human", seed=SEED)
            n_eval_episodes = int(input("评估回合数 (默认3): ") or "3")
            evaluate_maddpg(eval_env, maddpg, n_episodes=n_eval_episodes)
            eval_env.close()

    elif option.startswith('e'):
        print("\n===== 评估模式 - 使用同一组模型 =====")
        folders = list_model_folders()
        if not folders:
            exit(1)

        selected = int(input("\n请选择要评估的模型 (输入序号): ")) - 1
        if selected < 0 or selected >= len(folders):
            print("无效的选择!")
            exit(1)
        
        selected_model = f"multi_model/{folders[selected]}"
        print(f"选择模型: {selected_model}")
        
        # 询问是否使用训练过的逃避者
        use_trained_prey = input("\n是否使用训练过的逃避者? (y/n): ").lower().strip().startswith('y')
        if use_trained_prey:
            print("逃避者将使用训练模型")
        else:
            print("逃避者将保持静止")
        
        # 创建并加载模型
        eval_env = make_env(render_mode="human", seed=SEED)
        maddpg = MADDPG(eval_env, train_prey=True)  # 创建完整模型
        maddpg.load(selected_model)                # 加载模型参数
        
        # 评估模型
        n_eval_episodes = int(input("评估回合数 (默认3): ") or "3")
        evaluate_maddpg(eval_env, maddpg, n_episodes=n_eval_episodes, use_trained_prey=use_trained_prey)
    elif option.startswith('c'):
        print("\n===== 评估模式 - 自定义组合 =====")
        folders = list_model_folders()
        if not folders:
            exit(1)

        print("\n为3个追捕者选择模型 (adversary_0, adversary_1, adversary_2):")
        predator_models = []
        for i in range(3):
            selected = int(input(f"请为追捕者 {i} 选择模型 (输入序号): ")) - 1
            if selected < 0 or selected >= len(folders):
                print("无效的选择!")
                exit(1)
            predator_models.append(f"multi_model/{folders[selected]}")
        
        selected = int(input("\n请为逃避者 (agent_0) 选择模型 (输入序号): ")) - 1
        if selected < 0 or selected >= len(folders):
            print("无效的选择!")
            exit(1)
        prey_model = f"multi_model/{folders[selected]}"
        
        # 询问是否使用训练过的逃避者
        use_trained_prey = input("\n是否使用训练过的逃避者? (y/n): ").lower().strip().startswith('y')
        if use_trained_prey:
            print("逃避者将使用训练模型")
        else:
            print("逃避者将保持静止")
        
        print("\n你的选择:")
        print(f"追捕者模型: {predator_models}")
        print(f"逃避者模型: {prey_model}")
        print(f"逃避者行为: {'使用训练模型' if use_trained_prey else '保持静止'}")
        
        confirm = input("确认? (y/n): ").lower().strip()
        if not confirm.startswith('y'):
            print("已取消")
            exit(0)
        
        # 创建并加载模型
        eval_env = make_env(render_mode="human", seed=SEED)
        maddpg = MADDPG(eval_env, train_prey=True)  # 创建完整模型
        maddpg.load_custom(predator_models, prey_model)
        
        # 评估模型
        n_eval_episodes = int(input("评估回合数 (默认3): ") or "3")
        evaluate_maddpg(eval_env, maddpg, n_episodes=n_eval_episodes, use_trained_prey=use_trained_prey)        
        eval_env.close()
    
    else:
        print("无效的选择!")
    env.close()
    print("\n程序执行完毕!")