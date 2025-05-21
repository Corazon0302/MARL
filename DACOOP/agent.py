import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from architecture import *
from utils import *
class DACOOPAgent:
    def __init__(self, agent_id, obs_size, act_size):
        self.agent_id = agent_id
        self.obs_size = obs_size
        self.act_size = ACTION_SIZE
        
        self.embedding_net = EmbeddingNetwork(EMBEDDING_SIZE).to(device)
        self.key_net = KeyNetwork(EMBEDDING_SIZE).to(device)
        self.attention_net = AttentionNetwork(EMBEDDING_SIZE).to(device)
        self.dueling_net = DuelingNetwork(6 + EMBEDDING_SIZE, self.act_size).to(device)
        
        self.target_embedding_net = EmbeddingNetwork(EMBEDDING_SIZE).to(device)
        self.target_key_net = KeyNetwork(EMBEDDING_SIZE).to(device)
        self.target_attention_net = AttentionNetwork(EMBEDDING_SIZE).to(device)
        self.target_dueling_net = DuelingNetwork(6 + EMBEDDING_SIZE, self.act_size).to(device)
        
        self.apf = APFA()        
        self.hard_update_targets()
        all_params = list(self.embedding_net.parameters()) + \
                      list(self.key_net.parameters()) + \
                      list(self.attention_net.parameters()) + \
                      list(self.dueling_net.parameters())
        self.optimizer = optim.Adam(all_params, lr=LR_ACTOR)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    # 通过观察值，计算注意力加权的嵌入特征，完成论文算法A部分的实现
    def process_observation(self, obs):
        evader_dist, evader_angle, obstacle_dist_list, obstacle_angle_list = extract_local_observation(obs)
        local_obs = np.array([evader_dist, evader_angle, obstacle_dist_list[0], obstacle_angle_list[0], obstacle_dist_list[1], obstacle_angle_list[1]], dtype=np.float32) # o_{loc, i}
        local_obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(device)
        
        # Process neighbor information
        neighbor_embeddings = []
        neighbor_keys = []
        neighbor_info = extract_neighbor_info(obs) # d_{j,i}, φ_{j,i}
        
        # Create embeddings for each neighbor
        for i, n_info in enumerate(neighbor_info):
            distance, angle = n_info[0], n_info[1]
            input_arr = np.array([distance, angle], dtype=np.float32)
            n_input = torch.FloatTensor(input_arr).unsqueeze(0).to(device)
            
            n_embedding = self.embedding_net(n_input) # e_{j,i}
            n_key = self.key_net(n_embedding) # k_{j,i}
            
            neighbor_embeddings.append(n_embedding)
            neighbor_keys.append(n_key)
        
        # Calculate mean embedding (equation 8 in the paper) e_{m, i}
        if neighbor_embeddings:
            neighbor_embeddings_tensor = torch.cat(neighbor_embeddings, dim=0) 
            mean_embedding = torch.mean(neighbor_embeddings_tensor, dim=0, keepdim=True)
        else:
            mean_embedding = torch.zeros(1, EMBEDDING_SIZE).to(device)
        
        # Calculate attention scores for each neighbor
        attention_scores_raw = [] # \hat a_{j,i}
        for i, n_key in enumerate(neighbor_keys):
            raw_score = self.attention_net(local_obs_tensor, mean_embedding, n_key)
            attention_scores_raw.append(raw_score)
        
        # Apply softmax to get normalized attention scores (equation 10)
        if attention_scores_raw:
            attention_scores_tensor = torch.cat(attention_scores_raw, dim=0)
            attention_scores = F.softmax(attention_scores_tensor, dim=0) # a_{j,i}
        else:
            attention_scores = torch.tensor([]).to(device)
        
        # Calculate weighted embedding (equation 11)
        weighted_embedding = torch.zeros(1, EMBEDDING_SIZE).to(device) # e_i
        for i, (n_emb, score) in enumerate(zip(neighbor_embeddings, attention_scores)):
            weighted_embedding += score * n_emb
        
        # Combine local observation with weighted embedding，用于接入MLP网络
        combined_features = torch.cat([local_obs_tensor, weighted_embedding], dim=1)
        
        return combined_features, attention_scores.detach().cpu().numpy(), local_obs_tensor
  
    def act(self, obs, neighbor_obs_list, prey_ob, add_noise=False):
        features, attention_scores, local_obs_tensor = self.process_observation(obs)

        with torch.no_grad():
            q_values, _ = self.dueling_net(features)
            q_values = q_values.cpu().numpy().squeeze()
        
        # # Select action using epsilon-greedy policy
        # if add_noise and np.random.random() < self.epsilon:
        #     action_idx = np.random.choice(len(q_values))
        # else:
        #     action_idx = np.argmax(q_values)
            
        # # Decay epsilon
        # if add_noise:
        #     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        action_idx = np.argmax(q_values)
        lambda_val, eta_val = action_pairs[action_idx]
        
        apf_action = self.compute_apf_action(
            obs, 
            lambda_val,
            eta_val,
            attention_scores
        )
        
        return apf_action, lambda_val, eta_val, attention_scores
    
    def compute_apf_action(self, obs, lambda_val, eta_val, attention_scores):
        agent_pos = obs[2:4]
        agent_vel = obs[0:2]
        
        # Get evader information
        evader_dist, evader_angle, obstacle_dist_list, obstacle_angle_list = extract_local_observation(obs)
        evader_rel_pos = obs[12:14]  # Relative position of evader
        evader_pos = agent_pos + evader_rel_pos
        
        # Get obstacle information
        obstacle_info = []
        obstacle_info= [(obstacle_dist_list[0], obstacle_angle_list[0], np.array(obs[6: 8])), (obstacle_dist_list[1], obstacle_angle_list[1], np.array(obs[8: 10]))]
        
        # Get neighbor information
        neighbor_info = extract_neighbor_info(obs) # [float32, float32, [float32, float32]]
        
        # Calculate APF-A force
        apf_force = self.apf.compute_forces(
            agent_pos,
            evader_pos,
            agent_vel, 
            obstacle_info, 
            neighbor_info, 
            lambda_val, 
            eta_val, 
            attention_scores
        )
        
        # Convert force vector to 5D action space [no_action, left, right, down, up]
        force_action = np.zeros(5, dtype=np.float32)
        # Determine primary movement direction based on force vector
        force_magnitude = np.linalg.norm(apf_force)
        
        if force_magnitude > 1e-3:
            normalized_force = apf_force / force_magnitude
            # Convert force to action indices
            fx, fy = normalized_force
            if fx < 0:
                force_action[1] = - fx
            elif fx > 0:
                force_action[2] = fx
            if fy < 0:
                force_action[3] = - fy
            elif fy > 0:
                force_action[4] = fy

            # if abs(fx) > abs(fy):  # Horizontal movement dominates
            #     if fx < 0:  # Move left
            #         force_action[1] = 1.0
            #     else:       # Move right
            #         force_action[2] = 1.0
            # else:           # Vertical movement dominates
            #     if fy < 0:  # Move down
            #         force_action[3] = 1.0
            #     else:       # Move up
            #         force_action[4] = 1.0
        else:
            # No significant force, default to no_action
            force_action[0] = 1.0
            
        return force_action
    
    def update(self, experiences, other_agents):
        def process_batch_observation(states_batch, target = False):
            evader_info = states_batch[:, 12:14]
            agent_positions = states_batch[:, 2:4]
            agent_velocities = states_batch[:, 0:2]
            
            # 被捕者距离以及方位角
            evader_distances = torch.norm(evader_info, dim=1, keepdim=True)
            evader_angles = torch.rad2deg(torch.atan2(evader_info[:, 1], evader_info[:, 0]))
            
            # 障碍物距离以及方位角
            obstacle1_info = states_batch[:, 4:6]
            obstacle2_info = states_batch[:, 6:8]
            
            obstacle1_distances = torch.norm(obstacle1_info, dim=1, keepdim=True)
            obstacle1_angles = torch.rad2deg(torch.atan2(obstacle1_info[:, 1], obstacle1_info[:, 0]))
            
            obstacle2_distances = torch.norm(obstacle2_info, dim=1, keepdim=True)
            obstacle2_angles = torch.rad2deg(torch.atan2(obstacle2_info[:, 1], obstacle2_info[:, 0]))
            
            # 构建本地观察张量
            local_obs_batch = torch.cat([
                evader_distances, evader_angles.unsqueeze(1),
                obstacle1_distances, obstacle1_angles.unsqueeze(1),
                obstacle2_distances, obstacle2_angles.unsqueeze(1)
            ], dim=1)
            
            # 处理邻居信息（批量）
            neighbor1_info = states_batch[:, 8:10]
            neighbor2_info = states_batch[:, 10:12]
            
            neighbor1_distances = torch.norm(neighbor1_info, dim=1, keepdim=True)
            neighbor1_angles = torch.rad2deg(torch.atan2(neighbor1_info[:, 1], neighbor1_info[:, 0])).unsqueeze(1)
            
            neighbor2_distances = torch.norm(neighbor2_info, dim=1, keepdim=True)
            neighbor2_angles = torch.rad2deg(torch.atan2(neighbor2_info[:, 1], neighbor2_info[:, 0])).unsqueeze(1)
            
            # 批量创建嵌入
            n1_input = torch.cat([neighbor1_distances, neighbor1_angles], dim=1)
            n2_input = torch.cat([neighbor2_distances, neighbor2_angles], dim=1)
            if not target:
                n1_embedding = self.embedding_net(n1_input)
                n1_key = self.key_net(n1_embedding)
                
                n2_embedding = self.embedding_net(n2_input)
                n2_key = self.key_net(n2_embedding)
            else:
                n1_embedding = self.target_embedding_net(n1_input)
                n1_key = self.target_key_net(n1_embedding)
                
                n2_embedding = self.target_embedding_net(n2_input)
                n2_key = self.target_key_net(n2_embedding)
            # 计算平均嵌入（批量）
            mean_embedding = (n1_embedding + n2_embedding) / 2
            
            if not target:
                # 批量计算注意力分数
                attn1 = self.attention_net(local_obs_batch, mean_embedding, n1_key)
                attn2 = self.attention_net(local_obs_batch, mean_embedding, n2_key)
            else:
                attn1 = self.target_attention_net(local_obs_batch, mean_embedding, n1_key)
                attn2 = self.target_attention_net(local_obs_batch, mean_embedding, n2_key)
            
            # 拼接并应用softmax（批量）
            attn_cat = torch.cat([attn1, attn2], dim=1)
            attention_scores = F.softmax(attn_cat, dim=1)
            
            # 计算加权嵌入（批量）
            weighted_embedding = attention_scores[:, 0:1] * n1_embedding + attention_scores[:, 1:2] * n2_embedding
            
            # 组合局部观察和加权嵌入（批量）
            combined_features = torch.cat([local_obs_batch, weighted_embedding], dim=1)
            
            return combined_features, attention_scores, local_obs_batch
        
        # 主要更新函数逻辑
        states, neighbor_states, actions, rewards, next_states, next_neighbor_states, dones = experiences
        
        # 批量处理当前状态
        features, attention_scores, local_obs = process_batch_observation(states, False)
        
        # 批量处理下一状态
        next_features, _, _ = process_batch_observation(next_states, False)
        
        # 使用目标网络批量处理
        with torch.no_grad():
            target_features, target_attention_scores, _ = process_batch_observation(states, True)
            target_features = target_features.detach()
            target_attention_scores = target_attention_scores.detach()
        
        # 批量获取Q值
        q_values, _ = self.dueling_net(features)
        actions_idx = torch.argmax(q_values, dim=1)
        
        # 批量计算目标Q值
        with torch.no_grad():
            next_q_values, _ = self.target_dueling_net(next_features)
            next_max_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards.squeeze() + GAMMA * next_max_q * (1 - dones.squeeze())
        
        # 计算TD误差
        batch_indices = torch.arange(q_values.size(0))
        q_val = q_values[batch_indices, actions_idx]
        q_loss = F.mse_loss(q_val, target_q)
        
        # 计算KL散度
        kl_loss = F.kl_div(
            F.log_softmax(attention_scores, dim=1),
            F.softmax(target_attention_scores, dim=1),
            reduction='batchmean'
        )
        
        # 总损失
        total_loss = q_loss + KL_WEIGHT * kl_loss
        
        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), CLIP_GRAD)
        self.optimizer.step()
        
        # 软更新目标网络
        self.soft_update_targets()
        
        return total_loss.item(), q_loss.item(), kl_loss.item()
    
    def hard_update_targets(self):
        self.target_embedding_net.load_state_dict(self.embedding_net.state_dict())
        self.target_key_net.load_state_dict(self.key_net.state_dict())
        self.target_attention_net.load_state_dict(self.attention_net.state_dict())
        self.target_dueling_net.load_state_dict(self.dueling_net.state_dict())
        
    def soft_update_targets(self):
        for target_param, param in zip(self.target_embedding_net.parameters(), self.embedding_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
        for target_param, param in zip(self.target_key_net.parameters(), self.key_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
        for target_param, param in zip(self.target_attention_net.parameters(), self.attention_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
        for target_param, param in zip(self.target_dueling_net.parameters(), self.dueling_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
    def parameters(self):
        return list(self.embedding_net.parameters()) + \
               list(self.key_net.parameters()) + \
               list(self.attention_net.parameters()) + \
               list(self.dueling_net.parameters())
               
    def save(self, path):
        torch.save({
            'embedding_net': self.embedding_net.state_dict(),
            'key_net': self.key_net.state_dict(),
            'attention_net': self.attention_net.state_dict(),
            'dueling_net': self.dueling_net.state_dict(),
        }, f'{path}/dacoop_agent_{self.agent_id}.pth')
        
    def load(self, path):
        checkpoint = torch.load(f'{path}/dacoop_agent_{self.agent_id}.pth')
        self.embedding_net.load_state_dict(checkpoint['embedding_net'])
        self.key_net.load_state_dict(checkpoint['key_net'])
        self.attention_net.load_state_dict(checkpoint['attention_net'])
        self.dueling_net.load_state_dict(checkpoint['dueling_net'])
        self.hard_update_targets()
