import numpy as np
import time
from mpe2 import simple_tag_v3
import supersuit as ss
import pygame
import os

def make_env(render_mode="human"):
    """创建并返回环境"""
    env = simple_tag_v3.parallel_env(
        num_good=1,              # 1 prey (绿色)
        num_adversaries=3,       # 3 predators (红色)
        num_obstacles=2,         # 2 obstacles (黑色)
        max_cycles=100,          # 最大步数
        continuous_actions=True, # 使用连续动作空间
        dynamic_rescaling=True,
        render_mode=render_mode,
    )
    return env

def explain_observation(obs, agent_id):
    """解释智能体的观察空间内容"""
    print(f"\n{agent_id} 观察:")
    print(f"  自身速度: {obs[0:2]}")
    print(f"  自身位置: {obs[2:4]}")
    
    # 障碍物位置
    print(f"  障碍物相对位置:")
    for i in range(2):  # 两个障碍物
        start_idx = 4 + i*2
        print(f"    障碍物 {i}: {obs[start_idx:start_idx+2]}")
    
    # 其他智能体的位置
    other_agents_start = 8  # 4 + 2*2 (自身 + 障碍物)
    print(f"  其他智能体相对位置:")
    for i in range(3):  # 假设3个其他智能体
        start_idx = other_agents_start + i*2
        if start_idx+2 <= len(obs):
            print(f"    智能体 {i}: {obs[start_idx:start_idx+2]}")

def print_action_guide():
    """打印连续动作空间指南"""
    print("\n连续动作空间指南:")
    print("  每个智能体的动作是一个5维向量，每个值范围在[0,1]之间")
    print("  动作向量的含义:")
    print("    值1: 无动作权重 (越大越不动)")
    print("    值2: 向左移动权重")
    print("    值3: 向右移动权重")
    print("    值4: 向下移动权重")
    print("    值5: 向上移动权重")
    print("  例如: 0 0 1 0 0 表示向右移动")
    print("        0 0 0 0 1 表示向上移动")
    print("        0.1 0.2 0.3 0.2 0.2 表示混合动作，偏向右移")
    print("        1 0 0 0 0 表示不移动")

def get_user_action(agent_id):
    """获取用户为智能体输入的连续动作"""
    print(f"\n请为 {agent_id} 输入5个连续值 (范围0-1，用空格分隔):")
    while True:
        try:
            action_str = input("输入5个值 (如 0 0.2 0.8 0 0): ")
            values = action_str.split()
            
            if len(values) != 5:
                print("错误: 需要输入5个值")
                continue
                
            action = np.array([float(v) for v in values], dtype=np.float32)
            
            # 验证数值范围
            if np.any(action < 0) or np.any(action > 1):
                print("错误: 所有值必须在0到1之间")
                continue
                
            return action
        except ValueError:
            print("错误: 请输入有效的数字")

def run_interactive_env(control_type="all", others_static=True):
    """运行交互式环境
    
    Args:
        control_type: 控制类型，"all"控制所有智能体，"predators"只控制捕食者，
                     "prey"只控制猎物，"one_predator"只控制一个捕食者
        others_static: 是否让其他未控制的智能体保持静止
    """
    env = make_env()
    observations, _ = env.reset()
    
    print("\n欢迎使用Simple Tag环境交互控制器!")
    print("环境中有3个红色捕食者(adversary)和1个绿色猎物(agent)")
    print("黑色圆形是障碍物，会阻挡智能体移动")
    print_action_guide()
    
    if control_type == "all":
        print("\n控制模式: 控制所有智能体")
    elif control_type == "predators":
        print("\n控制模式: 只控制捕食者")
    elif control_type == "prey":
        print("\n控制模式: 只控制猎物")
    elif control_type == "one_predator":
        print("\n控制模式: 只控制adversary_0捕食者")
    
    if others_static:
        print("其他未控制的智能体将保持静止")
    
    step = 0
    episode_done = False
    
    # 主循环
    while not episode_done and step < 100:
        env.render()
        
        actions = {}
        
        # 获取所有智能体的动作
        for agent_id in env.agents:
            
            # 根据控制类型决定是否由用户控制该智能体
            is_predator = "adversary" in agent_id
            is_prey = not is_predator
            
            should_control = (
                control_type == "all" or
                (control_type == "predators" and is_predator) or
                (control_type == "prey" and is_prey) or
                (control_type == "one_predator" and agent_id == "adversary_0")
            )
            
            if should_control:
                # 由用户控制
                explain_observation(observations[agent_id], agent_id)
                actions[agent_id] = get_user_action(agent_id)
            else:
                # 不由用户控制的智能体
                if others_static:
                    # 保持静止 - 设置"无动作"权重为1，其他为0
                    actions[agent_id] = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    # 原代码中的AI策略
                    if is_predator:
                        # 捕食者随机移动
                        action = np.zeros(5, dtype=np.float32)
                        main_dir = np.random.randint(1, 5)
                        action[main_dir] = 0.8
                        action += np.random.uniform(0, 0.2, 5)
                        actions[agent_id] = np.clip(action, 0, 1)
                    else:
                        # 猎物逃跑策略
                        prey_obs = observations[agent_id]
                        escape_vec = np.zeros(2)
                        
                        for i in range(3):
                            pred_pos = prey_obs[8+i*2:10+i*2]
                            direction = -pred_pos
                            dist = np.linalg.norm(direction)
                            if dist > 0.1:
                                escape_vec += direction / (dist**2)
                        
                        action = np.zeros(5, dtype=np.float32)
                        
                        if np.linalg.norm(escape_vec) > 0:
                            escape_vec = escape_vec / np.linalg.norm(escape_vec)
                            
                            if escape_vec[0] < -0.3:
                                action[1] = -escape_vec[0]
                            elif escape_vec[0] > 0.3:
                                action[2] = escape_vec[0]
                                
                            if escape_vec[1] < -0.3:
                                action[3] = -escape_vec[1]
                            elif escape_vec[1] > 0.3:
                                action[4] = escape_vec[1]
                        else:
                            action = np.random.uniform(0, 0.3, 5)
                            
                        actions[agent_id] = np.clip(action, 0, 1)
        
        # 执行所有动作
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 显示每个智能体的奖励
        print("\n奖励:")
        for agent_id, reward in rewards.items():
            print(f"  {agent_id}: {reward:.2f}")
        
        # 检查是否任何智能体结束
        episode_done = any(terminations.values()) or any(truncations.values())
        
        observations = next_observations
        step += 1
        
        # 等待用户确认继续
        input("\n按Enter继续到下一步...")
    
    print("Episode结束!")
    env.close()

if __name__ == "__main__":
    # 让用户选择控制模式
    print("请选择控制模式:")
    print("1: 控制所有智能体")
    print("2: 只控制捕食者")
    print("3: 只控制猎物")
    print("4: 只控制一个捕食者(adversary_0)")
    
    while True:
        try:
            mode = int(input("请选择 (1-4): "))
            if 1 <= mode <= 4:
                break
            else:
                print("无效选择，请输入1-4之间的数字")
        except ValueError:
            print("无效输入，请输入1-4之间的数字")
    
    control_modes = {
        1: "all",
        2: "predators",
        3: "prey", 
        4: "one_predator"
    }
    
    # 静态选择
    print("\n其他智能体行为:")
    print("1: 保持静止")
    print("2: 使用AI策略")
    
    while True:
        try:
            static_mode = int(input("请选择 (1-2): "))
            if 1 <= static_mode <= 2:
                break
            else:
                print("无效选择，请输入1-2之间的数字")
        except ValueError:
            print("无效输入，请输入1-2之间的数字")
    
    others_static = (static_mode == 1)
    
    run_interactive_env(control_modes[mode], others_static)