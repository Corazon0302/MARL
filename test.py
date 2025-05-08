import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

class MyWrapper(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make('CartPole-v1', render_mode=render_mode)
        super().__init__(env)
        self.env = env
    
    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        return state, info
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        return state, reward, done, truncated, info

class MetricsCallback(BaseCallback):
    """
    记录训练过程中的指标
    """
    def __init__(self, check_freq=1000, verbose=1):
        super(MetricsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.ep_len_means = []
        self.explained_variances = []
        self.timesteps = []
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # 记录当前时间步
            self.timesteps.append(self.n_calls)
            
            # 记录episode长度平均值
            if len(self.model.ep_info_buffer) > 0:
                ep_lens = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                ep_len_mean = sum(ep_lens) / len(ep_lens)
                self.ep_len_means.append(ep_len_mean)
            else:
                self.ep_len_means.append(0)
            
            # 获取explained_variance - 不同算法可能存储在不同的位置
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                exp_var_key = None
                # 查找可能的explained_variance键
                possible_keys = ['train/explained_variance', 'explained_variance', 'rollout/explained_variance']
                for key in possible_keys:
                    if key in self.model.logger.name_to_value:
                        exp_var_key = key
                        break
                
                if exp_var_key:
                    explained_variance = self.model.logger.name_to_value[exp_var_key]
                    self.explained_variances.append(explained_variance)
                else:
                    self.explained_variances.append(np.nan)
            else:
                self.explained_variances.append(np.nan)
                
            if self.verbose > 0:
                exp_var_str = f"{self.explained_variances[-1]:.4f}" if not np.isnan(self.explained_variances[-1]) else "N/A"
                print(f"步数: {self.n_calls}, 平均回合长度: {self.ep_len_means[-1]:.2f}, 解释方差: {exp_var_str}")
                
        return True
    
    def plot_metrics(self, method_name, save_path="./"):
        """绘制收集的指标"""
        os.makedirs(save_path, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # 绘制ep_len_mean
        ax1.plot(self.timesteps, self.ep_len_means, 'b-', label='平均回合长度')
        ax1.set_ylabel('回合长度')
        ax1.set_title(f'{method_name} 训练指标')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制explained_variance，过滤掉NaN值
        valid_indices = ~np.isnan(self.explained_variances)
        if np.any(valid_indices):
            x_values = [self.timesteps[i] for i in range(len(self.timesteps)) if valid_indices[i]]
            y_values = [self.explained_variances[i] for i in range(len(self.explained_variances)) if valid_indices[i]]
            ax2.plot(x_values, y_values, 'r-', label='解释方差')
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('解释方差')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, '无可用的解释方差数据', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{method_name}_training_metrics.png'))
        plt.show()
        return fig

def train(args):
    # 创建日志目录
    log_dir = f"./logs/{args.method}"
    os.makedirs(log_dir, exist_ok=True)

    # 用于训练的环境 - 无渲染，使用Monitor包装
    train_env = Monitor(MyWrapper(render_mode=None), log_dir)
    train_env.reset()

    from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
    if args.method == "PPO":
        model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")
    elif args.method == "A2C":
        model = A2C("MlpPolicy", train_env, verbose=1, device="cpu")
    elif args.method == "DQN":
        model = DQN("MlpPolicy", train_env, verbose=1, device="cpu")
    elif args.method == "SAC":
        model = SAC("MlpPolicy", train_env, verbose=1, device="cpu")
    elif args.method == "TD3":
        model = TD3("MlpPolicy", train_env, verbose=1, device="cpu")
    elif args.method == "DDPG":
        model = DDPG("MlpPolicy", train_env, verbose=1, device="cpu")

    from stable_baselines3.common.evaluation import evaluate_policy
    
    # 创建指标记录回调
    metrics_callback = MetricsCallback(check_freq=args.log_freq, verbose=1)
    
    # 训练模型 - 无可视化
    model.learn(total_timesteps=100000, progress_bar=True, callback=metrics_callback)
    
    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=20)
    print(f"评估结果 - 平均回报: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 绘制训练指标图
    metrics_callback.plot_metrics(args.method, save_path=log_dir)
    
    # 保存模型
    model.save(os.path.join(log_dir, "cartpole_model"))

    # 取消注释这部分代码以启用可视化测试
    if args.visualize:
        test_env = MyWrapper(render_mode="human")
        if args.method == "PPO":
            model = PPO.load(os.path.join(log_dir, "cartpole_model"), env=test_env)
        elif args.method == "A2C":
            model = A2C.load(os.path.join(log_dir, "cartpole_model"), env=test_env)
        elif args.method == "DQN":
            model = DQN.load(os.path.join(log_dir, "cartpole_model"), env=test_env)
        elif args.method == "SAC":
            model = SAC.load(os.path.join(log_dir, "cartpole_model"), env=test_env)
        elif args.method == "TD3":
            model = TD3.load(os.path.join(log_dir, "cartpole_model"), env=test_env)
        elif args.method == "DDPG":
            model = DDPG.load(os.path.join(log_dir, "cartpole_model"), env=test_env)
        
        obs, _ = test_env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, info = test_env.step(action)
        test_env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a RL agent on CartPole-v1")
    parser.add_argument("-m", "--method", default="PPO", type=str, help="RL方法 (PPO, A2C, DQN, SAC, TD3, DDPG)")
    parser.add_argument("-f", "--log_freq", type=int, default=1000, help="记录指标的频率（步数）")
    parser.add_argument("-v", "--visualize", action="store_true", help="训练后可视化")
    args = parser.parse_args()
    train(args)