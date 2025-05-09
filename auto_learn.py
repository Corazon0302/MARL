import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv
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
        ax1.plot(self.timesteps, self.ep_len_means, 'b-', label='ep_len_means')
        ax1.set_ylabel('ep_len_means')
        ax1.set_title(f'{method_name} Training Metrics')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制explained_variance，过滤掉NaN值
        valid_indices = ~np.isnan(self.explained_variances)
        if np.any(valid_indices):
            x_values = [self.timesteps[i] for i in range(len(self.timesteps)) if valid_indices[i]]
            y_values = [self.explained_variances[i] for i in range(len(self.explained_variances)) if valid_indices[i]]
            ax2.plot(x_values, y_values, 'r-', label='explained_variance')
            ax2.set_xlabel('steps')
            ax2.set_ylabel('explained_variance')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, '没有可用的explained_variance数据', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{method_name}_training_metrics.png'))
        plt.show()
        return fig

from datetime import datetime
def train(args, params, save_model=False):
    # 创建日志目录
    current_time = datetime.now().strftime("%m%d-%H%M")
    log_dir = f"./logs/PPO/{current_time}"
    os.makedirs(log_dir, exist_ok=True)

    # 用于训练的环境 - 无渲染，使用Monitor包装
    if args.multi_env:
        train_env = DummyVecEnv([
            lambda: Monitor(MyWrapper(render_mode=None), f"{log_dir}/env_0"),
            lambda: Monitor(MyWrapper(render_mode=None), f"{log_dir}/env_1"),
            lambda: Monitor(MyWrapper(render_mode=None), f"{log_dir}/env_2"),
            lambda: Monitor(MyWrapper(render_mode=None), f"{log_dir}/env_3")
        ])
    else:
        train_env = Monitor(MyWrapper(render_mode=None), log_dir)

    from stable_baselines3 import PPO
    model = PPO(
        "MlpPolicy", 
        env=train_env, 
        n_steps=1024,
        batch_size=64,
        n_epochs=params['n_epochs'],
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=0, 
        device="cpu"
        )

    from stable_baselines3.common.evaluation import evaluate_policy
    
    # 创建指标记录回调
    metrics_callback = MetricsCallback(check_freq=args.log_freq, verbose=1)
    
    # 训练模型 - 无可视化
    model.learn(total_timesteps=params['total_timesteps'], progress_bar=True, callback=metrics_callback)

    if save_model:
        model.save(os.path.join(log_dir, "ppo_cartpole"))
        metrics_callback.plot_metrics(PPO, save_path=log_dir)

    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=50)
    print(f"评估结果 - 平均回报: {mean_reward:.2f} +/- {std_reward:.2f}")
    return (mean_reward - std_reward)

def optimal_param(args, trial):
    params = {
        'n_epochs': trial.suggest_int('n_epochs', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'total_timesteps': trial.suggest_int('total_timesteps', 20000, 80000),
    }
    return train(args, params)

def study(args):
    import optuna
    from optuna.samplers import TPESampler
    study = optuna.create_study(sampler=TPESampler(), study_name='PPO-LunarLander-v2', direction="maximize")
    study.optimize(lambda trial: optimal_param(args, trial), n_trials=10)
    
    # 打印最佳参数
    print("最佳参数：", study.best_params)
    print("最佳值：", study.best_value)
    return study.best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a RL agent on CartPole-v1")
    parser.add_argument("-f", "--log_freq", type=int, default=1000, help="记录指标的频率（步数）")
    parser.add_argument("--multi_env" , action="store_true", help="使用多个环境进行训练")
    parser.add_argument("-v", "--visualize", action="store_true", help="训练后可视化")
    args = parser.parse_args()
    # best_params = study(args)
    train(args,  {'n_epochs': 9, 'learning_rate': 0.001523891976554767, 'gamma': 0.9655161006681183, 'total_timesteps': 52807}, True)