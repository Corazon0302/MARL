import gym 

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
    
def train(args):
    # 用于训练的环境 - 无渲染
    train_env = MyWrapper(render_mode=None)
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

    # 训练模型 - 无可视化
    model.learn(total_timesteps=100000, progress_bar=True)
    evaluate_policy(model, train_env, n_eval_episodes=20)

    model.save("cartpole")

    # 创建新的环境用于测试 - 带可视化
    # test_env = MyWrapper(render_mode="human")
    # if args.method == "PPO":
    #     model = PPO.load("cartpole", env=test_env)
    # elif args.method == "A2C":
    #     model = A2C.load("cartpole", env=test_env)
    # elif args.method == "DQN":
    #     model = DQN.load("cartpole", env=test_env)
    # elif args.method == "SAC":
    #     model = SAC.load("cartpole", env=test_env)
    # elif args.method == "TD3":
    #     model = TD3.load("cartpole", env=test_env)
    # elif args.method == "DDPG":
    #     model = DDPG.load("cartpole", env=test_env)
    
    # obs, _ = test_env.reset()
    # done, truncated = False, False
    # while not done and not truncated:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, truncated, info = test_env.step(action)
    # test_env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a PPO agent on CartPole-v1")
    parser.add_argument("-m", "--method", default="PPO", type=str, help="RL method to use")
    args = parser.parse_args()
    train(args)
