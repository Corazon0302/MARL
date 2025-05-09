from stable_baselines3 import PPO
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim):
        super().__init__(observation_space, hidden_dim)
        input_dim = observation_space.shape[0]
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=1,
                kernel_size=1,
                padding=0
            ),
            torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, state):
        batch_size = state.shape[0]
        state = state.reshape(batch_size, -1, 1, 1) # -1表示自适应通道数量
        return self.sequential(state)
    
class CustomNet(torch.nn.Module):
    def __init__(self, feature_dim, last_layer_dim_pi=64, last_layer_dim_vf=64):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, last_layer_dim_pi),
            torch.nn.ReLU(),
        )

        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, last_layer_dim_vf),
            torch.nn.ReLU(),
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class CustomAC(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args,
                         **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNet(self.features_dim)

env = gym.make('CartPole-v1')
env.reset()

model = PPO(
    CustomAC,
    env,
    policy_kwargs={
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': {'hidden_dim': 8}
    },
    verbose=1
)
env.reset()
model.learn(50000, progress_bar=True)
from datetime import datetime
current_time = datetime.now().strftime("%m%d%H%M")
dir_path = f"logs/custom/{current_time}_model"
model.save(dir_path)
mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=50)

print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
env = gym.make('CartPole-v1', render_mode='human')
obs, _ = env.reset()
done, truncated = False, False
while not done and not truncated:
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
env.close()