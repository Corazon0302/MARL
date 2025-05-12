from mpe2 import simple_tag_v3
env = simple_tag_v3.parallel_env(
        num_good=1,              # 1个逃避者
        num_adversaries=2,       # 3个追捕者
        num_obstacles=2,         # 2个障碍物
        max_cycles=25,    # 最大步数
        continuous_actions=True # 使用连续动作空间
    )

env.reset()
print(env.observation_space(env.agents[0]))
print(env.action_space(env.agents[0]))
print(env.possible_agents)
