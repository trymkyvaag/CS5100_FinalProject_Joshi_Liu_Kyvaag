"""
replay.py

This script loads a trained PPO agent and a VecNormalize object, and then evaluates the agent's performance in a
SoccerField environment. The environment is then rendered allowing each individual episode to be observed.
"""
import random

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from main import SoccerFieldEnv


def make_env():
    """
    Creates an instance of the SoccerFieldEnv that is monitored with rendering that is viewable.
    Returns:
        Monitor wrapped SoccerField environment for keeping track of individual episode statistics.
    """
    return Monitor(SoccerFieldEnv(render_mode="human"))


eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

eval_env.training = False
eval_env.norm_reward = False

model = PPO.load("soccer_agent_ppo.zip", env=eval_env)

NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    eval_env.seed(random.randint(0, 1_000_000))
    obs = eval_env.reset()
    done = False
    print(f"\nEpisode {episode + 1}")

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
