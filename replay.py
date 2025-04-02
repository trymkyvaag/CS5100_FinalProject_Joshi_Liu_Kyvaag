from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from main import SoccerFieldEnv

env = DummyVecEnv([lambda: SoccerFieldEnv(render_mode="human")])

model_path = "model_checkpoints/currLearn/soccer_model_100000_steps.zip"
model = PPO.load(model_path, env=env)

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, done, info = env.step(action)
    env.render()
