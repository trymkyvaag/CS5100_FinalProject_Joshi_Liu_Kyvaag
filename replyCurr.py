from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from mainCur import SoccerFieldEnv

env = DummyVecEnv([lambda: SoccerFieldEnv(render_mode="human")])
model_path = "model_checkpoints/currLearn/soccer_model_400000_steps.zip"
model = PPO.load(model_path, env=env)
obs = env.reset()
done = False

# set curriculum stage for reply!
# env.unwrapped.envs[0].set_curriculum_stage({
#     'blue_player_1_disabled': False,
#     'blue_player_2_disabled': False,
#     'red_player_1_disabled': True,  # Disable red player 1
#     'red_player_2_disabled': True,  # Disable red player 2
#     'red_player_1_moving': False,
#     'red_player_2_moving': False
# })

while not done:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, infos = env.step(action)
    done = dones[0]
    env.render()
