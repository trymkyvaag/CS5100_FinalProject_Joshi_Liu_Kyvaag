from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from mainCur import SoccerFieldEnv

# Create environment
env = DummyVecEnv([lambda: SoccerFieldEnv(render_mode="human")])

# Load the model
model_path = "model_checkpoints/currLearn/soccer_model_100000_steps.zip"
model = PPO.load(model_path, env=env)

# Reset environment
obs = env.reset()
done = False

## set curriculum stage for reply!
# env.unwrapped.envs[0].set_curriculum_stage({
#     'blue_player_1_disabled': False,
#     'blue_player_2_disabled': False,
#     'red_player_1_disabled': True,  # Disable red player 1
#     'red_player_2_disabled': True,  # Disable red player 2
#     'red_player_1_moving': False,
#     'red_player_2_moving': False
# })

while not done:
    # Predict action
    action, _states = model.predict(obs, deterministic=False)
    
    # Step environment (DummyVecEnv handles the interface difference)
    obs, rewards, dones, infos = env.step(action)
    
    # Update done flag
    done = dones[0]
    
    # Render
    env.render()