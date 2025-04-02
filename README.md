# CS5100_FinalProject_Joshi_Liu_Kyvaag

Final Project for CS5100 FAI

# Work in progress

## For early report use these instructions

1. Checkout the code.
2. Shift to branch 7-git-ignore-readme-requirements
3. Make sure python3 is installed, latest is better
4. We recommend a venv `python3 -m venv .venv`
5. Install requirements `pip -r requirements.txt`
6. To see live training set in main.py line 354 render_mode='human' else just run `python3 main.py` (we will use args
   later for this)
7. Once training is done model_checkpoints is populated with checkpoints for 100k timesteps each, a final model named
   soccer_agent_ppo.zip would be saved in the root directory of this project as well.
8. During training a live graph and rewards csv would be populated in reward_logs directory.
9. To replay with a trained model run `python3 replay.py` by setting the model you want to use in line 8 of replay.py

## Important note

1. We have two zip files for nonoptim and optuna for each of model checkpoints, reward_logs and soccer_agent_ppo (our
   final model). Unzip the ones under model checkpoints and reward_logs to see the contents, use soccer_agent_ppo as
   is. (don't unzip)
2. The previous step is important to run point 9 of the instructions if you plan to use a checkpoint. If not just run
   replay.py and you will use the trained model by default.
