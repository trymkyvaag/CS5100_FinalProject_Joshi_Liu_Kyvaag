import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episodic_reward_list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                self.episodic_reward_list.append(episode_reward)
                minimum_reward = np.min(self.episodic_reward_list)
                maximum_reward = np.max(self.episodic_reward_list)
                average_reward = np.mean(self.episodic_reward_list)
                print(f"Episode: {len(self.episodic_reward_list)}\n"
                      f"Reward : {episode_reward:f}\n"
                      f"Min    : {minimum_reward:.4f}\n"
                      f"Max    : {maximum_reward:.4f}\n"
                      f"Avg    : {average_reward:.4f}")
        return True

import rewards.heuristic

class SoccerFieldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, game_duration=30, render_mode=None):
        super(SoccerFieldEnv, self).__init__()
        from Visual_Components.field import SoccerField
        self.soccer_field = SoccerField(game_duration=game_duration)
        self.width = self.soccer_field.width
        self.height = self.soccer_field.height
        self.game_duration = game_duration
        self.players = self.soccer_field.players
        self.ball = self.soccer_field.ball
        self.screen = self.soccer_field.screen
        self.clock = self.soccer_field.clock
        self.scoring_team = None
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])
        low = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, -6, -6,
            0, 0, 0
        ], dtype=np.float32)
        high = np.array([
            self.width, self.height, self.width, self.height,
            self.width, self.height, self.width, self.height,
            self.width, self.height, 6, 6,
            10, 10, self.game_duration
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        pygame.init()
        self.render_mode = render_mode
        self.screen = None
        if self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("SoccerFieldEnv")
            self.screen = pygame.display.set_mode((self.width, self.height))
        elif self.render_mode == "rgb_array":
            pygame.display.init()
            self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soccer_field.reset_positions()
        self.ball = self.soccer_field.ball
        self.players = self.soccer_field.players
        self.soccer_field.red_score = 0
        self.soccer_field.blue_score = 0
        self.soccer_field.kickoff_started = False
        self.soccer_field.start_time = pygame.time.get_ticks()
        observation = self._get_observation()
        info = {}
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def _get_observation(self):
        player_blue_1 = self.players[0]
        player_blue_2 = self.players[1]
        player_red_1 = self.players[2]
        player_red_2 = self.players[3]
        remaining_time = (
            self.soccer_field.game_duration
            - (pygame.time.get_ticks() - self.soccer_field.start_time) / 1000
        )
        observation = np.array([
            player_blue_1.x, player_blue_1.y,
            player_blue_2.x, player_blue_2.y,
            player_red_1.x, player_red_1.y,
            player_red_2.x, player_red_2.y,
            self.ball.x, self.ball.y,
            self.ball.velocity[0], self.ball.velocity[1],
            self.soccer_field.red_score,
            self.soccer_field.blue_score,
            remaining_time
        ], dtype=np.float32)
        return observation

    def step(self, action):
        (action_agent_blue_1,
         action_agent_blue_2,
         action_agent_red_1,
         action_agent_red_2) = action
        self._take_action(action_agent_blue_1, 0)
        self._take_action(action_agent_blue_2, 1)
        self._take_action(action_agent_red_1, 2)
        self._take_action(action_agent_red_2, 3)
        self._update_game_state()
        reward = self._calculate_reward()
        terminated = self._is_done()
        truncated = self._is_truncated()
        observation = self._get_observation()
        info = {}
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    def _take_action(self, action, player_index):
        player = self.players[player_index]
        if not player.frozen:
            if not self.soccer_field.kickoff_started:
                self.soccer_field.kickoff_started = True
                if self.scoring_team is None:
                    self.soccer_field.unfreeze_team("red")
                    self.soccer_field.unfreeze_team("blue")
                else:
                    self.soccer_field.unfreeze_team(self.scoring_team)
                self.scoring_team = None
            if action == 0:
                player.move(0, -1, self.width, self.height, self.players)
            elif action == 1:
                player.move(0, 1, self.width, self.height, self.players)
            elif action == 2:
                player.move(-1, 0, self.width, self.height, self.players)
            elif action == 3:
                player.move(1, 0, self.width, self.height, self.players)

    def _update_game_state(self):
        self.soccer_field.check_player_ball_overlaps()

    def _calculate_reward(self):
        return rewards.heuristic.reward_function(self)

    def _is_done(self):
        return self.soccer_field.check_goal()[0]

    def _is_truncated(self):
        elapsed_time = (pygame.time.get_ticks() - self.soccer_field.start_time) / 1000
        return elapsed_time > self.game_duration

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        self.soccer_field.draw_field()
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

def train_ppo(env, hyperparams, total_timesteps=1_000_000, callbacks=None, log_path=None):
    """
    Train a PPO model given hyperparams and return the trained model.
    """
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        **hyperparams
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    return model

if __name__ == "__main__":
    original_params = {
        "learning_rate": 3e-5
    }
    best_params = {
        "learning_rate": 9.374410314646429e-05,
        "n_steps": 3296,
        "gamma": 0.9708197855085876,
        "gae_lambda": 0.9464920978639838,
        "ent_coef": 0.010549674409905044,
        "clip_range": 0.3941564070073835
    }
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./model_checkpoints/",
        name_prefix="soccer_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    reward_callback = RewardCallback()
    callbacks = [checkpoint_callback, reward_callback]
    dummy_env_orig = SoccerFieldEnv(render_mode="rgb_array", game_duration=30)
    monitored_env_orig = Monitor(dummy_env_orig)
    env_orig = DummyVecEnv([lambda: monitored_env_orig])
    model_original = train_ppo(
        env=env_orig,
        hyperparams=original_params,
        total_timesteps=1_000_000,
        callbacks=callbacks,
        log_path="./tb_logs/original"
    )
    model_original.save("soccer_agent_ppo_original")
    env_orig.close()
    dummy_env_optuna = SoccerFieldEnv(render_mode="rgb_array", game_duration=30)
    monitored_env_optuna = Monitor(dummy_env_optuna)
    env_optuna = DummyVecEnv([lambda: monitored_env_optuna])
    model_optuna = train_ppo(
        env=env_optuna,
        hyperparams=best_params,
        total_timesteps=1_000_000,
        callbacks=callbacks,
        log_path="./tb_logs/optuna"
    )
    model_optuna.save("soccer_agent_ppo_optuna")
    env_optuna.close()
    eval_env = SoccerFieldEnv(render_mode="rgb_array", game_duration=30)
    eval_env_mon = Monitor(eval_env)
    eval_env_vec = DummyVecEnv([lambda: eval_env_mon])
    mean_reward_org, std_reward_org = evaluate_policy(
        model_original, eval_env_vec, n_eval_episodes=10
    )
    print(f"\n[Evaluation - Original Model] Mean reward: {mean_reward_org:.2f} +/- {std_reward_org:.2f}")
    mean_reward_opt, std_reward_opt = evaluate_policy(
        model_optuna, eval_env_vec, n_eval_episodes=10
    )
    print(f"[Evaluation - Optuna Model]   Mean reward: {mean_reward_opt:.2f} +/- {std_reward_opt:.2f}")
    eval_env_vec.close()
