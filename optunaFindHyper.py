import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import optuna
import rewards.heuristic
from Visual_Components.field import SoccerField

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episodic_reward_list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                self.episodic_reward_list.append(episode_reward)
                min_r = np.min(self.episodic_reward_list)
                max_r = np.max(self.episodic_reward_list)
                avg_r = np.mean(self.episodic_reward_list)
                print(
                    f"Episode: {len(self.episodic_reward_list)}\n"
                    f"Reward : {episode_reward:.2f}\n"
                    f"Min    : {min_r:.2f}\n"
                    f"Max    : {max_r:.2f}\n"
                    f"Avg    : {avg_r:.2f}"
                )
        return True

class SoccerFieldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, game_duration=30, render_mode=None):
        super().__init__()
        self.scoring_team = None
        self.soccer_field = SoccerField(game_duration=game_duration)
        self.width = self.soccer_field.width
        self.height = self.soccer_field.height
        self.game_duration = game_duration
        self.players = self.soccer_field.players
        self.ball = self.soccer_field.ball
        self.screen = self.soccer_field.screen
        self.clock = self.soccer_field.clock
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])
        low = np.array([
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -6,
            -6,
            0,
            0,
            0,
        ], dtype=np.float32)
        high = np.array([
            self.width,
            self.height,
            self.width,
            self.height,
            self.width,
            self.height,
            self.width,
            self.height,
            self.width,
            self.height,
            6,
            6,
            10,
            10,
            self.game_duration,
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
        p_b1 = self.players[0]
        p_b2 = self.players[1]
        p_r1 = self.players[2]
        p_r2 = self.players[3]
        remaining_time = (
            self.soccer_field.game_duration
            - (pygame.time.get_ticks() - self.soccer_field.start_time) / 1000
        )
        observation = np.array([
            p_b1.x, p_b1.y,
            p_b2.x, p_b2.y,
            p_r1.x, p_r1.y,
            p_r2.x, p_r2.y,
            self.ball.x,
            self.ball.y,
            self.ball.velocity[0],
            self.ball.velocity[1],
            self.soccer_field.red_score,
            self.soccer_field.blue_score,
            remaining_time,
        ], dtype=np.float32)
        return observation

    def step(self, action):
        (action_b1, action_b2, action_r1, action_r2) = action
        self._take_action(action_b1, 0)
        self._take_action(action_b2, 1)
        self._take_action(action_r1, 2)
        self._take_action(action_r2, 3)
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

def optimize_ppo(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 512, 4096, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    base_env = SoccerFieldEnv(render_mode="rgb_array", game_duration=30)
    monitored_env = Monitor(base_env)
    vec_env = DummyVecEnv([lambda: monitored_env])
    eval_env = DummyVecEnv([lambda: Monitor(SoccerFieldEnv(render_mode="rgb_array", game_duration=30))])
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=50000, callback=eval_callback)
    return eval_callback.best_mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=5)
    print("===== Hyperparameter Optimization Results =====")
    print("Best Trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Mean Reward): {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    best_params = study.best_params
    print("Best hyperparameters found by Optuna:", best_params)
