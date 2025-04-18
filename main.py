"""
main.py

This script trains a PPO agent to play a custom 2v2 soccer game using Gymnasium and Stable-Baselines3.
It features a custom environment, a reward tracker, a logging callback, and evaluation logic.
"""

import os

import gymnasium as gym
import numpy as np
import pandas as pd
import pygame
from gymnasium import Wrapper
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import rewards.heuristic
from Visual_Components.field import SoccerField
from utils import set_seed

SEED = 42
set_seed(SEED)


class RewardTracker(Wrapper):
    """
    A Gym wrapper that tracks reward statistics during training.

    Attributes:
        episode_reward (float): Accumulated reward for the current episode.
        episode_rewards (list): List of total rewards from completed episodes.
        episode_lengths (list): Length of each completed episode.
        step_count (int): Total steps taken across all episodes.
    """

    def __init__(self, env):
        """
        Initialize the RewardTracker wrapper.

        Args:
            env (gym.Env): The environment to wrap.
        """
        super().__init__(env)
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_count = 0

    def reset(self, **kwargs):
        """
        Reset the environment and reward statistics for the new episode.

        Returns:
            observation (np.ndarray): Initial environment observation.
            info (dict): Additional reset info.
        """
        self.episode_reward = 0
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        """
        Step the environment and track reward and episode statistics.

        Args:
            action (any): Action to perform.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.step_count += 1

        if terminated or truncated:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.step_count - sum(self.episode_lengths))
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_lengths[-1],
                "episode_num": len(self.episode_rewards),
            }
            self.episode_reward = 0

        return observation, reward, terminated, truncated, info

    def get_stats(self):
        """
        Compute and return aggregated reward statistics.

        Returns:
            dict: Dictionary containing reward and episode statistics.
        """
        stats = {
            "episode_count": len(self.episode_rewards),
            "total_steps": self.step_count,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "median_reward": (
                np.median(self.episode_rewards) if self.episode_rewards else 0
            ),
            "min_reward": np.min(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": np.max(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "mean_episode_length": (
                np.mean(self.episode_lengths) if self.episode_lengths else 0
            ),
        }

        if len(self.episode_rewards) >= 100:
            stats["last_100_mean_reward"] = np.mean(self.episode_rewards[-100:])
            stats["last_100_median_reward"] = np.median(self.episode_rewards[-100:])

        return stats


class RewardLoggingCallback(BaseCallback):
    """
    A Stable-Baselines3 callback to log reward statistics and evaluate the model periodically.

    Attributes:
        reward_tracker (RewardTracker): The reward tracker used for stat collection.
        eval_env (VecEnv): Evaluation environment.
        log_dir (str): Directory to save CSV logs.
        eval_freq (int): Frequency in timesteps to evaluate the model.
        stats_history (list): Logged statistics history.
    """

    def __init__(
        self,
        reward_tracker,
        eval_env=None,
        log_dir="./reward_logs/",
        eval_freq=50000,
        verbose=0,
    ):
        """
        Initialize the RewardLoggingCallback.

        Args:
            reward_tracker (RewardTracker): Tracker for training rewards.
            eval_env (VecEnv): Environment used for periodic evaluation.
            log_dir (str): Path to save CSV logs.
            eval_freq (int): Timesteps between evaluations.
            verbose (int): Verbosity level.
        """
        super(RewardLoggingCallback, self).__init__(verbose)
        self.reward_tracker = reward_tracker
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.stats_history = []
        self._last_eval_step = 0
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self):
        """
        Called at each environment step.

        Returns:
            bool: Always True to continue training.
        """
        return True

    def _on_rollout_end(self):
        """
        Called at the end of each rollout to log training and evaluation metrics.
        """
        stats = self.reward_tracker.get_stats()
        stats["timesteps"] = self.num_timesteps

        metrics_to_track = [
            "train/approx_kl",
            "train/clip_fraction",
            "train/entropy_loss",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/loss",
            "train/explained_variance",
            "train/learning_rate",
        ]

        for key in metrics_to_track:
            value = self.logger.name_to_value.get(key)
            if value is not None:
                stats[key] = value

        if (
            self.eval_env is not None
            and (self.num_timesteps - self._last_eval_step) >= self.eval_freq
        ):
            eval_mean, eval_std = self._evaluate_model()
            stats["eval_mean_reward"] = eval_mean
            stats["eval_std_reward"] = eval_std
            self._last_eval_step = self.num_timesteps
        self.stats_history.append(stats)

        if len(self.stats_history) % 10 == 0:
            self._save_stats()

    def _save_stats(self):
        """
        Save reward and training stats to CSV.
        """
        df = pd.DataFrame(self.stats_history)
        df.to_csv(os.path.join(self.log_dir, "reward_stats.csv"), index=False)

    def _evaluate_model(self):
        """
        Evaluate the current policy on the evaluation environment.

        Returns:
            tuple: (mean_reward, std_reward)
        """
        if self.eval_env is None:
            return None, None

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=5,
            render=False,
            deterministic=True,
        )
        return mean_reward, std_reward

    def on_training_end(self):
        """
        Called when training is completed to persist stats.
        """
        self._save_stats()


class SoccerFieldEnv(gym.Env):
    """
    A custom Gym environment simulating a 2v2 soccer game using Pygame.

    The environment features multi-agent control, ball dynamics, goal scoring,
    and support for both human and RGB rendering.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, game_duration=30, render_mode=None):
        """
        Initialize the soccer environment.

        Args:
            game_duration (int): Length of a game in seconds.
            render_mode (str): Either 'human' for display or 'rgb_array' for image frames.
        """
        super(SoccerFieldEnv, self).__init__()

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

        low = np.array(
            [
                0,  # Player 1 blue X
                0,  # Player 1 blue Y
                0,  # Player 2 blue X
                0,  # Player 2 blue Y
                0,  # Player 3 red X
                0,  # Player 3 red Y
                0,  # Player 4 red X
                0,  # Player 4 red Y
                0,  # Ball X
                0,  # Ball Y
                -6,  # Ball X Velocity
                -6,  # Ball Y Velocity
                0,  # Red Score
                0,  # Blue Score
                0,  # Remaining Time
            ],
            dtype=np.float32,
        )

        high = np.array(
            [
                self.width,  # Player 1 blue X
                self.height,  # Player 1 blue Y
                self.width,  # Player 2 blue X
                self.height,  # Player 2 blue Y
                self.width,  # Player 3 red X
                self.height,  # Player 3 red Y
                self.width,  # Player 4 red X
                self.height,  # Player 4 red Y
                self.width,  # Ball X
                self.height,  # Ball Y
                6,  # Ball X Velocity
                6,  # Ball Y Velocity
                10,  # Red Score
                10,  # Blue Score
                self.game_duration,  # Remaining Time
            ],
            dtype=np.float32,
        )

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
        """
        Reset the game environment for a new episode.

        Args:
            seed (int): Optional seed for randomization.
            options (dict): Optional reset parameters.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)

        self.soccer_field.reset_positions()

        # Randomize player positions
        for player in self.players:
            if player.team == "blue":
                x_min, x_max = 50, self.width // 2 - 50
            else:
                x_min, x_max = self.width // 2 + 50, self.width - 50

            y_min, y_max = 50, self.height - 50

            player.x = self.np_random.uniform(x_min, x_max)
            player.y = self.np_random.uniform(y_min, y_max)

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
        """
        Collect environment observations.

        Returns:
            np.ndarray: The current state observation.
        """
        player_blue_1 = self.players[0]
        player_blue_2 = self.players[1]
        player_red_1 = self.players[2]
        player_red_2 = self.players[3]
        remaining_time = (
            self.soccer_field.game_duration
            - (pygame.time.get_ticks() - self.soccer_field.start_time) / 1000
        )
        observation = np.array(
            [
                player_blue_1.x,
                player_blue_1.y,
                player_blue_2.x,
                player_blue_2.y,
                player_red_1.x,
                player_red_1.y,
                player_red_2.x,
                player_red_2.y,
                self.ball.x,
                self.ball.y,
                self.ball.velocity[0],
                self.ball.velocity[1],
                self.soccer_field.red_score,
                self.soccer_field.blue_score,
                remaining_time,
            ],
            dtype=np.float32,
        )
        return observation

    def step(self, action):
        """
        Perform a step in the environment.

        Args:
            action (list[int]): Actions for the 4 players.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        (
            action_agent_blue_1,
            action_agent_blue_2,
            action_agent_red_1,
            action_agent_red_2,
        ) = action

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
        """
        Apply a single player's action.

        Args:
            action (int): The action to perform.
            player_index (int): Index of the player.
        """

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

            if action == 0:  # forward
                player.move(0, -1, self.width, self.height, self.players)
            elif action == 1:  # backward
                player.move(0, 1, self.width, self.height, self.players)
            elif action == 2:  # left
                player.move(-1, 0, self.width, self.height, self.players)
            elif action == 3:  # right
                player.move(1, 0, self.width, self.height, self.players)

    def _update_game_state(self):
        """
        Update the game state, like detect ball collisions.
        """
        self.soccer_field.check_player_ball_overlaps()

    def _calculate_reward(self):
        """
        Compute the reward based on current environment state.

        Returns:
            float: Computed reward.
        """
        return rewards.heuristic.reward_function(self)
        # return rewards.checkpoint.reward_function(self)

    def _is_done(self):
        """
        Check whether the episode should terminate due to a goal.

        Returns:
            bool: Whether a terminal condition is met.
        """
        return self.soccer_field.check_goal()[0]

    def _is_truncated(self):
        """
        Check whether the episode exceeded the time limit.

        Returns:
            bool: Whether the episode is truncated.
        """
        elapsed_time = (pygame.time.get_ticks() - self.soccer_field.start_time) / 1000
        return elapsed_time > self.game_duration

    def _render_frame(self):
        """
        Render the game frame based on the render mode.

        Returns:
            np.ndarray or None: Image frame if 'rgb_array', else None.
        """
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
        """
        Render the environment externally.

        Returns:
            np.ndarray or None: Rendered frame if applicable.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        """
        Close the environment and Pygame resources.
        """
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./model_checkpoints/",
        name_prefix="soccer_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    def make_train_env():
        """
        Factory function to create a training environment.

        Returns:
            gym.Env: A wrapped and monitored SoccerFieldEnv.
        """
        base_env = Monitor(SoccerFieldEnv(render_mode="rgb_array", game_duration=30))
        return RewardTracker(base_env)

    # === Training and Evaluation Environments ===
    env = DummyVecEnv([make_train_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    env.seed(SEED)

    def make_eval_env():
        """
        Factory function to create an evaluation environment.

        Returns:
            gym.Env: A monitored SoccerFieldEnv instance.
        """
        return Monitor(SoccerFieldEnv(render_mode="rgb_array", game_duration=30))

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)

    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    # === Logging Callback ===
    reward_logging_callback = RewardLoggingCallback(
        reward_tracker=env.venv.envs[0],
        eval_env=eval_env,
        log_dir="./reward_logs/",
        eval_freq=50000,
    )

    # === Model Configuration ===
    policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[128, 128, 64])])

    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=9.374410314646429e-05,
        n_steps=3296,
        gamma=0.99,
        gae_lambda=0.98,
        ent_coef=0.010549674409905044,
        clip_range=0.3941564070073835,
        batch_size=3296,
        vf_coef=0.3,
    )

    # === Training ===
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, reward_logging_callback],
    )

    # === Saving ===
    model.save("soccer_agent_ppo")
    env.save("vec_normalize.pkl")

    env.close()
