import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Visual_Components.field import SoccerField


class SoccerFieldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, game_duration=60, render_mode=None):
        super(SoccerFieldEnv, self).__init__()

        self.soccer_field = SoccerField(game_duration=game_duration)
        self.width = self.soccer_field.width
        self.height = self.soccer_field.height
        self.game_duration = game_duration
        self.players = self.soccer_field.players
        self.ball = self.soccer_field.ball
        self.screen = self.soccer_field.screen
        self.clock = self.soccer_field.clock

        self.action_space = spaces.Discrete(5)

        low = np.array(
            [
                0,  # Player 1 X
                0,  # Player 1 Y
                0,  # Player 2 X
                0,  # Player 2 Y
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
                self.width,  # Player 1 X
                self.height,  # Player 1 Y
                self.width,  # Player 2 X
                self.height,  # Player 2 Y
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
        player_blue = self.players[0]
        player_red = self.players[1]
        remaining_time = (
            self.soccer_field.game_duration
            - (pygame.time.get_ticks() - self.soccer_field.start_time) / 1000
        )
        observation = np.array(
            [
                player_blue.x,
                player_blue.y,
                player_red.x,
                player_red.y,
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
        self._take_action(action)

        self._update_game_state()

        reward = self._calculate_reward()

        terminated = self._is_done()
        truncated = self._is_truncated()

        observation = self._get_observation()

        info = {}
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _take_action(self, action):
        player = self.players[0]
        if not player.frozen:
            if not self.soccer_field.kickoff_started:
                self.soccer_field.unfreeze_team("red")
                self.soccer_field.kickoff_started = True
            if action == 0:  # forward
                player.move(0, -1, self.width, self.height, self.players)
            elif action == 1:  # backward
                player.move(0, 1, self.width, self.height, self.players)
            elif action == 2:  # left
                player.move(-1, 0, self.width, self.height, self.players)
            elif action == 3:  # right
                player.move(1, 0, self.width, self.height, self.players)

    def _update_game_state(self):
        self.soccer_field.check_player_ball_overlaps()
        self.soccer_field.check_goal()

    def _calculate_reward(self):
        if self.soccer_field.check_goal():
            if self.ball.last_touched_by == "blue":
                reward = 1
            else:
                reward = -1
        else:
            player = self.players[0]
            distance = np.sqrt(
                (player.x - self.ball.x) ** 2 + (player.y - self.ball.y) ** 2
            )
            reward = -distance / 1000
        reward -= 0.001

        return reward

    def _is_done(self):
        return self.soccer_field.check_goal()

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
            import pygame

            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    dummy_env = SoccerFieldEnv(render_mode="human", game_duration=30)

    env = DummyVecEnv([lambda: dummy_env])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    model.save("soccer_agent_ppo")

    env.close()
