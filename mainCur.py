import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

import rewards.heuristic
from Visual_Components.field import SoccerField


class CurriculumCallback(BaseCallback):
    """
    Custom callback for curriculum learning progression based on timesteps
    """

    def __init__(self, env, curriculum_stages, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.env = env
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.next_stage_timestep = curriculum_stages[0]['timestep']

    def _on_step(self) -> bool:
        if (self.current_stage < len(self.curriculum_stages) - 1 and
                self.num_timesteps >= self.next_stage_timestep):

            self.current_stage += 1
            next_stage = self.curriculum_stages[self.current_stage]
            self.next_stage_timestep = next_stage['timestep']

            # Update the environment with new curriculum settings
            self.env.set_curriculum_stage(next_stage)

            if self.verbose > 0:
                print(
                    f"Curriculum advanced to stage {self.current_stage + 1}: {next_stage['name']}")

        return True


class SoccerFieldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, game_duration=30, render_mode=None):
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

        # Curriculum learning parameters
        self.curriculum_config = {
            'blue_player_1_disabled': False,
            'blue_player_2_disabled': False,
            'red_player_1_disabled': False,
            'red_player_2_disabled': False,
            'red_player_1_moving': False,
            'red_player_2_moving': False
        }

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

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

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

    def set_curriculum_stage(self, stage_config):
        """Update environment based on curriculum stage configuration"""
        self.curriculum_config.update(stage_config)

        self.players[0].disabled = self.curriculum_config['blue_player_1_disabled']
        self.players[1].disabled = self.curriculum_config['blue_player_2_disabled']
        self.players[2].disabled = self.curriculum_config['red_player_1_disabled']
        self.players[3].disabled = self.curriculum_config['red_player_2_disabled']

        print(f"Curriculum updated: {self.curriculum_config}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soccer_field.reset_positions()
        self.ball = self.soccer_field.ball
        self.players = self.soccer_field.players

        # Apply curriculum settings to players
        self.players[0].disabled = self.curriculum_config['blue_player_1_disabled']
        self.players[1].disabled = self.curriculum_config['blue_player_2_disabled']
        self.players[2].disabled = self.curriculum_config['red_player_1_disabled']
        self.players[3].disabled = self.curriculum_config['red_player_2_disabled']

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
        (
            action_agent_blue_1,
            action_agent_blue_2,
            action_agent_red_1,
            action_agent_red_2,
        ) = action

        self._take_action(action_agent_blue_1, 0)
        self._take_action(action_agent_blue_2, 1)

        if self.curriculum_config['red_player_1_moving']:
            # Implement simple AI for red player 1
            self._take_ai_action(2)
        else:
            self._take_action(action_agent_red_1, 2)

        if self.curriculum_config['red_player_2_moving']:
            # Implement simple AI for red player 2
            self._take_ai_action(3)
        else:
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

    def _take_ai_action(self, player_index):
        """Simple AI for red players that move towards the ball"""
        player = self.players[player_index]
        if player.disabled or player.frozen:
            return

        # Simple AI: move towards the ball
        dx = self.ball.x - player.x
        dy = self.ball.y - player.y

        # Determine direction with highest magnitude
        if abs(dx) > abs(dy):
            if dx > 0:
                player.move(1, 0, self.width, self.height,
                            self.players)  # right
            else:
                player.move(-1, 0, self.width, self.height,
                            self.players)  # left
        else:
            if dy > 0:
                player.move(0, 1, self.width, self.height,
                            self.players)  # down
            else:
                player.move(0, -1, self.width, self.height, self.players)  # up

    def _take_action(self, action, player_index):
        player = self.players[player_index]
        if player.disabled or player.frozen:
            return

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
        self.soccer_field.check_player_ball_overlaps()

    def _calculate_reward(self):
        return rewards.heuristic.reward_function(self)

    def _is_done(self):
        return self.soccer_field.check_goal()[0]

    def _is_truncated(self):
        elapsed_time = (pygame.time.get_ticks() -
                        self.soccer_field.start_time) / 1000
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
    # Define curriculum stages based on timesteps - always with two attackers
    curriculum_stages = [
        {
            'name': 'Stage 1 - Two attackers vs no defenders',
            'timestep': 0,
            'blue_player_1_disabled': False,
            'blue_player_2_disabled': False,
            'red_player_1_disabled': True,
            'red_player_2_disabled': True,
            'red_player_1_moving': False,
            'red_player_2_moving': False
        },
        {
            'name': 'Stage 2 - Two attackers vs one stationary defender',
            'timestep': 200000,
            'blue_player_1_disabled': False,
            'blue_player_2_disabled': False,  # One attacker stationnary
            'red_player_1_disabled': False,
            'red_player_2_disabled': True,
            'red_player_1_moving': False,
            'red_player_2_moving': False
        },
        {
            'name': 'Stage 3 - Two attackers vs one moving defender',
            'timestep': 400000,
            'blue_player_1_disabled': False,
            'blue_player_2_disabled': False,  # One active
            'red_player_1_disabled': False,
            'red_player_2_disabled': True,
            'red_player_1_moving': True,
            'red_player_2_moving': False
        },
        {
            'name': 'Stage 4 - Two attackers vs two defenders (one moving)',
            'timestep': 600000,
            'blue_player_1_disabled': False,
            'blue_player_2_disabled': False,  # Both def stationary
            'red_player_1_disabled': False,
            'red_player_2_disabled': False,
            'red_player_1_moving': True,
            'red_player_2_moving': False
        },
        {
            'name': 'Stage 5 - Two attackers vs two moving defenders',
            'timestep': 800000,
            'blue_player_1_disabled': False,
            'blue_player_2_disabled': False,  # Both def active
            'red_player_1_disabled': False,
            'red_player_2_disabled': False,
            'red_player_1_moving': True,
            'red_player_2_moving': True
        }
    ]

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./model_checkpoints/currLearn/",
        name_prefix="soccer_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    dummy_env = SoccerFieldEnv(render_mode="rgb_array", game_duration=30)

    # Set initial curriculum stage (no def)
    dummy_env.set_curriculum_stage(curriculum_stages[0])

    env = DummyVecEnv([lambda: dummy_env])

    # Create curriculum callback
    curriculum_callback = CurriculumCallback(
        dummy_env, curriculum_stages, verbose=1)

    # Combine callbacks

    callbacks = CallbackList([checkpoint_callback, curriculum_callback])

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00003)
    
    
    model.learn(total_timesteps=1000000, callback=callbacks)

    model.save("soccer_agent_curriculum_ppo")

    env.close()
