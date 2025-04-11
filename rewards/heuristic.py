import numpy as np


def reward_function(self):
    """
    A reward function to guide agent behavior in a 2v2 soccer game.

    This function encourages scoring goals, approaching the ball, maintaining
    team formation, defending own goal, and penalizes bad behavior
    such as stagnation or own goals.
    """
    reward = 0

    # Calculate the diagonal of the field (used for normalizing distances)
    field_diagonal = np.sqrt(self.width**2 + self.height**2)

    # Extract players from both teams
    blue1, blue2 = self.players[0], self.players[1]
    red1, red2 = self.players[2], self.players[3]

    # 1. Goal Reward — large reward/penalty for scoring or conceding
    scored, team = self.soccer_field.check_goal()
    if scored:
        if team == "Blue: Goal":
            self.scoring_team = "blue"
            return 10.0  # positive reward for scoring
        elif team == "Blue: Own Goal":
            self.scoring_team = "blue"
            return -10.0  # penalty for own goal
        elif team == "Red: Goal":
            self.scoring_team = "red"
            return -10.0  # penalty for conceding
        elif team == "Red: Own Goal":
            self.scoring_team = "red"
            return 10.0  # reward if opponent scores own goal

    # Save previous positions for tracking movement and trends
    if not hasattr(self, "prev_positions"):
        self.prev_positions = {
            "blue1": (blue1.x, blue1.y),
            "blue2": (blue2.x, blue2.y),
            "red1": (red1.x, red1.y),
            "red2": (red2.x, red2.y),
            "ball": (self.ball.x, self.ball.y),
        }

    # 2. Ball Proximity — reward blue team for staying closer to the ball
    blue1_ball_dist = np.linalg.norm([blue1.x - self.ball.x, blue1.y - self.ball.y])
    blue2_ball_dist = np.linalg.norm([blue2.x - self.ball.x, blue2.y - self.ball.y])
    red1_ball_dist = np.linalg.norm([red1.x - self.ball.x, red1.y - self.ball.y])
    red2_ball_dist = np.linalg.norm([red2.x - self.ball.x, red2.y - self.ball.y])

    # Normalize distances to field size
    norm_blue1_dist = blue1_ball_dist / field_diagonal
    norm_blue2_dist = blue2_ball_dist / field_diagonal
    norm_red1_dist = red1_ball_dist / field_diagonal
    norm_red2_dist = red2_ball_dist / field_diagonal

    # Encourage blue to approach the ball and discourage red from doing so
    reward += 0.5 * (1 - norm_blue1_dist) + 0.5 * (1 - norm_blue2_dist)
    reward += 0.5 * norm_red1_dist + 0.5 * norm_red2_dist

    # 3. Ball Possession — small reward for being close enough to possess the ball
    blue_possession = False
    if blue1_ball_dist < blue1.radius * 1.5 or blue2_ball_dist < blue2.radius * 1.5:
        blue_possession = True
        reward += 0.5

    # 4. Ball Movement Direction — reward if ball is moving toward opponent goal
    ball_velocity_angle = np.arctan2(self.ball.velocity[1], self.ball.velocity[0])

    opponent_goal_x, opponent_goal_y = self.width - 10, self.height // 2
    own_goal_x, own_goal_y = 10, self.height // 2

    angle_to_opp_goal = np.arctan2(
        opponent_goal_y - self.ball.y, opponent_goal_x - self.ball.x
    )
    angle_to_own_goal = np.arctan2(own_goal_y - self.ball.y, own_goal_x - self.ball.x)

    # Calculate angle difference (range 0 to 1) — 0 means perfect alignment
    angle_diff_opp = (
        min(
            abs(angle_to_opp_goal - ball_velocity_angle),
            2 * np.pi - abs(angle_to_opp_goal - ball_velocity_angle),
        )
        / np.pi
    )
    angle_diff_own = (
        min(
            abs(angle_to_own_goal - ball_velocity_angle),
            2 * np.pi - abs(angle_to_own_goal - ball_velocity_angle),
        )
        / np.pi
    )

    if np.linalg.norm(self.ball.velocity) > 0.5:
        reward += 0.5 * (1 - angle_diff_opp)  # reward if going toward opponent goal
        reward += 0.3 * angle_diff_own  # penalize if moving toward own goal

    # 5. Strategic Positioning — reward if ball moves away from own goal
    ball_to_own_goal = np.linalg.norm(
        [self.ball.x - own_goal_x, self.ball.y - own_goal_y]
    )
    norm_ball_to_own_goal = ball_to_own_goal / field_diagonal

    if hasattr(self, "prev_positions"):
        prev_ball_to_own_goal = np.linalg.norm(
            [
                self.prev_positions["ball"][0] - own_goal_x,
                self.prev_positions["ball"][1] - own_goal_y,
            ]
        )
        norm_prev_ball_to_own_goal = prev_ball_to_own_goal / field_diagonal

        # Encourage pushing the ball away from own goal
        reward -= 0.3 * -(norm_ball_to_own_goal - norm_prev_ball_to_own_goal)

    # 6. Team Coordination — reward optimal spacing between blue players
    blue_team_dist = np.linalg.norm([blue1.x - blue2.x, blue1.y - blue2.y])
    optimal_spacing = self.width / 4
    norm_spacing_diff = min(
        abs(blue_team_dist - optimal_spacing) / optimal_spacing, 1.0
    )
    reward += 0.2 * (1 - norm_spacing_diff)

    # Additional possession-based reward scaled by ball position (further is better)
    if blue_possession:
        position_multiplier = self.ball.x / self.width  # further into red territory
        reward += 0.5 * (1 + position_multiplier)

    # 7. Interception & Movement — penalize red for inaction and penalize blue if red is intercepting
    if hasattr(self, "prev_positions"):
        red1_speed = np.linalg.norm(
            [
                red1.x - self.prev_positions["red1"][0],
                red1.y - self.prev_positions["red1"][1],
            ]
        )
        red2_speed = np.linalg.norm(
            [
                red2.x - self.prev_positions["red2"][0],
                red2.y - self.prev_positions["red2"][1],
            ]
        )

        red1_intercept_angle = np.arctan2(self.ball.y - red1.y, self.ball.x - red1.x)
        red2_intercept_angle = np.arctan2(self.ball.y - red2.y, self.ball.x - red2.x)

        if red1_speed < 0.1 and red1_ball_dist > red1.radius * 3:
            reward -= 0.1  # penalize red1 for standing still too far from ball
        if red2_speed < 0.1 and red2_ball_dist > red2.radius * 3:
            reward -= 0.1

        # If red is intercepting, it's bad for blue — reduce reward
        if red1_speed > 0.1:
            red1_movement_angle = np.arctan2(
                red1.y - self.prev_positions["red1"][1],
                red1.x - self.prev_positions["red1"][0],
            )
            red1_angle_diff = (
                min(
                    abs(red1_intercept_angle - red1_movement_angle),
                    2 * np.pi - abs(red1_intercept_angle - red1_movement_angle),
                )
                / np.pi
            )
            reward -= 0.2 * (1 - red1_angle_diff)

        if red2_speed > 0.1:
            red2_movement_angle = np.arctan2(
                red2.y - self.prev_positions["red2"][1],
                red2.x - self.prev_positions["red2"][0],
            )
            red2_angle_diff = (
                min(
                    abs(red2_intercept_angle - red2_movement_angle),
                    2 * np.pi - abs(red2_intercept_angle - red2_movement_angle),
                )
                / np.pi
            )
            reward -= 0.2 * (1 - red2_angle_diff)

    # 8. Defensive Positioning — reward blue if closer to own goal than the ball
    blue_to_own_goal = min(
        np.linalg.norm([blue1.x - own_goal_x, blue1.y - own_goal_y]),
        np.linalg.norm([blue2.x - own_goal_x, blue2.y - own_goal_y]),
    )
    ball_to_own_goal = np.linalg.norm(
        [self.ball.x - own_goal_x, self.ball.y - own_goal_y]
    )
    if blue_to_own_goal < ball_to_own_goal:
        reward += 0.3  # incentivize falling back when needed

    # 9. Time Penalty — encourage to try by penalizing time steps
    reward -= 0.01

    # Update previous positions for next step tracking
    self.prev_positions = {
        "blue1": (blue1.x, blue1.y),
        "blue2": (blue2.x, blue2.y),
        "red1": (red1.x, red1.y),
        "red2": (red2.x, red2.y),
        "ball": (self.ball.x, self.ball.y),
    }

    return reward
