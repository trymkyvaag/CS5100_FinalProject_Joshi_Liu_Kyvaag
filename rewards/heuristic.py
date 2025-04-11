import numpy as np


def reward_function(self):
    reward = 0
    field_diagonal = np.sqrt(self.width**2 + self.height**2)

    blue1, blue2 = self.players[0], self.players[1]
    red1, red2 = self.players[2], self.players[3]

    # 1. Goal Reward
    scored, team = self.soccer_field.check_goal()
    if scored:
        if team == "Blue: Goal":
            self.scoring_team = "blue"
            return 10.0
        elif team == "Blue: Own Goal":
            self.scoring_team = "blue"
            return -10.0
        elif team == "Red: Goal":
            self.scoring_team = "red"
            return -10.0
        elif team == "Red: Own Goal":
            self.scoring_team = "red"
            return 10.0

    if not hasattr(self, "prev_positions"):
        self.prev_positions = {
            "blue1": (blue1.x, blue1.y),
            "blue2": (blue2.x, blue2.y),
            "red1": (red1.x, red1.y),
            "red2": (red2.x, red2.y),
            "ball": (self.ball.x, self.ball.y),
        }

    # 2. Ball Proximity (normalized to 0-1 range)
    blue1_ball_dist = np.sqrt(
        (blue1.x - self.ball.x) ** 2 + (blue1.y - self.ball.y) ** 2
    )
    blue2_ball_dist = np.sqrt(
        (blue2.x - self.ball.x) ** 2 + (blue2.y - self.ball.y) ** 2
    )
    red1_ball_dist = np.sqrt((red1.x - self.ball.x) ** 2 + (red1.y - self.ball.y) ** 2)
    red2_ball_dist = np.sqrt((red2.x - self.ball.x) ** 2 + (red2.y - self.ball.y) ** 2)

    norm_blue1_dist = blue1_ball_dist / field_diagonal
    norm_blue2_dist = blue2_ball_dist / field_diagonal
    norm_red1_dist = red1_ball_dist / field_diagonal
    norm_red2_dist = red2_ball_dist / field_diagonal

    # reward blue for being closer to the ball
    reward += 0.5 * (1 - norm_blue1_dist) + 0.5 * (1 - norm_blue2_dist)
    reward += 0.5 * norm_red1_dist + 0.5 * norm_red2_dist

    # 3. Ball Possession
    blue_possession = False
    if blue1_ball_dist < blue1.radius * 1.5 or blue2_ball_dist < blue2.radius * 1.5:
        blue_possession = True
        reward += 0.5

    # 4. Ball Movement Direction
    ball_velocity_angle = np.arctan2(self.ball.velocity[1], self.ball.velocity[0])

    # check where the ball is headed to
    opponent_goal_x = self.width - 10
    opponent_goal_y = self.height // 2
    own_goal_x = 10
    own_goal_y = self.height // 2

    # is it in the direction?
    angle_to_opp_goal = np.arctan2(
        opponent_goal_y - self.ball.y, opponent_goal_x - self.ball.x
    )
    angle_to_own_goal = np.arctan2(own_goal_y - self.ball.y, own_goal_x - self.ball.x)

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
        # good we are in red's direction
        reward += 0.5 * (1 - angle_diff_opp)
        # bad we are in blue's direction
        reward += 0.3 * angle_diff_own

    # 5. Strategic Positioning  red
    ball_to_own_goal = np.sqrt(
        (self.ball.x - own_goal_x) ** 2 + (self.ball.y - own_goal_y) ** 2
    )
    norm_ball_to_own_goal = ball_to_own_goal / field_diagonal

    if hasattr(self, "prev_positions"):
        prev_ball_to_own_goal = np.sqrt(
            (self.prev_positions["ball"][0] - own_goal_x) ** 2
            + (self.prev_positions["ball"][1] - own_goal_y) ** 2
        )
        norm_prev_ball_to_own_goal = prev_ball_to_own_goal / field_diagonal
        # red should push the ball closer to blue's goal, if so blue should be penalized.
        reward -= 0.3 * -(norm_ball_to_own_goal - norm_prev_ball_to_own_goal)

    # 6. Team Coordination
    blue_team_dist = np.sqrt((blue1.x - blue2.x) ** 2 + (blue1.y - blue2.y) ** 2)
    optimal_spacing = self.width / 4
    norm_spacing_diff = min(
        abs(blue_team_dist - optimal_spacing) / optimal_spacing, 1.0
    )
    reward += 0.2 * (1 - norm_spacing_diff)

    if blue_possession:
        possession_reward = 0.5
        position_multiplier = self.ball.x / self.width
        reward += possession_reward * (1 + position_multiplier)

    if hasattr(self, "prev_positions"):
        # 7. Player Movement and Interception
        red1_speed = np.sqrt(
            (red1.x - self.prev_positions["red1"][0]) ** 2
            + (red1.y - self.prev_positions["red1"][1]) ** 2
        )
        red2_speed = np.sqrt(
            (red2.x - self.prev_positions["red2"][0]) ** 2
            + (red2.y - self.prev_positions["red2"][1]) ** 2
        )

        red1_intercept_angle = np.arctan2(self.ball.y - red1.y, self.ball.x - red1.x)
        red2_intercept_angle = np.arctan2(self.ball.y - red2.y, self.ball.x - red2.x)

        # try intercepting if red is stagnant.
        if red1_speed < 0.1 and red1_ball_dist > red1.radius * 3:
            reward -= 0.1
        if red2_speed < 0.1 and red2_ball_dist > red2.radius * 3:
            reward -= 0.1

        # if red can intercept then negatively reward blue
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

    # 8. Defensive Positioning for blue team, encourage them closer to each other.
    blue_to_own_goal = min(
        np.sqrt((blue1.x - own_goal_x) ** 2 + (blue1.y - own_goal_y) ** 2),
        np.sqrt((blue2.x - own_goal_x) ** 2 + (blue2.y - own_goal_y) ** 2),
    )
    ball_to_own_goal = np.sqrt(
        (self.ball.x - own_goal_x) ** 2 + (self.ball.y - own_goal_y) ** 2
    )

    if blue_to_own_goal < ball_to_own_goal:
        reward += 0.3

    # 9. Time Penalty.
    reward -= 0.01

    self.prev_positions = {
        "blue1": (blue1.x, blue1.y),
        "blue2": (blue2.x, blue2.y),
        "red1": (red1.x, red1.y),
        "red2": (red2.x, red2.y),
        "ball": (self.ball.x, self.ball.y),
    }

    return reward
