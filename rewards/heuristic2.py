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
    red1_ball_dist = np.sqrt((red1.x - self.ball.x) **
                             2 + (red1.y - self.ball.y) ** 2)
    red2_ball_dist = np.sqrt((red2.x - self.ball.x) **
                             2 + (red2.y - self.ball.y) ** 2)

    norm_blue1_dist = blue1_ball_dist / field_diagonal
    norm_blue2_dist = blue2_ball_dist / field_diagonal
    norm_red1_dist = red1_ball_dist / field_diagonal
    norm_red2_dist = red2_ball_dist / field_diagonal

    # reward blue for being closer to the ball
    reward += 0.5 * (1 - norm_blue1_dist) + 0.5 * (1 - norm_blue2_dist)
    reward += 0.5 * norm_red1_dist + 0.5 * norm_red2_dist

    # 3. Ball Possession
    blue_possession = False
    red_possession = False
    if blue1_ball_dist < blue1.radius * 1.5 or blue2_ball_dist < blue2.radius * 1.5:
        blue_possession = True
        reward += 0.5
    if red1_ball_dist < red1.radius * 1.5 or red2_ball_dist < red2.radius * 1.5:
        red_possession = True
        reward -= 0.5  # Penalty for blue when red has possession

    # 4. Ball Movement Direction
    ball_velocity_angle = np.arctan2(
        self.ball.velocity[1], self.ball.velocity[0])

    # check where the ball is headed to
    opponent_goal_x = self.width - 10
    opponent_goal_y = self.height // 2
    own_goal_x = 10
    own_goal_y = self.height // 2

    # is it in the direction?
    angle_to_opp_goal = np.arctan2(
        opponent_goal_y - self.ball.y, opponent_goal_x - self.ball.x
    )
    angle_to_own_goal = np.arctan2(
        own_goal_y - self.ball.y, own_goal_x - self.ball.x)

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

    # Better reward for moving ball toward opponent goal
    if np.linalg.norm(self.ball.velocity) > 0.5:
        reward += 0.8 * (1 - angle_diff_opp)  # Increased from 0.5
        reward += 0.5 * angle_diff_own  # Increased from 0.3

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
    blue_team_dist = np.sqrt((blue1.x - blue2.x) **
                             2 + (blue1.y - blue2.y) ** 2)
    optimal_spacing = self.width / 4
    norm_spacing_diff = min(
        abs(blue_team_dist - optimal_spacing) / optimal_spacing, 1.0
    )
    reward += 0.2 * (1 - norm_spacing_diff)

    if blue_possession:
        # Exponential reward for advancing with ball possession
        # Squared for emphasis!!!
        position_multiplier = (self.ball.x / self.width) ** 2
        reward += 0.7 * (1 + position_multiplier)  # Increased from 0.5

    # NEW: Reward red for being positioned between the ball and blue's goal
    # This encourages defensive positioning and intercepting the ball path
    for red_player in [red1, red2]:
        # Vector from ball to goal
        ball_to_goal_vector = [opponent_goal_x - self.ball.x, opponent_goal_y - self.ball.y]
        ball_to_goal_dist = np.sqrt(ball_to_goal_vector[0]**2 + ball_to_goal_vector[1]**2)
        
        # Vector from ball to red player
        ball_to_red_vector = [red_player.x - self.ball.x, red_player.y - self.ball.y]
        ball_to_red_dist = np.sqrt(ball_to_red_vector[0]**2 + ball_to_red_vector[1]**2)
        
        # Dot product to see if red is in front of ball toward blue's goal
        if ball_to_goal_dist > 0:  # Avoid division by zero
            dot_product = (ball_to_red_vector[0] * ball_to_goal_vector[0] + 
                          ball_to_red_vector[1] * ball_to_goal_vector[1]) / ball_to_goal_dist
            
            # Normalize to get a value between 0 and 1
            # Higher value means red is more directly in front of the ball toward goal
            in_front_factor = max(0, min(dot_product / ball_to_goal_dist, 1))
            
            # Penalize blue (reward red) for good defensive positioning
            reward -= 0.4 * in_front_factor
    
    # NEW: Reward red for moving toward the ball (trying to gain possession)
    if hasattr(self, "prev_positions"):
        for idx, red_player in enumerate([red1, red2]):
            player_key = f"red{idx+1}"
            prev_dist = np.sqrt(
                (self.prev_positions[player_key][0] - self.prev_positions["ball"][0])**2 +
                (self.prev_positions[player_key][1] - self.prev_positions["ball"][1])**2
            )
            current_dist = np.sqrt(
                (red_player.x - self.ball.x)**2 + (red_player.y - self.ball.y)**2
            )
            
            # If red is moving closer to the ball
            if current_dist < prev_dist:
                distance_improvement = (prev_dist - current_dist) / field_diagonal
                reward -= 0.3 * distance_improvement  # Penalize blue (reward red)

    if hasattr(self, "prev_positions"):
        # 7. Player Movement and Interception - MODIFIED to be more balanced
        red1_speed = np.sqrt(
            (red1.x - self.prev_positions["red1"][0]) ** 2
            + (red1.y - self.prev_positions["red1"][1]) ** 2
        )
        red2_speed = np.sqrt(
            (red2.x - self.prev_positions["red2"][0]) ** 2
            + (red2.y - self.prev_positions["red2"][1]) ** 2
        )

        red1_intercept_angle = np.arctan2(
            self.ball.y - red1.y, self.ball.x - red1.x)
        red2_intercept_angle = np.arctan2(
            self.ball.y - red2.y, self.ball.x - red2.x)

        # REMOVED the penalty for red being stagnant - this was causing issues
        # Now we'll only reward meaningful movement

        # Reward red for moving toward the ball (improved interception)
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
            # Better reward for red moving toward ball
            reward -= 0.3 * (1 - red1_angle_diff)  # Increased from 0.2

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
            # Better reward for red moving toward ball
            reward -= 0.3 * (1 - red2_angle_diff)  # Increased from 0.2
    
    # NEW: Add penalty for red players hanging out in corners
    corner_penalty = 0.0
    corners = [(0,0), (0,self.height), (self.width,0), (self.width,self.height)]
    for red_player in [red1, red2]:
        min_corner_dist = min([np.sqrt((red_player.x - cx)**2 + (red_player.y - cy)**2) for cx, cy in corners])
        corner_proximity = 1.0 - min(min_corner_dist / (self.width/5), 1.0)  # Normalize
        if corner_proximity > 0.7:  # Only penalize when very close to corners
            corner_penalty += 0.7 * corner_proximity
    reward += corner_penalty  # Positive for blue (penalty for red)

    # 8. Defensive Positioning for blue team, encourage them closer to each other.
    blue_to_own_goal = min(
        np.sqrt((blue1.x - own_goal_x) ** 2 + (blue1.y - own_goal_y) ** 2),
        np.sqrt((blue2.x - own_goal_x) ** 2 + (blue2.y - own_goal_y) ** 2),
    )
    ball_to_own_goal = np.sqrt(
        (self.ball.x - own_goal_x) ** 2 + (self.ball.y - own_goal_y) ** 2
    )

    # Normal Soccer tactics - one player forward, one back when not possessing
    if not blue_possession:
        front_player_x = max(blue1.x, blue2.x)
        back_player_x = min(blue1.x, blue2.x)
        field_coverage = (front_player_x - back_player_x) / self.width  

        if field_coverage < 0.3:
            reward += 0.4 * field_coverage
        else:
            reward -= 1  # make sure they do not go tooo far

    # Track ball possession changes
    if hasattr(self, "prev_possession"):
        # If ball was previously not possessed by blue but now is by the other blue player
        if not self.prev_possession and blue_possession:
            # Determine which blue player has possession
            current_blue_with_ball = 1 if blue1_ball_dist < blue1.radius * 1.5 else 2
            prev_blue_with_ball = self.prev_blue_with_ball if hasattr(
                self, "prev_blue_with_ball") else None

            # If possession changed between blue players (successful pass)
            if prev_blue_with_ball is not None and current_blue_with_ball != prev_blue_with_ball:
                reward += 0.8  # Substantial reward for successful passing

        # Track which blue player last had ball
        if blue_possession:
            self.prev_blue_with_ball = 1 if blue1_ball_dist < blue1.radius * 1.5 else 2

    self.prev_possession = blue_possession

    if blue_to_own_goal < ball_to_own_goal:
        reward += 0.3

    # 9. Time Penalty.
    # Reduced time penalty
    reward -= 0.005  # Changed from 0.01

    self.prev_positions = {
        "blue1": (blue1.x, blue1.y),
        "blue2": (blue2.x, blue2.y),
        "red1": (red1.x, red1.y),
        "red2": (red2.x, red2.y),
        "ball": (self.ball.x, self.ball.y),
    }

    return reward