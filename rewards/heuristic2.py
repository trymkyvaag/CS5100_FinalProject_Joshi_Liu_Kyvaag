import numpy as np


def reward_function(self):
    reward = 0
    field_diagonal = np.sqrt(self.width**2 + self.height**2)

    blue1, blue2 = self.players[0], self.players[1]
    red1, red2 = self.players[2], self.players[3]

    # 1. Goal Reward (SIGNIFICANTLY INCREASED)
    scored, team = self.soccer_field.check_goal()
    if scored:
        if team == "Blue: Goal":
            self.scoring_team = "blue"
            return 20.0  # Doubled reward for scoring
        elif team == "Blue: Own Goal":
            self.scoring_team = "blue"
            return -15.0
        elif team == "Red: Goal":
            self.scoring_team = "red"
            return -15.0
        elif team == "Red: Own Goal":
            self.scoring_team = "red"
            return 20.0  # Doubled reward for opponent own goal

    # Initialize previous positions if not available
    if not hasattr(self, "prev_positions"):
        self.prev_positions = {
            "blue1": (blue1.x, blue1.y),
            "blue2": (blue2.x, blue2.y),
            "red1": (red1.x, red1.y),
            "red2": (red2.x, red2.y),
            "ball": (self.ball.x, self.ball.y),
        }
        
    # Also track previous possession
    if not hasattr(self, "prev_possession"):
        self.prev_possession = {"blue1": False, "blue2": False}
        self.successful_passes = 0
        self.forward_progress = 0
        self.possession_duration = 0
    
    # 2. Calculate distances to ball
    blue1_ball_dist = np.sqrt((blue1.x - self.ball.x)**2 + (blue1.y - self.ball.y)**2)
    blue2_ball_dist = np.sqrt((blue2.x - self.ball.x)**2 + (blue2.y - self.ball.y)**2)
    red1_ball_dist = np.sqrt((red1.x - self.ball.x)**2 + (red1.y - self.ball.y)**2)
    red2_ball_dist = np.sqrt((red2.x - self.ball.x)**2 + (red2.y - self.ball.y)**2)
    
    # Define goal positions
    opponent_goal_x = self.width - 10
    opponent_goal_y = self.height // 2
    own_goal_x = 10
    own_goal_y = self.height // 2
    
    # 3. STRONG BALL PROXIMITY REWARD - Make this a top priority
    # Normalized distances (0 to 1)
    norm_blue1_dist = blue1_ball_dist / field_diagonal
    norm_blue2_dist = blue2_ball_dist / field_diagonal
    norm_red1_dist = red1_ball_dist / field_diagonal
    norm_red2_dist = red2_ball_dist / field_diagonal
    
    # Calculate which blue player is closer to ball
    closest_blue_dist = min(norm_blue1_dist, norm_blue2_dist)
    
    # Strong reward for blue players being close to the ball - INCREASED SUBSTANTIALLY
    # Especially reward the closest player approaching the ball
    reward += 2.0 * (1 - closest_blue_dist)  # Major reward for closest player
    reward += 0.8 * (1 - max(norm_blue1_dist, norm_blue2_dist))  # Smaller reward for other player
    
    # Penalty if red players are closer to ball than blue
    if min(norm_red1_dist, norm_red2_dist) < closest_blue_dist:
        reward -= 1.0  # Strong penalty when red is closer
    
    # 4. BALL POSSESSION - Even stronger reward
    blue1_possession = blue1_ball_dist < blue1.radius * 1.5
    blue2_possession = blue2_ball_dist < blue2.radius * 1.5
    blue_possession = blue1_possession or blue2_possession
    
    # Very strong reward for having possession
    if blue_possession:
        reward += 3.0  # Major reward for possession
        self.possession_duration += 1
        
        # Additional reward for keeping possession over time
        if self.possession_duration > 5:
            reward += 0.2 * min(self.possession_duration / 10, 1.0)
    else:
        self.possession_duration = 0
    
    # 5. PASSING REWARD: Check if possession changed between blue players - INCREASED
    if hasattr(self, "prev_possession"):
        # Detect successful pass: possession changed from one blue player to the other
        if (self.prev_possession["blue1"] and blue2_possession) or \
           (self.prev_possession["blue2"] and blue1_possession):
            # Strong reward for successful pass
            reward += 3.0  # Doubled from previous version
            self.successful_passes += 1
            
            # Extra reward if the pass was toward the opponent's goal
            prev_ball_x = self.prev_positions["ball"][0]
            if self.ball.x > prev_ball_x:
                reward += 1.5  # Increased forward passing reward
    
    # 6. FORWARD MOVEMENT REWARD with ball
    if blue_possession:
        if hasattr(self, "prev_positions"):
            prev_ball_x = self.prev_positions["ball"][0]
            forward_progress = self.ball.x - prev_ball_x
            
            # Reward for moving forward with the ball
            if forward_progress > 0:
                # Scale based on distance moved forward
                progress_ratio = forward_progress / self.width
                reward += 1.5 * progress_ratio * 10  # Increased reward for forward movement
                self.forward_progress += forward_progress
            
            # Smaller penalty for moving backward
            elif forward_progress < 0:
                reward -= 0.2 * abs(forward_progress) / self.width
    
    # 7. BALL RECOVERY - Reward getting the ball back from opponents
    if blue_possession and hasattr(self, "prev_positions"):
        # Check if neither blue player had possession in previous step
        if not self.prev_possession["blue1"] and not self.prev_possession["blue2"]:
            reward += 2.0  # Strong reward for recovering the ball
    
    # 8. Movement toward ball when not in possession
    if not blue_possession:
        if hasattr(self, "prev_positions"):
            # Check if players are moving toward the ball
            prev_blue1_ball_dist = np.sqrt(
                (self.prev_positions["blue1"][0] - self.prev_positions["ball"][0])**2 +
                (self.prev_positions["blue1"][1] - self.prev_positions["ball"][1])**2
            )
            prev_blue2_ball_dist = np.sqrt(
                (self.prev_positions["blue2"][0] - self.prev_positions["ball"][0])**2 +
                (self.prev_positions["blue2"][1] - self.prev_positions["ball"][1])**2
            )
            
            # Reward for decreasing distance to ball
            if blue1_ball_dist < prev_blue1_ball_dist:
                reward += 0.8 * ((prev_blue1_ball_dist - blue1_ball_dist) / field_diagonal) * 10
            if blue2_ball_dist < prev_blue2_ball_dist:
                reward += 0.8 * ((prev_blue2_ball_dist - blue2_ball_dist) / field_diagonal) * 10
    
    # 9. SHOOTING REWARD - Reward for shots toward the goal when near opponent's half
    if blue_possession and self.ball.x > self.width / 2:
        ball_velocity_angle = np.arctan2(self.ball.velocity[1], self.ball.velocity[0])
        angle_to_opp_goal = np.arctan2(
            opponent_goal_y - self.ball.y, opponent_goal_x - self.ball.x
        )
        
        angle_diff_opp = min(
            abs(angle_to_opp_goal - ball_velocity_angle),
            2 * np.pi - abs(angle_to_opp_goal - ball_velocity_angle)
        ) / np.pi
        
        # Strong reward for shooting toward goal
        if np.linalg.norm(self.ball.velocity) > 1.0 and angle_diff_opp < 0.3:
            # Distance-based multiplier (more reward when closer to goal)
            goal_dist = np.sqrt((self.ball.x - opponent_goal_x)**2 + (self.ball.y - opponent_goal_y)**2)
            goal_dist_factor = 1 - min(goal_dist / self.width, 1.0)
            reward += 2.0 * (1 - angle_diff_opp) * goal_dist_factor
    
    # 10. TEAM COORDINATION - Positioning for passing
    if blue_possession:
        # Identify the player with the ball
        ball_carrier = blue1 if blue1_possession else blue2
        receiver = blue2 if blue1_possession else blue1
        
        # Reward receiver for being ahead of ball carrier (toward opponent goal)
        if receiver.x > ball_carrier.x:
            reward += 0.6
            
            # Extra reward for being in a good passing lane (not too far away)
            pass_dist = np.sqrt((receiver.x - ball_carrier.x)**2 + (receiver.y - ball_carrier.y)**2)
            optimal_pass_dist = self.width / 5
            pass_dist_factor = 1 - min(abs(pass_dist - optimal_pass_dist) / optimal_pass_dist, 1.0)
            reward += 0.4 * pass_dist_factor
    
    # 11. Moderate time penalty to encourage faster play but not overly penalize
    reward -= 0.005
    
    # Update previous states for next iteration
    self.prev_positions = {
        "blue1": (blue1.x, blue1.y),
        "blue2": (blue2.x, blue2.y),
        "red1": (red1.x, red1.y),
        "red2": (red2.x, red2.y),
        "ball": (self.ball.x, self.ball.y),
    }
    
    self.prev_possession = {
        "blue1": blue1_possession,
        "blue2": blue2_possession
    }
    
    return reward