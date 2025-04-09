import sys
import pygame

from Visual_Components.ball import Ball
from Visual_Components.player import Player


class SoccerField:
    """
    Main class that represents the soccer field and manages the game.
    Handles initialization, drawing, gameplay logic, and scoring.
    """
    def __init__(self, width=600, height=400, game_duration=60):
        """
        Initialize the soccer field and game components.
        
        Parameters:
            width (int): Width of the game window in pixels
            height (int): Height of the game window in pixels
            game_duration (int): Game duration in seconds
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Soccer Field")

        # Define color constants
        self.GREEN = (34, 139, 34)  # Field color
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)

        # Create players (2 blue and 2 red)
        self.players = [
            Player(150, 200, self.BLUE, "blue"),
            Player(250, 150, self.BLUE, "blue"),
            Player(450, 250, self.RED, "red"),
            Player(550, 200, self.RED, "red"),
        ]

        # Game objects initialization
        self.clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)
        self.red_score = 0
        self.blue_score = 0
        self.scoring_team = None
        self.kickoff_started = False
        self.game_duration = game_duration
        self.start_time = pygame.time.get_ticks()

    def draw_field(self):
        """
        Draw the soccer field, goals, players, ball, scores, and timer.
        """
        # draw field
        self.screen.fill(self.GREEN)
        pygame.draw.rect(self.screen, self.BLACK, (0, 0, self.width, self.height), 10)
        pygame.draw.line(
            self.screen,
            self.WHITE,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            2,
        )
        pygame.draw.circle(
            self.screen, self.WHITE, (self.width // 2, self.height // 2), 50, 2
        )

        # Define goal dimensions
        goal_height = 100
        goal_top = (self.height - goal_height) // 2

        # Draw goal posts
        pygame.draw.rect(self.screen, self.BLACK, (0, 0, 20, goal_top))
        pygame.draw.rect(
            self.screen, self.BLACK, (0, goal_top + goal_height, 20, goal_top)
        )
        pygame.draw.rect(self.screen, self.BLACK, (self.width - 20, 0, 20, goal_top))
        pygame.draw.rect(
            self.screen,
            self.BLACK,
            (self.width - 20, goal_top + goal_height, 20, goal_top),
        )

        # Draw goal nets, player and time  
        self._draw_goal_net()
        for idx, player in enumerate(self.players):
            player.draw(self.screen)
        self.ball.draw(self.screen)
        self.draw_scores()
        self.draw_timer()

    def reset_game(self):
        """
        Reset the game by placing the ball at the center and resetting the timer.
        """
        self.ball.reset_position(self.width // 2, self.height // 2)
        self.start_time = pygame.time.get_ticks()

    def run(self):
        """
        Main game loop. Handles events, player movement, collisions, and game logic.
        """
        self.reset_game()
        game_over = False
        while not game_over:
            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            keys = pygame.key.get_pressed()
            
            # Blue team player 1 controls (WASD)
            if not self.players[0].frozen:
                if (
                    keys[pygame.K_w]
                    or keys[pygame.K_s]
                    or keys[pygame.K_a]
                    or keys[pygame.K_d]
                ):
                    # Start kickoff if this is the first movement
                    if not self.kickoff_started:
                        print("Kickoff started by Blue Team!")
                        self.unfreeze_team("red")
                        self.kickoff_started = True
                if keys[pygame.K_w]:  # Move up
                    self.players[0].move(0, -1, self.width, self.height, self.players)
                if keys[pygame.K_s]:  # Move down
                    self.players[0].move(0, 1, self.width, self.height, self.players)
                if keys[pygame.K_a]:  # Move left
                    self.players[0].move(-1, 0, self.width, self.height, self.players)
                if keys[pygame.K_d]:  # Move right
                    self.players[0].move(1, 0, self.width, self.height, self.players)

            # Red team player 1 controls (Arrow keys)
            if not self.players[1].frozen:
                if (
                    keys[pygame.K_UP]
                    or keys[pygame.K_DOWN]
                    or keys[pygame.K_LEFT]
                    or keys[pygame.K_RIGHT]
                ):
                    # Start kickoff if this is the first movement
                    if not self.kickoff_started:
                        print("Kickoff started by Red Team!")
                        self.unfreeze_team("blue")
                        self.kickoff_started = True
                if keys[pygame.K_UP]:  # Move up
                    self.players[1].move(0, -1, self.width, self.height, self.players)
                if keys[pygame.K_DOWN]:  # Move down
                    self.players[1].move(0, 1, self.width, self.height, self.players)
                if keys[pygame.K_LEFT]:  # Move left
                    self.players[1].move(-1, 0, self.width, self.height, self.players)
                if keys[pygame.K_RIGHT]:  # Move right
                    self.players[1].move(1, 0, self.width, self.height, self.players)

            # Check for overlaps and handle ball movement
            self.check_player_ball_overlaps()

            # Check if a goal was scored
            if self.check_goal()[0]:
                continue

            # Update display
            self.draw_field()
            pygame.display.flip()
            self.clock.tick(60) 

            # Check if game timer has expired
            if (pygame.time.get_ticks() - self.start_time) / 1000 > self.game_duration:
                game_over = True
                self.display_game_over()

    def check_player_ball_overlaps(self):
        """
        Check and handle player-player and player-ball collisions.
        """
        # Check and prevent collisionns
        for i, player in enumerate(self.players):
            for j, other_player in enumerate(self.players):
                if i != j:
                    player.prevent_overlap(other_player)

        # Move the ball and check for collisions
        self.ball.move()
        self.ball.check_collision_with_walls(self.width, self.height)

        # Check ball collision with each player
        for player in self.players:
            self.ball.check_collision_with_player(player)

        # Ensure ball doesn't get stuck between players
        self.ball.resolve_stuck_ball(self.players)

    def check_goal(self):
        """
        Check if a goal has been scored and handle scoring logic.
        
        Returns:
            list: [bool, str] - Whether a goal was scored and a message about the goal
        """
        # Goal dimensions
        goal_height = 100
        goal_top = (self.height - goal_height) // 2
        goal_bottom = goal_top + goal_height

        # Check for goal on left side (blue team's goal)
        if self.ball.x - self.ball.radius <= 20:
            if goal_top <= self.ball.y <= goal_bottom:
                if self.ball.last_touched_by == "blue":
                    # Own goal by blue team
                    print("Own Goal: Point for Red Team")
                    self.red_score += 1
                    self.reset_positions()
                    self.freeze_team("blue")
                    self.unfreeze_team("red")
                    return [True, "Blue: Own Goal"]
                else:
                    # Goal by red team
                    print("Goal for Red Team")
                    self.red_score += 1
                    self.reset_positions()
                    self.freeze_team("red")
                    self.unfreeze_team("blue")
                    return [True, "Red: Goal"]

        # Check for goal on right side (red team's goal)
        elif self.ball.x + self.ball.radius >= self.width - 20:
            if goal_top <= self.ball.y <= goal_bottom:
                if self.ball.last_touched_by == "red":
                    # Own goal by red team
                    print("Own Goal: Point for Blue Team")
                    self.blue_score += 1
                    self.reset_positions()
                    self.freeze_team("red")
                    self.unfreeze_team("blue")
                    return [True, "Red: Own Goal"]
                else:
                    # Goal by blue team
                    print("Goal for Blue Team!")
                    self.blue_score += 1
                    self.reset_positions()
                    self.freeze_team("blue")
                    self.unfreeze_team("red")
                    return [True, "Blue: Goal"]
        return [False, ""]  # No goal scored

    def reset_positions(self):
        """
        Reset ball and player positions after a goal is scored.
        """
        # Bring ball to the center
        self.ball.reset_position(self.width // 2, self.height // 2)
        self.kickoff_started = False

        # Return players to initial positions
        for player in self.players:
            player.x, player.y = player.initial_position

        # Pause briefly to show goal
        pygame.time.delay(2000)

    def draw_scores(self):
        """
        Draw the current scores on the screen.
        """
        font = pygame.font.Font(None, 36)
        red_score_text = font.render(f"Red Team: {self.red_score}", True, (255, 0, 0))
        blue_score_text = font.render(
            f"Blue Team: {self.blue_score}", True, (0, 0, 255)
        )

        # Position of scores
        self.screen.blit(blue_score_text, (50, 10))
        self.screen.blit(red_score_text, (self.width - 200, 10))

    def unfreeze_team(self, team):
        """
        Allow a team to move their players.
        
        Parameters:
            team (str): Team to unfreeze ("blue" or "red")
        """
        for player in self.players:
            if player.team == team:
                player.frozen = False

    def freeze_team(self, team):
        """
        freeze team.
        
        Parameters:
            team (str): Team to freeze ("blue" or "red")
        """
        for player in self.players:
            if player.team == team:
                player.frozen = True

    def _draw_goal_net(self):
        """
        Draw net patterns inside the goals for visuals.
        """
        goal_top = (self.height - 100) // 2
        goal_depth = 40
        net_spacing = 5
        net_color = (200, 200, 200)
        
        # Draw left goal net
        for x in range(0, goal_depth, net_spacing):
            for y in range(goal_top, goal_top + 100, net_spacing):
                pygame.draw.line(
                    self.screen, net_color, (20 - x, y), (20 - x, y + net_spacing), 1
                )
                pygame.draw.line(self.screen, net_color, (20 - x, y), (20, y), 1)
        
        # Draw right goal net
        for x in range(0, goal_depth, net_spacing):
            for y in range(goal_top, goal_top + 100, net_spacing):
                pygame.draw.line(
                    self.screen,
                    net_color,
                    (self.width - 20 + x, y),
                    (self.width - 20 + x, y + net_spacing),
                    1,
                )
                pygame.draw.line(
                    self.screen,
                    net_color,
                    (self.width - 20 + x, y),
                    (self.width - 20, y),
                    1,
                )

    def draw_timer(self):
        """
        Draw the game timer showing remaining time.
        """
        # Calculate remaining time
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000
        remaining_time = max(0, int(self.game_duration - elapsed_time))
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        timer_text = f"{minutes:02}:{seconds:02}"
        
        # show timer
        font = pygame.font.Font(None, 36)
        timer_render = font.render(timer_text, True, self.WHITE)
        timer_rect = timer_render.get_rect(center=(self.width // 2, 20))
        self.screen.blit(timer_render, timer_rect)

    def display_game_over(self):
        """
        Show game over screen with final result.
        """
        font = pygame.font.Font(None, 48)
        
        # Determine winner
        if self.red_score > self.blue_score:
            game_over_text = "Red Team Wins!"
        elif self.blue_score > self.red_score:
            game_over_text = "Blue Team Wins!"
        else:
            game_over_text = "It's a Tie!"

        # Display winner text
        text_render = font.render(game_over_text, True, self.WHITE)
        text_rect = text_render.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text_render, text_rect)

        # Update display and wait before closing
        pygame.display.update()
        pygame.time.delay(3000)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    field = SoccerField()
    field.run()