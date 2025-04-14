import pygame
import random

class Ball:
    """
    A class representing a ball.
    The ball moves with physics-like properties including velocity, friction,
    and collision logic with walls and players.
    """

    def __init__(self, x, y, radius=10, color=(255, 255, 255)):
        """
        Initialize a new ball with position, appearance, and physics properties.
        
        Parameters:
            x (float): Initial x-coordinate on the field
            y (float): Initial y-coordinate on the field
            radius (int): Size of the ball (default: 10)
            color (tuple): RGB color tuple for the ball (default: white)
        """
        self.x = x  # x-coordinate position
        self.y = y  # y-coordinate position
        self.radius = radius  # Size of the ball
        self.color = color  # Color of the ball (RGB tuple)
        self.velocity = [0, 0]  # Current movement vector [x_velocity, y_velocity]
        self.friction = 0.98  # Friction coefficient to slow the ball over time
        self.min_velocity = 0.1  # Minimum velocity threshold before stopping
        self.last_touched_by = None  # Tracks which team last touched the ball

    def draw(self, screen):
        """
        Render the ball.
        
        Parameters:
            screen: Pygame display
        """
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def move(self):
        """
        Update the ball's position based on its current velocity and apply friction.
        Called each frame to animate the ball's movement.
        """
        # Update coordinates according to velocity
        self.x += self.velocity[0]
        self.y += self.velocity[1]

        # Apply friction to slow the ball down gradually
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction

        # Stop the ball if its velocity is below the minimum threshold
        if abs(self.velocity[0]) < self.min_velocity:
            self.velocity[0] = 0
        if abs(self.velocity[1]) < self.min_velocity:
            self.velocity[1] = 0

    def check_collision_with_walls(self, field_width, field_height):
        """
        Check and handle collisions between the ball and field edges.
        Includes the goals.
        
        Parameters:
            field_width (int): Width of the playing field
            field_height (int): Height of the playing field
        """
        # Left wall collision with goal exception
        # If ball hits left wall outside of goal area, bounce it
        if self.x - self.radius < 0:
            if not (field_height // 2 - 50 <= self.y <= field_height // 2 + 50):
                self.x = self.radius  # Prevent ball from going through wall
                self.velocity[0] *= -1  # Reverse horizontal velocity (bounce)

        # Right wall collision with goal exception
        # If ball hits right wall outside of goal area, bounce it
        if self.x + self.radius > field_width:
            if not (field_height // 2 - 50 <= self.y <= field_height // 2 + 50):
                self.x = field_width - self.radius  
                self.velocity[0] *= -1 

        # Top wall collision
        if self.y - self.radius < 0:
            self.y = self.radius  

            self.velocity[1] *= -1 

        # Bottom wall collision
        if self.y + self.radius > field_height:
            self.y = field_height - self.radius 
            self.velocity[1] *= -1 

        # Add small random movement if ball is moving very slowly but not stopped
        if abs(self.velocity[0]) < 0.1 and abs(self.velocity[1]) < 0.1:
            if not (self.velocity[0] == 0 and self.velocity[1] == 0):
                import random
                # Add small random velocity to unstick the ball
                self.velocity[0] += random.uniform(-1, 1) * 0.5
                self.velocity[1] += random.uniform(-1, 1) * 0.5

    def reset_position(self, x, y):
        """
        Reset the ball to a specific position with zero velocity.
        Typically used after a goal is scored or to restart play.
        
        Parameters:
            x (float): New x-coordinate for the ball
            y (float): New y-coordinate for the ball
        """
        self.x = x
        self.y = y
        self.velocity = [0, 0]  # Stop the ball's movement

    def check_collision_with_player(self, player):
        """
        Check and handle collisions between the ball and a player.
        Calculates bounce direction and applies velocity when collisions occur.
        
        Parameters:
            player: Player object to check collision with
        """
        import math

        # Calculate distance between ball center and player center
        distance = math.sqrt((self.x - player.x) ** 2 + (self.y - player.y) ** 2)
        
        # If distance is less than sum of radii, collision has occurred
        if distance < self.radius + player.radius:
            # Calculate normalized direction vector from player to ball
            dx = self.x - player.x
            dy = self.y - player.y
            magnitude = math.sqrt(dx**2 + dy**2)
            
            # Prevent division by zero
            if magnitude != 0:
                dx /= magnitude
                dy /= magnitude

            # Apply new velocity in the direction away from player
            # Force of 5 units gives a consistent bounce speed regardless of incoming velocity
            self.velocity[0] = dx * 5
            self.velocity[1] = dy * 5

            # Track which team last touched the ball (for goal attribution)
            self.last_touched_by = player.team

    def resolve_stuck_ball(self, players):
        """
        Check if the ball is stuck inside any player and resolve the overlap.
        Adds small random velocity to prevent the ball from getting stuck again.
        
        Parameters:
            players (list): List of all player objects to check against
        """
        for player in players:
            import math

            # Calculate distance between ball and player
            distance = math.sqrt((self.x - player.x) ** 2 + (self.y - player.y) ** 2)
            
            # If ball is overlapping with player
            if distance < self.radius + player.radius:
                # Calculate overlap amount
                overlap = self.radius + player.radius - distance
                
                # Calculate direction vector to push ball out
                dx = (self.x - player.x) / distance
                dy = (self.y - player.y) / distance
                
                # Move ball out of player by the overlap amount
                self.x += dx * overlap
                self.y += dy * overlap

                # Add small random velocity to prevent immediate re-collision
                self.velocity[0] += random.uniform(-1, 1)
                self.velocity[1] += random.uniform(-1, 1)