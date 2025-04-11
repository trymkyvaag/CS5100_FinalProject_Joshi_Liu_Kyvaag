import math
import pygame
class Player:
    def __init__(self, x, y, color, team):
        """
        Initialize a new player with position, appearance, and movement properties.
        
        Parameters:
            x (float): Initial x-coordinate on the field
            y (float): Initial y-coordinate on the field
            color (tuple): RGB color tuple for the player
            team (str): Identifier for the player's team
        """
        self.x = x
        self.y = y
        self.color = color
        self.team = team
        self.angle = 0  # Direction the player is facing in degrees
        self.radius = 20  # Size of the player
        self.speed = 7  # Movement speed multiplier
        self.initial_position = (x, y)  # Store starting position for resets
        self.frozen = False  # Flag to prevent movement when True
    
    def draw(self, screen):
        """
        Render the player on the game screen as a circle with a direction indicator.
        
        Parameters:
            screen: Pygame display surface to draw on
        """
        # Draw the main player body as a circle
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Calculate position for the direction indicator based on player's angle
        indicator_x = self.x + self.radius * math.cos(math.radians(self.angle))
        indicator_y = self.y - self.radius * math.sin(math.radians(self.angle))
        
        # Draw a small white circle as direction indicator
        pygame.draw.circle(
            screen, (255, 255, 255), (int(indicator_x), int(indicator_y)), 3
        )
        pygame.draw.circle(
            screen, (255, 255, 255), (int(indicator_x), int(indicator_y)), 3
        )
    
    def move(self, dx, dy, field_width, field_height, players):
        """
        Move the player  while handling collision detection.
        
        Parameters:
            dx (float): Horizontal movement direction (-1, 0, or 1)
            dy (float): Vertical movement direction (-1, 0, or 1)
            field_width (int): Width of the playing field for boundary checking
            field_height (int): Height of the playing field for boundary checking
            players (list): List of all players for collision detection
        """
        # Don't move if player is frozen
        if self.frozen:
            return
            
        # Calculate new position based on direction and speed
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        
        # Check for field boundary collisions with a 10-pixel buffer on sides
        # This prevents the player from getting stuck against field edges
        if (
            new_x - self.radius < 10
            or new_x + self.radius > field_width - 10
            or new_y - self.radius < 0
            or new_y + self.radius > field_height
        ):
            return
            
        # Check for collisions with other players
        for player in players:
            if player != self:
                distance = math.sqrt((new_x - player.x) ** 2 + (new_y - player.y) ** 2)
                if distance < self.radius * 2:
                    return  # Cancel movement if collision would occur
        
        # Apply the movement if no collisions detected
        self.x = new_x
        self.y = new_y
        
        # Update player's facing direction based on movement direction
        if dx != 0 or dy != 0:
            self.angle = math.degrees(math.atan2(-dy, dx)) % 360

    
    def prevent_overlap(self, other_player):
        """
        Resolve collision by pushing both players apart when overlap is detected.
        
        Called when two players are found to be overlapping to ensure they don't
        remain stuck together.
        
        Parameters:
            other_player: Another Player object that is overlapping with this one
        """
        
        # Calculate distance between players
        distance = math.sqrt(
            (self.x - other_player.x) ** 2 + (self.y - other_player.y) ** 2
        )
        
        # If players are overlapping
        if distance < self.radius + other_player.radius:
            # Calculate how much they overlap
            overlap = self.radius + other_player.radius - distance
            
            # Calculate normalized direction vec betwween players
            dx = (self.x - other_player.x) / distance
            dy = (self.y - other_player.y) / distance
            
            # Push both players apart equally along the direction vec
            self.x += dx * overlap / 2
            self.y += dy * overlap / 2
            other_player.x -= dx * overlap / 2
            other_player.y -= dy * overlap / 2