import math

import pygame


class Player:
    def __init__(self, x, y, color, team):
        self.x = x
        self.y = y
        self.color = color
        self.team = team
        self.angle = 0
        self.radius = 20
        self.speed = 7
        self.initial_position = (x, y)
        self.frozen = False
        self.disabled = False

    def draw(self, screen):
        # Removed direction of looking, just look in the direction of the movement (WASD for now)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

        indicator_x = self.x + self.radius * math.cos(math.radians(self.angle))
        indicator_y = self.y - self.radius * math.sin(math.radians(self.angle))
        pygame.draw.circle(
            screen, (255, 255, 255), (int(indicator_x), int(indicator_y)), 3
        )

        pygame.draw.circle(
            screen, (255, 255, 255), (int(indicator_x), int(indicator_y)), 3
        )

    def move(self, dx, dy, field_width, field_height, players):
        if self.frozen:
            return

        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed

        # Changed this a bit because the ball would sometimes exactly sit on the
        # right and left wall and the player is unable to touch it.
        if (
            new_x - self.radius < 10
            or new_x + self.radius > field_width - 10
            or new_y - self.radius < 0
            or new_y + self.radius > field_height
        ):
            return

        for player in players:
            if player != self:
                distance = math.sqrt((new_x - player.x) ** 2 + (new_y - player.y) ** 2)
                if distance < self.radius * 2:
                    return

        self.x = new_x
        self.y = new_y

        # Look in the direction of movement
        if dx != 0 or dy != 0:
            self.angle = math.degrees(math.atan2(-dy, dx)) % 360

    # def turn(self, angle):
    #     self.angle = (self.angle + angle) % 360

    def prevent_overlap(self, other_player):
        import math

        distance = math.sqrt(
            (self.x - other_player.x) ** 2 + (self.y - other_player.y) ** 2
        )
        if distance < self.radius + other_player.radius:
            # Push this player back based on their velocity
            overlap = self.radius + other_player.radius - distance
            dx = (self.x - other_player.x) / distance
            dy = (self.y - other_player.y) / distance
            self.x += dx * overlap / 2
            self.y += dy * overlap / 2
            other_player.x -= dx * overlap / 2
            other_player.y -= dy * overlap / 2
