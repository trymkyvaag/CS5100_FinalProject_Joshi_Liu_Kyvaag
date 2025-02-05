import pygame
import math


class Player:
    def __init__(self, x, y, color, team):
        self.x = x
        self.y = y
        self.color = color
        self.team = team
        self.angle = 0
        self.radius = 20
        self.speed = 3

    def draw(self, screen, teamIdx):
        pygame.draw.circle(screen, self.color,
                           (int(self.x), int(self.y)), self.radius)

        if teamIdx < 2:
            self.angle = 0
        else:
            self.angle = 180

        indicator_x = self.x + self.radius * math.cos(math.radians(self.angle))
        indicator_y = self.y - self.radius * math.sin(math.radians(self.angle))
        pygame.draw.circle(screen, (255, 255, 255),
                           (int(indicator_x), int(indicator_y)), 3)

    def move(self, dx, dy, field_width, field_height, players):
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed

        if (new_x - self.radius < 20 or
            new_x + self.radius > field_width - 20 or
            new_y - self.radius < 0 or
                new_y + self.radius > field_height):
            return

        for player in players:
            if player != self:
                distance = math.sqrt((new_x - player.x) **
                                     2 + (new_y - player.y)**2)
                if distance < self.radius * 2:
                    return

        self.x = new_x
        self.y = new_y

    def turn(self, angle):
        self.angle = (self.angle + angle) % 360
