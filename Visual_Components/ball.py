import pygame

class Ball:
    """
    The idea is to have a functioning ball that moves with its own player ball collision logic.
    More like air hockey kind of scenario.
    """
    def __init__(self, x, y, radius=10, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.velocity = [0, 0]
        self.friction = 0.98
        self.min_velocity = 0.1
        self.last_touched_by = None

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def move(self):
        # Update coordinates
        self.x += self.velocity[0]
        self.y += self.velocity[1]

        # Apply friction to slow the ball down gradually
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction

        # Stop the ball if its velocity is very small
        if abs(self.velocity[0]) < self.min_velocity:
            self.velocity[0] = 0
        if abs(self.velocity[1]) < self.min_velocity:
            self.velocity[1] = 0

    def check_collision_with_walls(self, field_width, field_height):
        # Allow for ball to hit the goals (left goal)
        if self.x - self.radius < 0:
            if not (field_height // 2 - 50 <= self.y <= field_height // 2 + 50):
                self.x = self.radius
                self.velocity[0] *= -1

        # Allow for ball to hit the goals (right goal)
        if self.x + self.radius > field_width:
            if not (field_height // 2 - 50 <= self.y <= field_height // 2 + 50):
                self.x = field_width - self.radius
                self.velocity[0] *= -1

        # For top and bottom just change the direction and bounce (top)
        if self.y - self.radius < 0:
            self.y = self.radius
            self.velocity[1] *= -1

        # For top and bottom just change the direction and bounce (bottom)
        if self.y + self.radius > field_height:
            self.y = field_height - self.radius
            self.velocity[1] *= -1

        # Due to a min velocity to track of we stop when we hit a min velocity
        if abs(self.velocity[0]) < 0.1 and abs(self.velocity[1]) < 0.1:
            if not (self.velocity[0] == 0 and self.velocity[1] == 0):
                import random
                self.velocity[0] += random.uniform(-1, 1) * 0.5
                self.velocity[1] += random.uniform(-1, 1) * 0.5

    def reset_position(self, x, y):
        # Reset ball position (after goals)
        self.x = x
        self.y = y
        self.velocity = [0, 0]

    def check_collision_with_player(self, player):
        # We are bouncy after all
        import math
        distance = math.sqrt((self.x - player.x) ** 2 + (self.y - player.y) ** 2)
        if distance < self.radius + player.radius:
            # Collided where do I bounce
            dx = self.x - player.x
            dy = self.y - player.y
            magnitude = math.sqrt(dx ** 2 + dy ** 2)
            if magnitude != 0:
                dx /= magnitude
                dy /= magnitude

            # Add a bit of force for the bounce, no perfectly elastic bodies but RIP physics
            self.velocity[0] = dx * 5
            self.velocity[1] = dy * 5

            # State tracker for who to start the next round
            self.last_touched_by = player.team

    def resolve_stuck_ball(self, players):
        for player in players:
            import math
            # Ball sometimes can "lay" on a player, don't do that just bounce.
            distance = math.sqrt((self.x - player.x) ** 2 + (self.y - player.y) ** 2)
            if distance < self.radius + player.radius:  # Ball is stuck inside a player
                # Push the ball out of the player's radius
                overlap = self.radius + player.radius - distance
                dx = (self.x - player.x) / distance
                dy = (self.y - player.y) / distance
                self.x += dx * overlap
                self.y += dy * overlap

                # Prevent infinite collisions
                import random
                self.velocity[0] += random.uniform(-1, 1)
                self.velocity[1] += random.uniform(-1, 1)
