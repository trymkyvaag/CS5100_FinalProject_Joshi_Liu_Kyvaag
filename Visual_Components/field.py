import pygame
import sys
from player import Player


class SoccerField:
    def __init__(self, width=600, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Soccer Field")

        self.GREEN = (34, 139, 34)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)

        self.players = [
            Player(150, 200, self.BLUE, 'blue'),
            Player(250, 150, self.BLUE, 'blue'),
            Player(450, 250, self.RED, 'red'),
            Player(550, 200, self.RED, 'red')
        ]

        self.clock = pygame.time.Clock()

    def draw_field(self):
        self.screen.fill(self.GREEN)

        pygame.draw.rect(self.screen, self.BLACK,
                         (0, 0, self.width, self.height), 10)

        pygame.draw.line(self.screen, self.WHITE,
                         (self.width // 2, 0),
                         (self.width // 2, self.height), 2)

        pygame.draw.circle(self.screen, self.WHITE,
                           (self.width // 2, self.height // 2), 50, 2)

        goal_height = 100
        goal_top = (self.height - goal_height) // 2

        pygame.draw.rect(self.screen, self.BLACK,
                         (0, 0, 20, goal_top))
        pygame.draw.rect(self.screen, self.BLACK,
                         (0, goal_top + goal_height, 20, goal_top))

        pygame.draw.rect(self.screen, self.BLACK,
                         (self.width - 20, 0, 20, goal_top))
        pygame.draw.rect(self.screen, self.BLACK,
                         (self.width - 20, goal_top + goal_height, 20, goal_top))
        
        self._draw_goal_net()
        for idx, player in enumerate(self.players):
            player.draw(self.screen, idx)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            keys = pygame.key.get_pressed()
            # THis is for debugging only ( can only dfo one player for now just )
            # to test controls/collisons
            # Blue team controls
            if keys[pygame.K_w]:
                self.players[0].move(
                    0, -1, self.width, self.height, self.players)
            if keys[pygame.K_s]:
                self.players[0].move(
                    0, 1, self.width, self.height, self.players)
            if keys[pygame.K_a]:
                self.players[0].move(-1, 0, self.width,
                                     self.height, self.players)
            if keys[pygame.K_d]:
                self.players[0].move(
                    1, 0, self.width, self.height, self.players)
            if keys[pygame.K_q]:
                self.players[0].turn(5)
            if keys[pygame.K_e]:
                self.players[0].turn(-5)

            # Red team controls
            if keys[pygame.K_UP]:
                self.players[2].move(
                    0, -1, self.width, self.height, self.players)
            if keys[pygame.K_DOWN]:
                self.players[2].move(
                    0, 1, self.width, self.height, self.players)
            if keys[pygame.K_LEFT]:
                self.players[2].move(-1, 0, self.width,
                                     self.height, self.players)
            if keys[pygame.K_RIGHT]:
                self.players[2].move(
                    1, 0, self.width, self.height, self.players)
            if keys[pygame.K_m]:
                self.players[2].turn(5)
            if keys[pygame.K_n]:
                self.players[2].turn(-5)

            self.draw_field()
            pygame.display.flip()
            self.clock.tick(60)

    def _draw_goal_net(self):
        goal_top = (self.height - 100) // 2
        goal_depth = 40
        net_spacing = 5
        net_color = (200, 200, 200)
        for x in range(0, goal_depth, net_spacing):
            for y in range(goal_top, goal_top + 100, net_spacing):
                pygame.draw.line(self.screen, net_color, (20 - x, y), (20 - x, y + net_spacing), 1)
                pygame.draw.line(self.screen, net_color, (20 - x, y), (20, y), 1)
        for x in range(0, goal_depth, net_spacing):
            for y in range(goal_top, goal_top + 100, net_spacing):
                pygame.draw.line(self.screen, net_color, (self.width - 20 + x, y), (self.width - 20 + x, y + net_spacing), 1)
                pygame.draw.line(self.screen, net_color, (self.width - 20 + x, y), (self.width - 20, y), 1)


if __name__ == "__main__":
    field = SoccerField()
    field.run()
