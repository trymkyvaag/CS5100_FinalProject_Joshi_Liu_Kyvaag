import pygame
import sys

from Visual_Components.ball import Ball
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
            # Player(250, 150, self.BLUE, 'blue'),
            Player(450, 250, self.RED, 'red'),
            # Player(550, 200, self.RED, 'red')
        ]

        self.clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)
        self.red_score = 0
        self.blue_score = 0
        self.kickoff_started = False

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
            player.draw(self.screen)

        self.ball.draw(self.screen)
        self.draw_scores()

    def reset_game(self):
        self.ball.reset_position(self.width // 2, self.height // 2)

    def run(self):
        self.reset_game()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            keys = pygame.key.get_pressed()
            # THis is for debugging only ( can only dfo one player for now just )
            # to test controls/collisons
            # Blue team controls
            if not self.players[0].frozen:
                if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d]:
                    if not self.kickoff_started:  # Detect kickoff start
                        print("Kickoff started by Blue Team!")
                        self.unfreeze_team("red")  # Unfreeze red team
                        self.kickoff_started = True
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
            # if keys[pygame.K_q]:
            #     self.players[0].turn(5)
            # if keys[pygame.K_e]:
            #     self.players[0].turn(-5)

            # if keys[pygame.K_SPACE]:
            #     self.players[0].kick_ball(self.ball)

            # Red team controls
            if not self.players[1].frozen:
                if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                    if not self.kickoff_started:  # Detect kickoff start
                        print("Kickoff started by Red Team!")
                        self.unfreeze_team("blue")  # Unfreeze blue team
                        self.kickoff_started = True
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
            # if keys[pygame.K_m]:
            #     self.players[2].turn(5)
            # if keys[pygame.K_n]:
            #     self.players[2].turn(-5)

            for i, player in enumerate(self.players):
                for j, other_player in enumerate(self.players):
                    if i != j:
                        player.prevent_overlap(other_player)

            self.ball.move()
            self.ball.check_collision_with_walls(self.width, self.height)

            for player in self.players:
                self.ball.check_collision_with_player(player)

            self.ball.resolve_stuck_ball(self.players)

            if self.check_goal():
                continue

            self.draw_field()
            pygame.display.flip()
            self.clock.tick(60)

    def check_goal(self):
        # Wanna make this global
        goal_depth = 40
        goal_height = 100
        goal_top = (self.height - goal_height) // 2
        goal_bottom = goal_top + goal_height

        if self.ball.x - self.ball.radius <= 20 - goal_depth:
            if goal_top <= self.ball.y <= goal_bottom:
                if self.ball.last_touched_by == "blue":
                    print("Own Goal: Point for Red Team")
                    self.red_score += 1
                    self.reset_positions("red")
                else:
                    print("Goal for Red Team")
                    self.red_score += 1
                    self.reset_positions("blue")
                return True

        elif self.ball.x + self.ball.radius >= self.width - (20 - goal_depth):
            if goal_top <= self.ball.y <= goal_bottom:
                if self.ball.last_touched_by == "red":
                    print("Own Goal: Point for Blue Team")
                    self.blue_score += 1
                    self.reset_positions("red")
                else:
                    print("Goal for Blue Team!")
                    self.blue_score += 1
                    self.reset_positions("blue")

                return True

        return False

    def reset_positions(self, last_scoring_team):
        # Bring ball to the center
        self.ball.reset_position(self.width // 2, self.height // 2)

        # Reset the players
        # The team that scored does not start again
        for player in self.players:
            player.x, player.y = player.initial_position
            if player.team == last_scoring_team:
                player.frozen = True
            else:
                player.frozen = False

        self.kickoff_started = False

        pygame.time.delay(2000)

    def draw_scores(self):
        font = pygame.font.Font(None, 36)
        red_score_text = font.render(f"Red Team: {self.red_score}", True, (255, 0, 0))
        blue_score_text = font.render(f"Blue Team: {self.blue_score}", True, (0, 0, 255))

        self.screen.blit(blue_score_text, (50, 10))
        self.screen.blit(red_score_text, (self.width - 200, 10))

    def unfreeze_team(self, team):
        for player in self.players:
            if player.team == team:
                player.frozen = False

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
