import random
from typing import Tuple

import pygame

from Nullity.NullityNetworkMini import *


class Game:
    __slots__ = ["color", "screen", "clock", "board_size", "grid_size", "board_start", "WIDTH", "HEIGHT", "max_y",
                 "max_x", "walls", "radius", "arh", "font", "apple", "snake"]

    def __init__(self, arh, radius=1):
        pygame.init()
        pygame.display.set_caption("Змейка AI")
        self.color = {"BLACK": (0, 0, 0),
                      "GRAY": (71, 71, 71),
                      "RED": (255, 0, 0),
                      "GREEN": (66, 214, 43),
                      "WHITE": (255, 255, 255)
                      }
        self.screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.board_size = (20, 20)
        self.grid_size = (20, 20)
        self.board_start = (1000, 300)
        self.WIDTH, self.HEIGHT = self.board_size[0] * self.grid_size[0], self.board_size[1] * self.grid_size[1]
        self.max_y = self.HEIGHT // self.grid_size[0]
        self.max_x = self.WIDTH // self.grid_size[1]
        self.walls = True
        self.radius = radius
        self.arh = arh
        self.font = pygame.font.SysFont("Minecraft Title Cyrillic", 52)
        self.apple = Apple(self)
        self.snake = Snake(self, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), self.radius,
                           self.arh)

    def paint_grid(self, x: int, y: int, color: tuple) -> None:
        pygame.draw.rect(self.screen, color, (
            (x + self.board_start[0] / self.board_size[0]) * self.grid_size[0],
            (y + self.board_start[1] / self.board_size[1]) * self.grid_size[1], self.grid_size[0], self.grid_size[1]))

    def draw_board(self, color: tuple):
        start_x = self.board_start[0]
        start_y = self.board_start[1]
        end_x = self.board_start[0] + self.board_size[0] * self.grid_size[0]
        end_y = self.board_start[1] + self.board_size[1] * self.grid_size[1]
        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, start_y), 2)
        pygame.draw.line(self.screen, color, (start_x, start_y), (start_x, end_y), 2)
        pygame.draw.line(self.screen, color, (start_x, end_y), (end_x, end_y), 2)
        pygame.draw.line(self.screen, color, (end_x, start_y), (end_x, end_y), 2)

    def return_board(self, snake, start, end):
        board = []
        ds = {"змея": 10, "пустота": 0, "яблоко": 20, "стена": 30}
        for y in range(start[1], end[1] + 1):
            for x in range(start[0], end[0] + 1):
                f = False
                t = False
                for i in snake.coords:
                    if (x, y) == i:
                        board.append(ds["змея"] / 50)
                        t = True
                        continue

                if t:
                    continue
                if (x, y) == (self.apple.x, self.apple.y):
                    board.append(ds["яблоко"] / 50)
                    f = True
                    break
                if f:
                    continue
                if (x, y) in list((i, 0) for i in range(self.max_x)) or (x, y) in list(
                        (i, self.max_y) for i in range(self.max_x)) or (x, y) in list(
                    (0, i) for i in range(self.max_x)) or (x, y) in list((self.max_x, i) for i in range(self.max_x)):
                    board.append(ds["стена"] / 50)
                else:
                    board.append(ds["пустота"] / 50)
        apples_r = ((self.snake.x - self.apple.x) / 50, (self.snake.y - self.apple.y) / 50)
        board.append(apples_r[0])
        board.append(apples_r[1])
        return board[:((snake.radius * 2) + 1) ** 2 + 2]

    def draw_neurons(self, neirons, start=(200, 0)):
        maxs = len(neirons[0])
        for i in neirons:
            if len(i) > maxs:
                maxs = len(i)
        step_x = (32 * 32) / len(neirons)
        step_y = 900 / maxs
        radius = 1000 / maxs / 2

        font = pygame.font.SysFont('Minecraft Title Cyrillic', 40)
        text = font.render("Hidden Layer", False, (120, 255, 120))
        self.screen.blit(text, (start[0] + (len(neirons) - 2) / 2 * step_x, start[1] + step_y * (2 + maxs)))
        for numbers, i in enumerate(neirons):
            sm = (maxs - len(i)) * step_y / 2
            my_x = start[0] + numbers * step_x
            for number, j in enumerate(i):
                my_y = start[1] + step_y * (number + 1) + sm
                color = ((120, 255, 120), (50, 180, 80))
                if not numbers:
                    color = ((100, 200, 255), (20, 120, 220))
                    font = pygame.font.SysFont('Minecraft Title Cyrillic', 40)
                    text = font.render("Input Layer", False, color[0])
                    self.screen.blit(text, (my_x - 75, start[1] + step_y * (2 + maxs)))
                elif numbers == len(neirons) - 1:
                    color = ((255, 80, 80), (200, 40, 40))
                    font = pygame.font.SysFont('Minecraft Title Cyrillic', 40)
                    text = font.render("Output Layer", False, color[0])
                    self.screen.blit(text,
                                     (my_x - 75, start[1] + step_y * (2 + maxs)))
                try:
                    for num, k in enumerate(neirons[numbers + 1]):
                        sm_ = (maxs - len(neirons[numbers + 1])) * step_y / 2
                        your_y = start[1] + step_y * (num + 1) + sm_
                        pygame.draw.line(self.screen, color[0] if j > 0 else color[1], (my_x, my_y),
                                         (my_x + step_x, your_y), 1 if j > 0 else 1)
                except IndexError:
                    color = ((255, 80, 80), (200, 40, 40))
                if numbers == len(neirons) - 1:
                    o = j
                    for k in i:
                        if o != k:
                            break
                    else:
                        self.draw_neuron(False, (my_x, my_y), round(j, 2), color, 15, 2)
                        continue

                    self.draw_neuron(j >= max(i), (my_x, my_y), round(j, 2), color, 15, 2)
                else:
                    self.draw_neuron(j > 0, (my_x, my_y), round(j, 2), color, radius, 2)
            try:
                for num, k in enumerate(neirons[numbers + 1]):
                    sm_ = (maxs - len(neirons[numbers + 1])) * step_y / 2
                    your_y = start[1] + step_y * (num + 1) + sm_
                    pygame.draw.line(self.screen, ((255, 165, 0), (255, 165, 0))[0],
                                     (my_x + 0, start[1] + step_y * (len(i) + 1) + sm), (my_x + step_x, your_y), 1)
                self.draw_neuron(False, (my_x + 0, start[1] + step_y * (len(i) + 1) + sm), round(i[1], 2),
                                 ((255, 165, 0), (255, 165, 0)),
                                 radius, 2)
            except IndexError:
                pass

    def draw_neuron(self, activity: bool, coords: tuple, value,
                    color: Tuple[Tuple[int, int, int], Tuple[int, int, int]], radius, board):
        text_color = (0, 0, 0)
        if activity:
            colors = color[0]
        else:
            colors = (0, 0, 0)
            text_color = (255, 255, 255)
        pygame.draw.circle(self.screen, colors, coords, abs(radius))
        pygame.draw.circle(self.screen, color[0], coords, abs(radius), board)
        font = pygame.font.SysFont('Minecraft Title Cyrillic', 20)
        text = font.render(f"{value}", True, text_color)

        self.screen.blit(text, (coords[0] - (len(str(value)) + 1) * 2, coords[1] - len(str(value)) * 2))

    def text_rendering(self):
        text_1 = self.font.render(f"{round(self.clock.get_fps(), 2)} FPS", True, self.snake.color)
        text_2 = self.font.render(f"Счёт: {round(self.snake.score)}",
                                  True,
                                  (255, 255, 255))
        text_3 = self.font.render(f"Длина: {self.snake.length}",
                                  True,
                                  (255, 255, 255))
        text_4 = self.font.render(f"поколение: {self.snake.epochs}", True, (255, 255, 255))
        self.screen.blit(text_1, (50, 50))
        self.screen.blit(text_2, (1100, 50))
        self.screen.blit(text_3, (1350, 50))
        self.screen.blit(text_4, (1550, 50))

    def update_ai(self):
        running = True
        while running:
            self.screen.fill(self.color["BLACK"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            state = self.return_board(self.snake, (self.snake.x - self.snake.radius, self.snake.y - self.snake.radius),
                                      (self.snake.x + self.snake.radius, self.snake.y + self.snake.radius))
            self.draw_neurons(self.snake.network.neuron(state), (100, 75))
            predicted = self.snake.network.neuron(state)[-1]  # Выходной слой
            direction = predicted.index(max(predicted))
            self.snake.set_direction(direction)
            self.snake.move()
            self.apple.collision(self.snake)
            self.snake.rendering()
            self.apple.rendering()
            self.text_rendering()
            self.draw_board(self.color["WHITE"])
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


class Snake:
    __slots__ = ["parents", "x", "y", "length", "__direction", "coords", "life", "score", "color", "directions",
                 "direction", "radius", "epochs", "network"]
    def __init__(self, parents, color, radius, arh):
        self.parents: Game = parents
        self.x = random.randint(0, self.parents.max_x)
        self.y = random.randint(0, self.parents.max_y)
        self.length = 3
        self.__direction = 0
        self.coords = [(self.x - 1 * i, self.y - 0 * i) for i in range(self.length)]
        self.life = True
        self.score = 0
        self.color = color
        self.directions = {"up": 3,
                           "down": 2,
                           "right": 0,
                           "left": 1}
        self.direction = random.choice(list(self.directions.values()))
        self.radius = radius
        self.epochs = 1
        self.network = Neuralnetwork()
        self.network.add_layer(((self.radius * 2) + 1) ** 2 + 2, "SIGMOID")
        for i in arh:
            self.network.add_layer(i, "RELU")
        self.network.add_layer(4, "SIGMOID")
        self.network.calculating_weights()

    def append(self):
        self.coords.append(self.coords[-1])

    def rendering(self):
        if self.life: [self.parents.paint_grid(x, y, (72, 71, 71, 20)) for x in
                       range(self.x - self.radius, self.x + self.radius + 1) for y in
                       range(self.y - self.radius, self.y + self.radius + 1)]
        [self.parents.paint_grid(i[0], i[1], self.color) for i in self.coords[::-1]]

    def set_direction(self, new_direction):
        if {0: 1, 1: 0, 2: 3, 3: 2}[self.__direction] != new_direction: self.__direction = new_direction

    def check(self, x, y):
        return (x, y) in self.coords

    def checking_collision(self):
        self.life = False if self.x >= self.parents.max_x or self.x < 0 or self.y >= self.parents.max_y or self.y < 0 else self.life
        return self.life

    def alive(self):
        self.x = random.randint(0, self.parents.max_x)
        self.y = random.randint(0, self.parents.max_y)
        self.length = 3
        self.__direction = random.choice([0, 1, 2, 3])
        self.coords = [(self.x - 1 * i, self.y - 0 * i) for i in range(self.length)]
        self.life = True
        self.directions = {"up": 3,
                           "down": 2,
                           "right": 0,
                           "left": 1}
        self.epochs += 1
        self.network.calculating_weights()
        self.parents.apple.generate_coords(self)


    def move(self):
        if self.life:
            if len(self.coords) == self.parents.board_size[0] ** 2:
                return
            if self.__direction == 0:
                self.x += 1
            elif self.__direction == 1:
                self.x -= 1
            elif self.__direction == 2:
                self.y += 1
            elif self.__direction == 3:
                self.y -= 1
            self.score -= 0.01
            if self.check(self.x, self.y) or not self.checking_collision():
                self.life = False
                self.score -= 10
                self.rendering()
                self.alive()
                return
            self.coords.insert(0, (self.x, self.y))
        try:
            self.coords.pop()
        except IndexError:
            pass


class Apple:
    __slots__ = ["parents", "x", "y", "count"]

    def __init__(self, parents):
        self.parents: Game = parents
        self.x = random.randint(0, self.parents.max_x - 1)
        self.y = random.randint(0, self.parents.max_y - 1)
        self.count = 0

    def rendering(self):
        self.parents.paint_grid(self.x, self.y, self.parents.color["RED"])

    def generate_coords(self, parent):
        while (self.x, self.y) in parent.coords:
            self.x = random.randint(0, self.parents.max_x - 1)
            self.y = random.randint(0, self.parents.max_y - 1)

    def collision(self, parent):
        if (parent.x, parent.y) == (self.x, self.y):
            parent.length += 1
            parent.score += 10
            parent.append()
            self.generate_coords(parent)


if __name__ == "__main__":
    game = Game([8, 7, 6, 5], 3)  # 10 змеек
    game.update_ai()
