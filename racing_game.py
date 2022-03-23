import pygame
import sys
from enum import Enum
import os
import numpy as np

pygame.init()

size = width, height = 800, 600

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class RacingGame:
    def __init__(self, w=800, h=600) -> None:
        self.w = w
        self.h = h
        # Intialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Racing Game")
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self) -> None:
        pass

    def play_step(self):
        game_over = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
                pygame.quit()
                sys.exit()

        self.display.fill((0, 100, 100))
        pygame.display.flip()

        return game_over
                
                

game = RacingGame()
while True:
    game_over = game.play_step()
    if game_over:
        break

    