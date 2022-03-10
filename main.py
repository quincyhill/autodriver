from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT
)

import math
import random
import sys
import os
import pygame
import sqlite3

pygame.init()

# Sets screen dimensions
screen = pygame.display.set_mode((800, 600))

# Run until user quits
running = True
while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == KEYDOWN:
            if event.key == K_DOWN:
                print("down")
    
    # Fill the screen with grey
    screen.fill((200, 200, 200))
    
    # Draw a circle in the middle of the screen
    pygame.draw.circle(screen, (0, 0, 0), (400, 300), 100)
    
    # Flip the display
    pygame.display.flip()
    
# Done! Quit the game
pygame.quit()