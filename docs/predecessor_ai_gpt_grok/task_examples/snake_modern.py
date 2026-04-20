import pygame
import random
import sys
from collections import deque

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
CELL_SIZE = 20
INITIAL_SPEED = 10
SPEED_INCREMENT = 0.5

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Directions
DIRECTIONS = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0)
}

# Initial positions
INITIAL_SNAKE_POSITION = (10, 10)

class Snake:
    def __init__(self):
        self.body = deque([INITIAL_SNAKE_POSITION])
        self.direction = DIRECTIONS[pygame.K_RIGHT]
        self.body_set = set(self.body)

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.appendleft(new_head)
        self.body_set.add(new_head)
        tail = self.body.pop()
        self.body_set.remove(tail)

    def grow(self):
        tail = self.body[-1]
        self.body.append(tail)
        self.body_set.add(tail)

    def check_collision(self):
        head = self.body[0]
        return (head in list(self.body)[1:] or 
                not (0 <= head[0] < SCREEN_WIDTH // CELL_SIZE and 0 <= head[1] < SCREEN_HEIGHT // CELL_SIZE))

class Food:
    def __init__(self, snake_body_set):
        self.position = self.spawn(snake_body_set)

    def spawn(self, snake_body_set):
        available_positions = [
            (x, y) for x in range(SCREEN_WIDTH // CELL_SIZE)
            for y in range(SCREEN_HEIGHT // CELL_SIZE)
            if (x, y) not in snake_body_set
        ]
        return random.choice(available_positions)

def draw(screen, snake, food, score):
    screen.fill(BLACK)
    for segment in snake.body:
        pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (food.position[0] * CELL_SIZE, food.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw the score
    font = pygame.font.SysFont(None, 36)
    text = font.render(f'Score: {score}', True, WHITE)
    screen.blit(text, (10, 10))

    pygame.display.flip()

def handle_events(snake):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                return 'pause'
            new_direction = DIRECTIONS.get(event.key)
            if new_direction and (new_direction[0] != -snake.direction[0] or new_direction[1] != -snake.direction[1]):
                snake.direction = new_direction
    return True

def pause_game():
    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = False

def game_over_screen(screen):
    screen.fill(BLACK)
    font = pygame.font.SysFont(None, 72)
    text = font.render('Game Over', True, WHITE)
    screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(2000)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Modern Snake Game')
    clock = pygame.time.Clock()

    snake = Snake()
    food = Food(snake.body_set)
    score = 0
    speed = INITIAL_SPEED

    running = True
    while running:
        result = handle_events(snake)
        if result == 'pause':
            pause_game()
        elif result == False:
            running = False

        snake.move()

        if snake.body[0] == food.position:
            snake.grow()
            food.position = food.spawn(snake.body_set)
            score += 1
            speed += SPEED_INCREMENT

        if snake.check_collision():
            running = False

        draw(screen, snake, food, score)
        clock.tick(speed)

    game_over_screen(screen)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()