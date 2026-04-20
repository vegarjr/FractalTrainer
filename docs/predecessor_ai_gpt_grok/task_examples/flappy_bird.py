import pygame
import sys
import random
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class GameConfig:
    SCREEN_WIDTH: int = 288
    SCREEN_HEIGHT: int = 512
    FPS: int = 60
    GRAVITY: float = 0.25
    BIRD_JUMP: int = -9
    PIPE_SPEED: int = 3
    PIPE_GAP: int = 150
    SCROLL_SPEED: int = 2
    ANIMATION_SPEED: int = 5
    PIPE_SPAWN_DISTANCE: int = 200

class FlappyBird:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        
        self.config = GameConfig()
        self.setup_display()
        self.load_assets()
        self.reset_game()

    def setup_display(self):
        """Initialize display and clock"""
        self.screen = pygame.display.set_mode((self.config.SCREEN_WIDTH, 
                                             self.config.SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()

    def load_assets(self):
        """Load and prepare game assets with error handling"""
        try:
            # Load and convert images for better performance
            self.assets = {
                'background': self.load_image('assets/background.png'),
                'ground': self.load_image('assets/ground.png'),
                'pipe': self.load_image('assets/pipe.png', True),
                'bird_frames': [
                    self.load_image(f'assets/bird{i}.png', True)
                    for i in range(1, 4)
                ]
            }

            # Load sound effects
            self.sounds = {
                'flap': self.load_sound('assets/flap.wav'),
                'score': self.load_sound('assets/score.wav'),
                'hit': self.load_sound('assets/hit.wav')
            }

            self.font = pygame.font.Font(None, 36)

        except (pygame.error, FileNotFoundError) as e:
            self.handle_asset_error(e)

    @staticmethod
    def load_image(path: str, alpha: bool = False) -> pygame.Surface:
        """Load and convert a single image"""
        img = pygame.image.load(path)
        return img.convert_alpha() if alpha else img.convert()

    @staticmethod
    def load_sound(path: str) -> pygame.mixer.Sound:
        """Load a single sound effect"""
        return pygame.mixer.Sound(path)

    def handle_asset_error(self, error: Exception):
        """Handle missing or corrupt asset files"""
        print(f"Fatal Error: Could not load game assets: {error}")
        pygame.quit()
        sys.exit(1)

    def reset_game(self):
        """Initialize or reset game state"""
        self.bird = self.assets['bird_frames'][0].get_rect(
            center=(50, self.config.SCREEN_HEIGHT // 2))
        self.bird_velocity = 0
        self.score = 0
        self.high_score = self.load_high_score()
        self.pipes: List[Tuple[pygame.Rect, pygame.Rect, bool]] = []
        self.ground_x = 0
        self.bird_frame = 0
        self.animation_time = 0
        self.game_state = "start"

    def handle_input(self):
        """Process user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.handle_space_press()