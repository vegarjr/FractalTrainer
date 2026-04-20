import turtle
import random
from typing import Tuple

class HouseDrawer:
    def __init__(self):
        # Screen setup
        self.screen = turtle.Screen()
        self.screen.title("Perfect House Drawing")
        self.screen.setup(800, 600)
        self.screen.bgcolor("skyblue")
        
        # Initialize turtle
        self.t = turtle.Turtle()
        self.t.speed(0)  # Fastest speed
        self.t.hideturtle()

    def draw_shape(self, shape: str, size: int, color: str, position: Tuple[int, int]) -> None:
        """Generic shape drawing function"""
        self.t.penup()
        self.t.goto(*position)
        self.t.pendown()
        self.t.color(color)
        self.t.begin_fill()
        
        if shape == "square":
            for _ in range(4):
                self.t.forward(size)
                self.t.left(90)
        elif shape == "triangle":
            for _ in range(3):
                self.t.forward(size)
                self.t.left(120)
        elif shape == "circle":
            self.t.circle(size)
            
        self.t.end_fill()

    def draw_flower(self, position: Tuple[int, int], size: int, color: str) -> None:
        """Draw a flower with six petals"""
        self.t.penup()
        self.t.goto(*position)
        self.t.pendown()
        self.t.color(color)
        
        for _ in range(6):
            self.t.begin_fill()
            self.t.circle(size)
            self.t.end_fill()
            self.t.left(60)

    def draw_tree(self, position: Tuple[int, int], trunk_height: int, foliage_size: int) -> None:
        """Draw a tree with trunk and foliage"""
        # Draw trunk
        self.draw_shape("square", trunk_height, "brown", position)
        
        # Draw foliage
        foliage_position = (position[0], position[1] + trunk_height)
        self.draw_shape("circle", foliage_size, "green", foliage_position)

    def draw_house(self):
        """Main function to draw the complete house scene"""
        # House base
        self.draw_shape("square", 300, "lightyellow", (-150, -100))
        
        # Roof
        self.draw_shape("triangle", 300, "brown", (-150, 200))
        
        # Door
        self.draw_shape("square", 100, "darkred", (-50, -100))
        self.draw_shape("circle", 5, "black", (20, -50))  # Doorknob
        
        # Windows
        for x in [-130, 70]:
            self.draw_shape("square", 60, "lightblue", (x, 0))
        
        # Chimney
        self.draw_shape("square", 40, "gray", (100, 200))
        self.draw_shape("square", 20, "gray", (120, 240))
        
        # Path
        self.draw_shape("square", 100, "gray", (-50, -200))
        
        # Trees
        for x in [-250, 200]:
            self.draw_tree((x, -100), 80, 50)
        
        # Random flowers
        for _ in range(10):
            x = random.randint(-300, 300)
            y = random.randint(-200, -100)
            color = random.choice(["pink", "purple", "red", "orange"])
            self.draw_flower((x, y), 10, color)

    def run(self):
        """Start the drawing"""
        self.draw_house()
        self.screen.main