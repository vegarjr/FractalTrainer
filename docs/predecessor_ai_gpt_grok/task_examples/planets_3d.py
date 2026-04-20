import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class SolarSystem:
    # Astronomical data for planets (semi-major axis, eccentricity, size, color)
    PLANETS_DATA = [
        {'name': 'Mercury', 'a': 0.39, 'e': 0.205, 'size': 20, 'color': 'gray'},
        {'name': 'Venus', 'a': 0.72, 'e': 0.0067, 'size': 30, 'color': 'orange'},
        {'name': 'Earth', 'a': 1.0, 'e': 0.0167, 'size': 35, 'color': 'blue'},
        {'name': 'Mars', 'a': 1.52, 'e': 0.0934, 'size': 30, 'color': 'red'},
        {'name': 'Jupiter', 'a': 5.2, 'e': 0.0489, 'size': 60, 'color': 'brown'},
        {'name': 'Saturn', 'a': 9.54, 'e': 0.0542, 'size': 55, 'color': 'gold'},
        {'name': 'Uranus', 'a': 19.2, 'e': 0.0472, 'size': 45, 'color': 'cyan'},
        {'name': 'Neptune', 'a': 30.1, 'e': 0.0086, 'size': 40, 'color': 'blue'}
    ]

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.planets = []
        self.setup_plot()

    def calculate_orbit(self, a, e, num_points=200):
        """Calculate elliptical orbit coordinates"""
        theta = np.linspace(0, 2 * np.pi, num_points)
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sin(theta / 2) * 0.1  # Small z-component for 3D effect
        return x, y, z

    def create_celestial_body(self, x, y, z, size, color):
        """Create a celestial body (sun or planet)"""
        return self.ax.scatter(x, y, z, s=size, c=color, alpha=0.8)

    def add_starry_background(self, num_stars=200):
        """Add background stars"""
        star_positions = np.random.rand(num_stars, 3) * 70 - 35
        star_sizes = np.random.randint(1, 5, num_stars)
        self.ax.scatter(star_positions[:, 0], 
                       star_positions[:, 1], 
                       star_positions[:, 2], 
                       s=star_sizes, color='white', alpha=0.5)

    def add_sun_glow(self):
        """Add glowing effect around the sun"""
        glow = Circle((0, 0), 0.5, color='yellow', alpha=0.1)
        self.ax.add_patch(glow)

    def setup_plot(self):
        """Initialize the plot settings"""
        # Plot styling
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        
        # Axis labels and limits
        self.ax.set_xlim(-35, 35)