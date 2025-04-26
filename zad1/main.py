import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_flat_surface_cloud(width, length, num_points):
    x = np.random.uniform(-width / 2, width / 2, num_points)
    y = np.random.uniform(-length / 2 + 20, length / 2 + 20, num_points)
    z = np.zeros(num_points)  # Płaszczyzna pozioma
    return x, y, z

def generate_vertical_surface_cloud(width, height, num_points):
    y = np.random.uniform(-width / 2, width / 2, num_points)
    x = np.zeros(num_points)  # Płaszczyzna pionowa (X = 0)
    z = np.random.uniform(-height / 2, height / 2, num_points)
    return x, y, z

def generate_cylindrical_surface_cloud(radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-height / 2, height / 2, num_points)
    x = radius * np.cos(theta) 
    y = radius * np.sin(theta) + 10
    return x, y, z

def save_to_xyz_file(x, y, z, filename):
    with open(filename, 'w') as f:
        for xi, yi, zi in zip(x, y, z):
            f.write(f"{xi} {yi} {zi}\n")

# Parametry powierzchni
width = 8
length = 8
height = 8
radius = 4
num_points = 5000

# Generowanie punktów
x, y, z = generate_flat_surface_cloud(width, length, num_points)
save_to_xyz_file(x, y, z, "zad1/flat_surface.xyz")

x, y, z = generate_vertical_surface_cloud(width, height, num_points)
save_to_xyz_file(x, y, z, "zad1/vertical_surface.xyz")

x, y, z = generate_cylindrical_surface_cloud(radius, height, num_points)
save_to_xyz_file(x, y, z, "zad1/cylindrical_surface.xyz")