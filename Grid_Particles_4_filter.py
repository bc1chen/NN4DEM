#-- Import general libraries
import os
import numpy as np 
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

# SUBROUTINE AI4DEM – PARTICLES INTERACTION SIMULATION

# Functions ********************************************************************
# Function to generate a structured grid
def create_grid(l, d):
    num_cells = int(l / (d))
    grid = np.zeros((num_cells, num_cells), dtype=int)
    return grid

# Function to generate random non-overlapping circular particles
def generate_particles(grid, l, d):
    particle_radius = d
    num_particles = int((len(grid) * len(grid[0])) * 0.2)

    particles = []
    while len(particles) < num_particles:
        x = np.random.uniform(0, l)
        y = np.random.uniform(0, l)

        # Check for overlap with existing particles
        overlap = False
        for particle in particles:
            distance = np.sqrt((x - particle[0])**2 + (y - particle[1])**2)
            if distance < 2*d:
                overlap = True
                break

        if not overlap:
            particles.append((x, y))

    return particles

# Function to map particle positions to grid cells
def map_particles_to_grid(particles, d, grid_shape):
    num_cells_y, num_cells_x = grid_shape
    grid_particle_id = np.zeros((num_cells_y, num_cells_x), dtype=float)
    
    for ii, particle in enumerate(particles):
        x, y = particle
        cell_x = int(x / d)
        cell_y = int(y / d)
        grid_particle_id[cell_y, cell_x] = ii + 1  # Particle id starts from 1
    
    return grid_particle_id

# Function to plot the grid and particles
def plot_grid_and_particles(grid_particle_id, particles, d):
    fig, ax = plt.subplots()

    # Plot grid
    ax.pcolormesh(grid_particle_id, edgecolors='black', linewidth=1, cmap='Blues')

    # Plot particle circles
    for i, particle in enumerate(particles):
        x, y = particle
        circle = plt.Circle((x, y), d, color='red', fill=False)
        ax.add_patch(circle)
        ax.text(x, y, str(i + 1), ha='center', va='center', color='black')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Function to generate particle data array
def generate_particle_data(particles, vx_range=(-1, 1), vy_range=(-1, 1)):
    particle_data = np.zeros((len(particles), 7))
    
    for i, particle in enumerate(particles):
        particle_data[i, 0] = i + 1  # Particle id
        particle_data[i, 1:3] = particle  # x, y coordinates
        particle_data[i, 3] = np.random.uniform(*vx_range)  # Initial Vx
        particle_data[i, 4] = np.random.uniform(*vy_range)  # Initial Vy
        particle_data[i, 5:7] = 0  # Initial Fx and Fy
        
    return particle_data

# Function to generate a grid based on the particle data
def generate_property_grid(particle_data, property_index, grid_shape, cell_size):
    # property_index: index of the property in particle_data (e.g., 3 for Vx, 4 for Vy)
    num_cells_y, num_cells_x = grid_shape
    grid = np.zeros((num_cells_y, num_cells_x), dtype=float)

    for particle in particle_data:
        particle_id = int(particle[0])
        x, y = particle[1:3]
        property_value = particle[property_index]

        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)

        grid[cell_y, cell_x] = property_value

    return grid

def calculate_max_velocity(particle_data):
    vx_max = np.max(np.abs(particle_data[:, 3]))  # Index 3 is Vx in particle_data
    vy_max = np.max(np.abs(particle_data[:, 4]))  # Index 4 is Vy in particle_data
    return max(vx_max, vy_max)

def calculate_cfl_timestep(particle_diameter, max_velocity):
    return 0.5 * (particle_diameter / max_velocity)

# Inout Parameters ********************************************************************
l = 10  # Size of the square domain
d = 1   # Cell size and particle radius
simulation_time = 1
kn = 100; # kn is the normal stiffness of the spring
dn = 0.5; # dn is the normal damping coefficient
particle_mass = 0.01

# Module 1: Domain discretisation and initial particle insertion ***********************
# Create grid
grid = create_grid(l, d)
grid_shape = grid.shape
# Generate particles
particles = generate_particles(grid, l, d)
# Map particles to grid
grid_particle_id = map_particles_to_grid(particles, d, grid_shape)
mask = np.where(grid_particle_id != 0, 1, 0)
# Generate particle data array
particle_data = generate_particle_data(particles)
# plot_grid_and_particles(grid_particle_id, particles, d)
# exit()
# Generate X grid
x_grid = generate_property_grid(particle_data, property_index=1, grid_shape=grid.shape, cell_size=d)
# Generate Y grid
y_grid = generate_property_grid(particle_data, property_index=2, grid_shape=grid.shape, cell_size=d)
# Generate Vx grid
vx_grid = generate_property_grid(particle_data, property_index=3, grid_shape=grid.shape, cell_size=d)
# Generate Vy grid
vy_grid = generate_property_grid(particle_data, property_index=4, grid_shape=grid.shape, cell_size=d)

# Module 2: Contact detection using 5x5 filter, and contact force calculation *************
t = 0
dt = 2*np.pi*np.sqrt(particle_mass/kn)
# print(dt)

# ***************** Convert np.array into torch.tensor and transfer it to GPU ****************
filter_size = 5 
input_shape_local = (1,filter_size**2,grid_shape[0],grid_shape[1])
input_shape_global = (1,1,grid_shape[0],grid_shape[1])

diffx = torch.zeros(input_shape_local, device=device)
diffy = torch.zeros(input_shape_local, device=device)
zeros = torch.zeros(input_shape_local, device=device)
eplis = torch.ones(input_shape_local, device=device)*1e-04
fx_grid = torch.zeros(input_shape_global, device=device)
fy_grid = torch.zeros(input_shape_global, device=device)

mask = torch.reshape(torch.tensor(mask), input_shape_global).to(device)
x_grid = torch.reshape(torch.tensor(x_grid), input_shape_global).to(device)
y_grid = torch.reshape(torch.tensor(y_grid), input_shape_global).to(device)
vx_grid = torch.reshape(torch.tensor(vx_grid), input_shape_global).to(device)
vy_grid = torch.reshape(torch.tensor(vy_grid), input_shape_global).to(device)

# ***************** design filters to detect particel-to-particle interaction ****************
class AI4DEM(nn.Module):
    """docstring for AI4DEM"""
    def __init__(self):
        super(AI4DEM, self).__init__()
        # self.arg = arg

    def detector(self, grid, diff):
        """
        detect the neighbouring particles and calculate the distance between the two particles 
        0 0 0 0 0
        0 0 0 0 0 
        0 0 1 0 0
        0 0 0 0 0
        0 0 0 0 0 
        grid      : grid_x, grid_y (1,1,ny,nz)
        diff      : distance with 25 channels (1,25,ny,nz)
        dims(2, 3): 3 --> x axis (nx)     ; 2 --> y axis (ny) 
        rolling   : 2 --> filter size = 5 ; 1 --> filter size = 3
        ************************* Examples ***************************************
        shifts=(0, 1)   !shifts=(0, -2) 	!shifts=(-2, -2) 	!shifts=(1, -2)
        0 0 0 0 0	   	!0 0 0 0 0			!0 0 0 0 1			!0 0 0 0 0
        0 0 0 0 0 		!0 0 0 0 0 			!0 0 0 0 0 			!0 0 0 0 0 
        0 1 1 0 0   	!0 0 1 0 1			!0 0 1 0 0			!0 0 1 0 0
        0 0 0 0 0	   	!0 0 0 0 0 			!0 0 0 0 0			!0 0 0 0 1
        0 0 0 0 0      	!0 0 0 0 0 			!0 0 0 0 0  		!0 0 0 0 0 
        **************************************************************************
        """
        for i in range(5):
            for j in range(5):
                diff[0,i*5+j,:,:] = grid - torch.roll(grid, shifts=(j-2, i-2), dims=(2, 3))  
        return diff

    def forward(self, grid_x, grid_y, mask, d, kn, fx_grid, fy_grid, diffx, diffy):
        diffx = self.detector(grid_x, diffx)  	# 25
        diffy = self.detector(grid_y, diffy)  	# 25
        dist = torch.sqrt(diffx**2 + diffy**2) 	# 25
        Fx = torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffx / torch.maximum(eplis, dist), zeros) # 25
        Fy = torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffy / torch.maximum(eplis, dist), zeros) # 25
        fx_grid = torch.sum(Fx,1,keepdim=True) # 1
        fy_grid = torch.sum(Fy,1,keepdim=True) # 1
        return fx_grid*mask, fy_grid*mask 

model = AI4DEM().to(device)

# # # ################################### # # #
# # # #########  AI4DEM MAIN ############ # # #
# # # ################################### # # #
# Set up the figure and axis
fig, ax = plt.subplots()

while t < simulation_time:
    fx_grid = np.zeros_like(grid)
    fy_grid = np.zeros_like(grid)
    fx_grid, fy_grid = model(x_grid,y_grid, mask, d, kn, fx_grid, fy_grid, diffx, diffy) # to exclude cells with no particles in force calculations

    # Update velocity: Vel(tt+1) = Vel(tt) + F/particle_mass * delta_t
    vx_grid = vx_grid-(dt/particle_mass)*fx_grid
    vy_grid = vy_grid-(dt/particle_mass)*fy_grid
    # Update particle coordniates
    x_grid = x_grid + dt*vx_grid
    y_grid = y_grid + dt*vy_grid

    # Update particles and particle_data
    num_cells_y, num_cells_x = grid_shape
    grid_particle_id = np.zeros((num_cells_y, num_cells_x), dtype=float)
    particle_out_of_grid = []
    particle_in_grid = []
    fig, ax = plt.subplots()
    for i, particle in enumerate(particles):
        x, y = particle
        cell_x = int(x / d)
        cell_y = int(y / d)
        # Check if the indices are within the valid range
        if 0 <= x < l and 0 <= y < l:
            particles[i] = x_grid[0,0,cell_y,cell_x], y_grid[0,0,cell_y,cell_x]  # notice: dimension of array is (1,1,rows,cols) now 
            particle_data[i,1] = x_grid[0,0,cell_y,cell_x]
            particle_data[i,2] = y_grid[0,0,cell_y,cell_x]
            particle_data[i,3] = vx_grid[0,0,cell_y,cell_x]
            particle_data[i,4] = vy_grid[0,0,cell_y,cell_x]
            particle_data[i,5] = fx_grid[0,0,cell_y,cell_x]
            particle_data[i,6] = fy_grid[0,0,cell_y,cell_x]
            grid_particle_id[cell_y,cell_x] = i+1
            particle_in_grid.append(i)
            # Plot particle circles
            circle = plt.Circle((x, y), d, color='red', fill=False)
            ax.add_patch(circle)
            ax.text(x, y, str(i + 1), ha='center', va='center', color='black')
        else:
            particle_out_of_grid.append(i)

    grid_particle_multiplier = np.where(grid_particle_id != 0, 1, 0)
    # Plot grid
    ax.pcolormesh(grid_particle_id, edgecolors='black', linewidth=2, cmap='Blues', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Update maximum velocity
    vx_max = np.max(np.abs(particle_data[particle_in_grid, 3]))  # Index 3 is Vx in particle_data
    vy_max = np.max(np.abs(particle_data[particle_in_grid, 4]))  # Index 4 is Vy in particle_data

    # Update CFL timestep
    dt1 = calculate_cfl_timestep(d, max(vx_max, vy_max))
    dt2 = dt = 2*np.pi*np.sqrt(particle_mass/kn)
    dt = min(dt1, dt2)
    t +=dt

plt.show()


