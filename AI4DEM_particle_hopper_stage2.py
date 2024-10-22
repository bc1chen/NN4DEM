import os
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print("Using GPU:", is_gpu)

# Function to generate a structured grid

# Input Parameters

cell_size = 0.003  # Cell size and particle radius
simulation_time = 1
kn = 10000  # Normal stiffness of the spring
domain_size_x = 14
domain_size_y = 68
domain_size_z = 134

K_graph = 17*10000*1
S_graph = K_graph * (cell_size / domain_size_x) ** 2
restitution_coefficient = 0.79  # coefficient of restitution
friction_coefficient = 0.43  # coefficient of friction
wall_friction_coefficient = 0.8  # coefficient of friction


rho_p = 674 
particle_mass = 4/3*3.1415926*cell_size**3*rho_p #4188.7902


# particle_mass = 4/3*3.1415*cell_size**3*rho_p #4188.7902
half_domain_size_x = int(domain_size_x/2)+1
half_domain_size_y = int(domain_size_y/2)+1
half_domain_size_z = int(domain_size_z/2)+1
# coefficient of damping calculation 
# Discrete particle simulation of two-dimensional fluidized bed
damping_coefficient_Alpha      = -1 * math.log(restitution_coefficient)/math.pi
damping_coefficient_Gamma      = damping_coefficient_Alpha/math.sqrt(damping_coefficient_Alpha**2+1)
damping_coefficient_Eta        = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass/2)
damping_coefficient_Eta_wall   = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass)

print(damping_coefficient_Eta)

# Module 1: Domain discretisation and initial particle insertion
# Create grid

input_shape_global = (1, 1, domain_size_z, domain_size_y, domain_size_x)

# Generate particles

x_grid = np.zeros(input_shape_global)
y_grid = np.zeros(input_shape_global)
z_grid = np.zeros(input_shape_global)

vx_grid = np.zeros(input_shape_global)
vy_grid = np.zeros(input_shape_global)
vz_grid = np.zeros(input_shape_global)

angular_velocity_x = np.zeros(input_shape_global)
angular_velocity_y = np.zeros(input_shape_global)
angular_velocity_z = np.zeros(input_shape_global)

angular_x = np.zeros(input_shape_global)
angular_y = np.zeros(input_shape_global)
angular_z = np.zeros(input_shape_global)

mask = np.zeros(input_shape_global)

'''
x_grid[0, 0, 3, 34, 10] = 10*0.5
y_grid[0, 0, 3, 34, 10] = 34*0.5
z_grid[0, 0, 3, 34, 10] = 3*0.5
vx_grid[0, 0, 3, 34, 10] = 5*0.005
vy_grid[0, 0, 3, 34, 10] = 5*0.005
vz_grid[0, 0, 3, 34, 10] = -0.5
mask[0, 0, 3, 34, 10] = 1
'''

print('Number of particles:', np.count_nonzero(mask))


# Define the AI4DEM model
class AI4DEM(nn.Module):
    """AI4DEM model for particle interaction detection and force calculation"""
    def __init__(self):
        super(AI4DEM, self).__init__()

    def detector(self, grid, i, j, k):
        """Detect neighboring particles and calculate the distance between them"""
        diff = grid - torch.roll(grid, shifts=(k - 2, j - 2, i - 2), dims=(2, 3, 4))
        return diff

    def detector_angular_velocity(self, grid, i, j, k):
        """Detect neighboring particles and calculate the distance between them"""
        diff = grid + torch.roll(grid, shifts=(k - 2, j - 2, i - 2), dims=(2, 3, 4))
        return diff

    def forward(self, x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, angular_velocity_x, angular_velocity_y, angular_velocity_z, angular_x, angular_y, angular_z, fx_grid_collision, fy_grid_collision, fz_grid_collision, fx_grid_damping, fy_grid_damping, fz_grid_damping, mask, d, kn, damping_coefficient_Eta, friction_coefficient, diffx, diffy, diffz, diffvx, diffvy, diffvz, diff_angular_velocity_x, diff_angular_velocity_y, diff_angular_velocity_z, dt, input_shape, filter_size):
        cell_xold = x_grid / d
        cell_yold = y_grid / d 
        cell_zold = z_grid / d 
        
        cell_xold = torch.round(cell_xold).long()
        cell_yold = torch.round(cell_yold).long()
        cell_zold = torch.round(cell_zold).long()
        
        fx_grid_collision = torch.zeros(input_shape, device=device) 
        fy_grid_collision = torch.zeros(input_shape, device=device) 
        fz_grid_collision = torch.zeros(input_shape, device=device) 
        
        fx_grid_damping = torch.zeros(input_shape, device=device) 
        fy_grid_damping = torch.zeros(input_shape, device=device) 
        fz_grid_damping = torch.zeros(input_shape, device=device) 
        
        fx_grid_friction = torch.zeros(input_shape, device=device) 
        fy_grid_friction = torch.zeros(input_shape, device=device) 
        fz_grid_friction = torch.zeros(input_shape, device=device) 
        
        collision_torque_x = torch.zeros(input_shape, device=device) 
        collision_torque_y = torch.zeros(input_shape, device=device) 
        collision_torque_z = torch.zeros(input_shape, device=device) 
        
        fn_grid_collision = torch.zeros(input_shape, device=device) 
                        
        diffvx_Vn = torch.zeros(input_shape, device=device) 
        diffvy_Vn = torch.zeros(input_shape, device=device) 
        diffvz_Vn = torch.zeros(input_shape, device=device) 
        
        diffv_Vn  = torch.zeros(input_shape, device=device) 
        
        diffv_Vn_x = torch.zeros(input_shape, device=device) 
        diffv_Vn_y = torch.zeros(input_shape, device=device) 
        diffv_Vn_z = torch.zeros(input_shape, device=device) 
        
        diffv_Vt_x = torch.zeros(input_shape, device=device) 
        diffv_Vt_y = torch.zeros(input_shape, device=device) 
        diffv_Vt_z = torch.zeros(input_shape, device=device) 
        
        diffv_Vt = torch.zeros(input_shape, device=device) 
        
        for i in range(filter_size):
            for j in range(filter_size):
                for k in range(filter_size):
                    # calculate distance between the two particles
                    diffx = self.detector(x_grid, i, j, k) # individual
                    diffy = self.detector(y_grid, i, j, k) # individual
                    diffz = self.detector(z_grid, i, j, k) # individual
                    dist = torch.sqrt(diffx**2 + diffy**2 + diffz**2)  
                    
                    # calculate collision force between the two particles
                    fx_grid_collision =  fx_grid_collision + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffx / torch.maximum(eplis, dist), zeros) # individual
                    fy_grid_collision =  fy_grid_collision + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffy / torch.maximum(eplis, dist), zeros) # individual
                    fz_grid_collision =  fz_grid_collision + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffz / torch.maximum(eplis, dist), zeros) # individual            
                    
                    fn_grid_collision = torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) , zeros)
                    
                    diff_angular_velocity_x = self.detector_angular_velocity(angular_velocity_x, i, j, k) # individual
                    diff_angular_velocity_y = self.detector_angular_velocity(angular_velocity_y, i, j, k) # individual
                    diff_angular_velocity_z = self.detector_angular_velocity(angular_velocity_z, i, j, k) # individual
                    
                    moment_arm_x = d * diffx / torch.maximum(eplis, dist) 
                    moment_arm_y = d * diffy / torch.maximum(eplis, dist)
                    moment_arm_z = d * diffz / torch.maximum(eplis, dist)
                    
                    vx_angular = diff_angular_velocity_y * moment_arm_z * mask - diff_angular_velocity_z * moment_arm_y * mask
                    vy_angular = diff_angular_velocity_z * moment_arm_x * mask - diff_angular_velocity_x * moment_arm_z * mask
                    vz_angular = diff_angular_velocity_x * moment_arm_y * mask - diff_angular_velocity_y * moment_arm_x * mask

                    # calculate nodal velocity difference between the two particles
                    # diffv_Vn = self.detector(vx_grid, i, j, k) * diffx / torch.maximum(eplis, dist) + self.detector(vy_grid, i, j, k) * diffy / torch.maximum(eplis, dist) + self.detector(vz_grid, i, j, k)  * diffz / torch.maximum(eplis, dist)
                    diffvx = self.detector(vx_grid, i, j, k) # individual
                    diffvy = self.detector(vy_grid, i, j, k) # individual
                    diffvz = self.detector(vz_grid, i, j, k) # individual 
                    
                    diffvx_Vn =  (diffvx) * diffx /  torch.maximum(eplis, dist)
                    diffvy_Vn =  (diffvy) * diffy /  torch.maximum(eplis, dist)
                    diffvz_Vn =  (diffvz) * diffz /  torch.maximum(eplis, dist)
                    
                    diffv_Vn = diffvx_Vn + diffvy_Vn + diffvz_Vn
                    
                    fn_grid_damping =  torch.where(torch.lt(dist, 2*d), - damping_coefficient_Eta * diffv_Vn, zeros)
                    
                    # calculate the damping force between the two particles
                    diffv_Vn_x =  diffv_Vn * diffx /  torch.maximum(eplis, dist)
                    diffv_Vn_y =  diffv_Vn * diffy /  torch.maximum(eplis, dist)
                    diffv_Vn_z =  diffv_Vn * diffz /  torch.maximum(eplis, dist)         
                    
                    fx_grid_damping =  fx_grid_damping + torch.where(torch.lt(dist,2*d), damping_coefficient_Eta * diffv_Vn_x, zeros) # individual   
                    fy_grid_damping =  fy_grid_damping + torch.where(torch.lt(dist,2*d), damping_coefficient_Eta * diffv_Vn_y, zeros) # individual 
                    fz_grid_damping =  fz_grid_damping + torch.where(torch.lt(dist,2*d), damping_coefficient_Eta * diffv_Vn_z, zeros) # individual 
                    
                    fn_grid =torch.abs(fn_grid_collision + fn_grid_damping)
                    
                    # calculate angular velocity between the two particles
                    # calculate the friction force between the two particles
                    diffv_Vt_x = (diffvx) * (diffy**2 + diffz**2) / torch.maximum(eplis, dist**2)  + vx_angular
                    diffv_Vt_y = (diffvy) * (diffx**2 + diffz**2) / torch.maximum(eplis, dist**2)  + vy_angular
                    diffv_Vt_z = (diffvz) * (diffx**2 + diffy**2) / torch.maximum(eplis, dist**2)  + vz_angular
                    
                    diffv_Vt = torch.sqrt (diffv_Vt_x**2 + diffv_Vt_y**2 + diffv_Vt_z**2)
                    
                    fx_grid_friction =  fx_grid_friction - torch.where(torch.lt(dist, 2*d), friction_coefficient * fn_grid * diffv_Vt_x /  torch.maximum(eplis, diffv_Vt), zeros) # fx_grid_friction - torch.where(torch.lt(dist,2*d), friction_coefficient * fn_grid * diffv_Vt_x / torch.maximum(eplis,diffv_Vt), zeros)
                    fy_grid_friction =  fy_grid_friction - torch.where(torch.lt(dist, 2*d), friction_coefficient * fn_grid * diffv_Vt_y /  torch.maximum(eplis, diffv_Vt), zeros)
                    fz_grid_friction =  fz_grid_friction - torch.where(torch.lt(dist, 2*d), friction_coefficient * fn_grid * diffv_Vt_z /  torch.maximum(eplis, diffv_Vt), zeros)
                    
                    collision_torque_x = collision_torque_x + torch.where(torch.lt(dist, 2*d), fz_grid_friction * moment_arm_y * mask - fy_grid_friction * moment_arm_z * mask, zeros)
                    collision_torque_y = collision_torque_y + torch.where(torch.lt(dist, 2*d), fx_grid_friction * moment_arm_z * mask - fz_grid_friction * moment_arm_x * mask, zeros)
                    collision_torque_z = collision_torque_z + torch.where(torch.lt(dist, 2*d), fy_grid_friction * moment_arm_x * mask - fx_grid_friction * moment_arm_y * mask, zeros)

        # judge whether the particle is colliding the boundaries
        is_left_overlap     = torch.ne(x_grid, 0.0000) & torch.lt(x_grid, d) # Overlap with bottom wall
        is_right_overlap    = torch.gt(x_grid, domain_size_x*cell_size-2*d)# Overlap with bottom wall
        is_bottom_overlap   = torch.ne(y_grid, 0.0000) & torch.lt(y_grid, d) # Overlap with bottom wall
        is_top_overlap      = torch.gt(y_grid, domain_size_y*cell_size-2*d ) # Overlap with bottom wall
        is_forward_overlap  = torch.ne(z_grid, 0.0000) & torch.lt(z_grid, 3*d) & (torch.lt(y_grid, 29*d) | torch.gt(y_grid, 40*d)) # Overlap with bottom wall
        is_backward_overlap = torch.gt(z_grid, domain_size_z*cell_size-2*d ) # Overlap with bottom wall             
        
        is_left_overlap     = is_left_overlap.to(device)
        is_right_overlap    = is_right_overlap.to(device)
        is_bottom_overlap   = is_bottom_overlap.to(device)
        is_top_overlap      = is_top_overlap.to(device)
        is_forward_overlap  = is_forward_overlap.to(device)
        is_backward_overlap = is_backward_overlap.to(device)
        
        # calculate the elastic force from the boundaries
        fx_grid_boundary_left     = kn * torch.where(is_left_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (d - x_grid)
        fx_grid_boundary_right    = kn * torch.where(is_right_overlap,   torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (x_grid - domain_size_x*cell_size + 2*d)
        fy_grid_boundary_bottom   = kn * torch.where(is_bottom_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (d - y_grid)
        fy_grid_boundary_top      = kn * torch.where(is_top_overlap,     torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (y_grid - domain_size_y*cell_size + 2*d)
        fz_grid_boundary_forward  = kn * torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (3*d - z_grid)
        fz_grid_boundary_backward = kn * torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (z_grid - domain_size_z*cell_size + 2*d)
        
        # calculate damping force from the boundaries
        fx_grid_left_damping      = damping_coefficient_Eta_wall * vx_grid *torch.where(is_left_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fx_grid_right_damping     = damping_coefficient_Eta_wall * vx_grid *torch.where(is_right_overlap,   torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fy_grid_bottom_damping    = damping_coefficient_Eta_wall * vy_grid *torch.where(is_bottom_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fy_grid_top_damping       = damping_coefficient_Eta_wall * vy_grid *torch.where(is_top_overlap,     torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fz_grid_forward_damping   = damping_coefficient_Eta_wall * vz_grid *torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fz_grid_backward_damping  = damping_coefficient_Eta_wall * vz_grid *torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask

        # calculate friction force from the boundaries
        fx_grid_left_friction_y      = - torch.where(is_left_overlap,    wall_friction_coefficient * torch.abs (  fx_grid_boundary_left     - fx_grid_left_damping    )   * (vy_grid - d*angular_velocity_z) / torch.maximum(eplis, torch.sqrt(vx_grid**2+(vy_grid - d*angular_velocity_z)**2+(vz_grid + d*angular_velocity_y)**2)), torch.tensor(0.0, device=device))
        fx_grid_left_friction_z      = - torch.where(is_left_overlap,    wall_friction_coefficient * torch.abs (  fx_grid_boundary_left     - fx_grid_left_damping    )   * (vz_grid + d*angular_velocity_y) / torch.maximum(eplis, torch.sqrt(vx_grid**2+(vy_grid - d*angular_velocity_z)**2+(vz_grid + d*angular_velocity_y)**2)), torch.tensor(0.0, device=device))

        fx_grid_right_friction_y     = - torch.where(is_right_overlap,   wall_friction_coefficient * torch.abs (- fx_grid_boundary_right    - fx_grid_right_damping   )   * (vy_grid + d*angular_velocity_z) / torch.maximum(eplis, torch.sqrt(vx_grid**2+(vy_grid + d*angular_velocity_z)**2+(vz_grid - d*angular_velocity_y)**2)), torch.tensor(0.0, device=device))
        fx_grid_right_friction_z     = - torch.where(is_right_overlap,   wall_friction_coefficient * torch.abs (- fx_grid_boundary_right    - fx_grid_right_damping   )   * (vz_grid - d*angular_velocity_y) / torch.maximum(eplis, torch.sqrt(vx_grid**2+(vy_grid + d*angular_velocity_z)**2+(vz_grid - d*angular_velocity_y)**2)), torch.tensor(0.0, device=device))

        fy_grid_bottom_friction_x    = - torch.where(is_bottom_overlap,  wall_friction_coefficient * torch.abs (  fy_grid_boundary_bottom   - fy_grid_bottom_damping  )   * (vx_grid + d*angular_velocity_z) / torch.maximum(eplis, torch.sqrt((vx_grid + d*angular_velocity_z)**2+vy_grid**2+(vz_grid - d*angular_velocity_x)**2)), torch.tensor(0.0, device=device))
        fy_grid_bottom_friction_z    = - torch.where(is_bottom_overlap,  wall_friction_coefficient * torch.abs (  fy_grid_boundary_bottom   - fy_grid_bottom_damping  )   * (vz_grid - d*angular_velocity_x) / torch.maximum(eplis, torch.sqrt((vx_grid + d*angular_velocity_z)**2+vy_grid**2+(vz_grid - d*angular_velocity_x)**2)), torch.tensor(0.0, device=device))
                
        fy_grid_top_friction_x       = - torch.where(is_top_overlap,     wall_friction_coefficient * torch.abs (- fy_grid_boundary_top      - fy_grid_top_damping     )   * (vx_grid - d*angular_velocity_z) / torch.maximum(eplis, torch.sqrt((vx_grid - d*angular_velocity_z)**2+vy_grid**2+(vz_grid + d*angular_velocity_x)**2)), torch.tensor(0.0, device=device))
        fy_grid_top_friction_z       = - torch.where(is_top_overlap,     wall_friction_coefficient * torch.abs (- fy_grid_boundary_top      - fy_grid_top_damping     )   * (vz_grid + d*angular_velocity_x) / torch.maximum(eplis, torch.sqrt((vx_grid - d*angular_velocity_z)**2+vy_grid**2+(vz_grid + d*angular_velocity_x)**2)), torch.tensor(0.0, device=device))
    
        fz_grid_forward_friction_x   = - torch.where(is_forward_overlap, wall_friction_coefficient * torch.abs (  fz_grid_boundary_forward  - fz_grid_forward_damping )   * (vx_grid - d*angular_velocity_y) / torch.maximum(eplis, torch.sqrt((vx_grid - d*angular_velocity_y)**2+(vy_grid + d*angular_velocity_x)**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fz_grid_forward_friction_y   = - torch.where(is_forward_overlap, wall_friction_coefficient * torch.abs (  fz_grid_boundary_forward  - fz_grid_forward_damping )   * (vy_grid + d*angular_velocity_x) / torch.maximum(eplis, torch.sqrt((vx_grid - d*angular_velocity_y)**2+(vy_grid + d*angular_velocity_x)**2+vz_grid**2)), torch.tensor(0.0, device=device))
        
        fz_grid_backward_friction_x  = - torch.where(is_backward_overlap,wall_friction_coefficient * torch.abs (- fz_grid_boundary_backward - fz_grid_backward_damping)   * (vx_grid + d*angular_velocity_y) / torch.maximum(eplis, torch.sqrt((vx_grid + d*angular_velocity_y)**2+(vy_grid - d*angular_velocity_x)**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fz_grid_backward_friction_y  = - torch.where(is_backward_overlap,wall_friction_coefficient * torch.abs (- fz_grid_boundary_backward - fz_grid_backward_damping)   * (vy_grid - d*angular_velocity_x) / torch.maximum(eplis, torch.sqrt((vx_grid + d*angular_velocity_y)**2+(vy_grid - d*angular_velocity_x)**2+vz_grid**2)), torch.tensor(0.0, device=device))
        
        torque_x = mask * d * (fy_grid_top_friction_z    - fy_grid_bottom_friction_z + fz_grid_forward_friction_y  - fz_grid_backward_friction_y) + collision_torque_x
        torque_y = mask * d * (fx_grid_left_friction_z   - fx_grid_right_friction_z  + fz_grid_backward_friction_x - fz_grid_forward_friction_x ) + collision_torque_y
        torque_z = mask * d * (fy_grid_bottom_friction_x - fy_grid_top_friction_x    + fx_grid_right_friction_y    - fx_grid_left_friction_y    ) + collision_torque_z

        particle_inertia = (2/5) * particle_mass * d**2
        
        angular_velocity_x =  angular_velocity_x + torque_x * dt / particle_inertia
        angular_velocity_y =  angular_velocity_y + torque_y * dt / particle_inertia
        angular_velocity_z =  angular_velocity_z + torque_z * dt / particle_inertia
        
        angular_x = angular_x + angular_velocity_x * dt
        angular_y = angular_y + angular_velocity_y * dt
        angular_z = angular_z + angular_velocity_z * dt     
        
        # calculate the new velocity with acceleration calculated by forces
        vx_grid = vx_grid  +  (dt / particle_mass) * ( - 0   * particle_mass) * mask + (dt / particle_mass) * ( - fx_grid_boundary_right    + fx_grid_boundary_left    - fx_grid_collision - fx_grid_damping - fx_grid_left_damping    - fx_grid_right_damping    + fx_grid_friction + fz_grid_forward_friction_x + fz_grid_backward_friction_x + fy_grid_bottom_friction_x + fy_grid_top_friction_x  ) * mask
        vy_grid = vy_grid  +  (dt / particle_mass) * ( - 0   * particle_mass) * mask + (dt / particle_mass) * ( - fy_grid_boundary_top      + fy_grid_boundary_bottom  - fy_grid_collision - fy_grid_damping - fy_grid_bottom_damping  - fy_grid_top_damping      + fy_grid_friction + fz_grid_forward_friction_y + fz_grid_backward_friction_y + fx_grid_left_friction_y   + fx_grid_right_friction_y) * mask 
        vz_grid = vz_grid  +  (dt / particle_mass) * ( -9.8  * particle_mass) * mask + (dt / particle_mass) * ( - fz_grid_boundary_backward + fz_grid_boundary_forward - fz_grid_collision - fz_grid_damping - fz_grid_forward_damping - fz_grid_backward_damping + fz_grid_friction + fy_grid_bottom_friction_z  + fy_grid_top_friction_z      + fx_grid_left_friction_z   + fx_grid_right_friction_z )* mask 
       
        # Update particle coordniates
        x_grid = x_grid + dt * vx_grid
        y_grid = y_grid + dt * vy_grid
        z_grid = z_grid + dt * vz_grid
                
        x_grid_merge = x_grid.clone()
        y_grid_merge = y_grid.clone()
        z_grid_merge = z_grid.clone()
                
        vx_grid_merge = vx_grid.clone()
        vy_grid_merge = vy_grid.clone()
        vz_grid_merge = vz_grid.clone()
        
        angular_velocity_x_merge = angular_velocity_x.clone()
        angular_velocity_y_merge = angular_velocity_y.clone()
        angular_velocity_z_merge = angular_velocity_z.clone()
        
        angular_x_merge = angular_x.clone()
        angular_y_merge = angular_y.clone()
        angular_z_merge = angular_z.clone()
        
        
        # update new index tensor 
        cell_x = x_grid / d 
        cell_y = y_grid / d     
        cell_z = z_grid / d     
                
        cell_x = torch.round(cell_x).long()
        cell_y = torch.round(cell_y).long()    
        cell_z = torch.round(cell_z).long()  
        
        # extract index (previous and new) from sparse index tensor (previous and new)
        cell_x = cell_x[cell_x!=0]
        cell_y = cell_y[cell_y!=0]         
        cell_z = cell_z[cell_z!=0]         
                
        cell_xold = cell_xold[cell_xold!=0]
        cell_yold = cell_yold[cell_yold!=0]   
        cell_zold = cell_zold[cell_zold!=0]   

        # get rid of values at previous index 
        mask[0,0,cell_zold.long(), cell_yold.long(), cell_xold.long()] = 0
        x_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        y_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        z_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
               
        vx_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        vy_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        vz_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        
        angular_x[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        angular_y[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        angular_z[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
               
        angular_velocity_x[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        angular_velocity_y[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        angular_velocity_z[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
        
        
        # update new values based on new index         
        mask[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = 1
        x_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = x_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        y_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = y_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        z_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = z_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 

        vx_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vx_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        vy_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vy_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
        vz_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vz_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
        
        angular_x[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = angular_x_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        angular_y[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = angular_y_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        angular_z[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = angular_z_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 

        angular_velocity_x[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = angular_velocity_x_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        angular_velocity_y[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = angular_velocity_y_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
        angular_velocity_z[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = angular_velocity_z_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
        
        x_grid_next = x_grid + dt * vx_grid
        y_grid_next = y_grid + dt * vy_grid 
        z_grid_next = z_grid + dt * vz_grid*1.1
        is_bottom_out = torch.ne(z_grid_next, 0.0000) & torch.lt(z_grid_next, 0.65*d) # Overlap with bottom wall
        
        combined_condition = is_bottom_out
        x_grid[combined_condition] = 0
        y_grid[combined_condition] = 0
        z_grid[combined_condition] = 0
        vx_grid[combined_condition] = 0
        vy_grid[combined_condition] = 0
        vz_grid[combined_condition] = 0   
        mask[combined_condition] = 0   
                
        angular_velocity_x[combined_condition] = 0
        angular_velocity_y[combined_condition] = 0
        angular_velocity_z[combined_condition] = 0
        angular_x[combined_condition] = 0
        angular_y[combined_condition] = 0
        angular_z[combined_condition] = 0        
        return x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, angular_velocity_x, angular_velocity_y, angular_velocity_z, angular_x, angular_y, angular_z, mask

model = AI4DEM().to(device)
# model = torch.compile(model, mode="reduce-overhead")
# Module 2: Contact detection and force calculation
t = 0
dt = 0.000008  # 0.0001
ntime = 10000000000
# Convert np.array into torch.tensor and transfer it to GPU
filter_size = 5 

# Initialize tensors
diffx = torch.zeros(input_shape_global, device=device)
diffy = torch.zeros(input_shape_global, device=device)
diffz = torch.zeros(input_shape_global, device=device)

diffvx = torch.zeros(input_shape_global, device=device)
diffvy = torch.zeros(input_shape_global, device=device)
diffvz = torch.zeros(input_shape_global, device=device)

diff_angular_velocity_x = torch.zeros(input_shape_global, device=device)
diff_angular_velocity_y = torch.zeros(input_shape_global, device=device)
diff_angular_velocity_z = torch.zeros(input_shape_global, device=device)

zeros = torch.zeros(input_shape_global, device=device)
eplis = torch.ones(input_shape_global, device=device) * 1e-04

fx_grid_collision = torch.zeros(input_shape_global, device=device)
fy_grid_collision = torch.zeros(input_shape_global, device=device)
fz_grid_collision = torch.zeros(input_shape_global, device=device)

fx_grid_damping = torch.zeros(input_shape_global, device=device)
fy_grid_damping = torch.zeros(input_shape_global, device=device)
fz_grid_damping = torch.zeros(input_shape_global, device=device)

mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)

vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)
vz_grid = torch.from_numpy(vz_grid).float().to(device)

angular_velocity_x = torch.from_numpy(angular_velocity_x).float().to(device)
angular_velocity_y = torch.from_numpy(angular_velocity_y).float().to(device)
angular_velocity_z = torch.from_numpy(angular_velocity_z).float().to(device)

angular_x = torch.from_numpy(angular_x).float().to(device)
angular_y = torch.from_numpy(angular_y).float().to(device)
angular_z = torch.from_numpy(angular_z).float().to(device)

torque_x = torch.zeros(input_shape_global, device=device)
torque_y = torch.zeros(input_shape_global, device=device)
torque_z = torch.zeros(input_shape_global, device=device)

# Main simulation loop
start = time.time()
with torch.no_grad():
    for itime in range(1, ntime + 1):
        [x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, angular_velocity_x, angular_velocity_y, angular_velocity_z, angular_x, angular_y, angular_z, mask] = model(x_grid, y_grid,z_grid, vx_grid, vy_grid, vz_grid, angular_velocity_x, angular_velocity_y, angular_velocity_z, angular_x, angular_y, angular_z, fx_grid_collision, fy_grid_collision, fz_grid_collision, fx_grid_damping, fy_grid_damping, fz_grid_damping, mask, cell_size, kn, damping_coefficient_Eta, friction_coefficient, diffx, diffy, diffz, diffvx, diffvy, diffvz, diff_angular_velocity_x, diff_angular_velocity_y, diff_angular_velocity_z, dt, input_shape_global, filter_size)
        print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
        '''
        print(z_grid[z_grid!=0])
        '''
        if itime % 600 == 0:
            save_name = "hopper/"+str(itime)
            torch.save(x_grid, save_name + 'x_grid.pt')
            torch.save(y_grid, save_name + 'y_grid.pt')
            torch.save(z_grid, save_name + 'z_grid.pt')
            torch.save(vx_grid, save_name + 'vx_grid.pt')
            torch.save(vy_grid, save_name + 'vy_grid.pt')
            torch.save(vz_grid, save_name + 'vz_grid.pt')
            torch.save(angular_velocity_x, save_name + 'angular_velocity_x_grid.pt')
            torch.save(angular_velocity_y, save_name + 'angular_velocity_y_grid.pt')
            torch.save(angular_velocity_z, save_name + 'angular_velocity_z_grid.pt')

            torch.save(torch.count_nonzero(mask), save_name + 'Number_of_particles.pt')
        
        '''
        if itime == 1:
            for i in range(1, half_domain_size_x-1):
                for j in range(1, half_domain_size_y-1):
                    for k in range(2,  half_domain_size_z-2):
                        x_grid[0, 0, k*2, j*2, i*2] = i * cell_size * 2
                        y_grid[0, 0, k*2, j*2, i*2] = j * cell_size * 2
                        z_grid[0, 0, k*2, j*2, i*2] = k * cell_size * 2
                        
                        vx_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.001
                        vy_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.001
                        vz_grid[0, 0, k*2, j*2, i*2] = -0.00003
        '''
        if itime == 1:
            x_grid = torch.load('16200x_grid.pt')
            y_grid = torch.load('16200y_grid.pt')
            z_grid = torch.load('16200z_grid.pt')

            
            '''
        if itime % 100 == 0:
            
            # Visualize particles
            xp = x_grid[x_grid!=0].cpu() 
            yp = y_grid[y_grid!=0].cpu() 
            zp = z_grid[z_grid!=0].cpu() 

            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(xp,yp,zp)
            ax = plt.gca()

            ax.set_xlim([0, domain_size_x*cell_size-cell_size])
            ax.set_ylim([0, domain_size_y*cell_size-cell_size])
            ax.set_zlim([0, domain_size_z*cell_size-cell_size])
            ax.view_init(elev=45, azim=45)
            ax.set_xlabel('x')
            ax.set_ylabel('y')            
            ax.set_zlabel('z')


            # Save visualization
            if itime < 10:
                save_name = "3D_new/"+str(itime)+".jpg"
            elif itime >= 10 and itime < 100:
                save_name = "3D_new/"+str(itime)+".jpg"
            elif itime >= 100 and itime < 1000:
                save_name = "3D_new/"+str(itime)+".jpg"
            elif itime >= 1000 and itime < 10000:
                save_name = "3D_new/"+str(itime)+".jpg"
            else:
                save_name = "3D_new/"+str(itime)+".jpg"
            plt.savefig(save_name, dpi=1000, bbox_inches='tight')
            plt.close()
            '''
end = time.time()
print('Elapsed time:', end - start)
