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

# # Function to generate a structured grid
# def create_grid(domain_size, cell_size):
#     num_cells = int(domain_size / cell_size)
#     grid = np.zeros((num_cells, num_cells, num_cells), dtype=int)
#     return grid

# Input Parameters
domain_size = 101  # Size of the square domain
domain_height = 100
half_domain_size = int(domain_size/2)+1
cell_size = 0.05   # Cell size 
R_p = cell_size * 1
simulation_time = 1
kn = 600000#0#000  # Normal stiffness of the spring
dn = 0.5  # Normal damping coefficient
rho_p = 2700 
particle_mass = 4/3*3.1415*R_p**3*rho_p #4188.7902

print('mass of particle:', particle_mass)

K_graph = 17*10000*1
S_graph = K_graph * (cell_size / domain_size) ** 2
restitution_coefficient = 0.5  # coefficient of restitution
friction_coefficient = 0.5  # coefficient of friction
# coefficient of damping calculation 
# Discrete particle simulation of two-dimensional fluidized bed
damping_coefficient_Alpha      = -1 * math.log(restitution_coefficient)/math.pi
damping_coefficient_Gamma      = damping_coefficient_Alpha/math.sqrt(damping_coefficient_Alpha**2+1)
damping_coefficient_Eta        = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass/2)
damping_coefficient_Eta_wall   = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass)

print('Damping Coefficient:', damping_coefficient_Eta)

# Module 1: Domain discretisation and initial particle insertion
# Create grid
# grid = create_grid(domain_size, cell_size)
# grid_shape = grid.shape
input_shape_global = (1, 1, domain_height, domain_size, domain_size)

# Generate particles
# npt = int(domain_size ** 3)
'''
x_grid = torch.zeros(input_shape_global, device=device)
y_grid = torch.zeros(input_shape_global, device=device)
z_grid = torch.zeros(input_shape_global, device=device)

vx_grid = torch.zeros(input_shape_global, device=device)
vy_grid = torch.zeros(input_shape_global, device=device)
vz_grid = torch.zeros(input_shape_global, device=device)
mask = torch.zeros(input_shape_global, device=device)

'''
x_grid = np.zeros(input_shape_global)
y_grid = np.zeros(input_shape_global)
z_grid = np.zeros(input_shape_global)

vx_grid = np.zeros(input_shape_global)
vy_grid = np.zeros(input_shape_global)
vz_grid = np.zeros(input_shape_global)
mask = np.zeros(input_shape_global)


'''
for i in range(1, half_domain_size-1):
    for j in range(1, half_domain_size-1):
        for k in range(1, half_domain_size-1): 
            x_grid[0, 0, domain_size -3, j*2, i*2] = i * cell_size *2
            y_grid[0, 0, domain_size -3, j*2, i*2] = j * cell_size *2
            z_grid[0, 0, domain_size -3, j*2, i*2] =  domain_size -3
            vx_grid[0, 0, domain_size -3, j*2, i*2] = random.uniform(-1.0,1.0)
            mask[0, 0, domain_size -3, j*2, i*2] = 1

x_grid[0, 0, 10, 10, 10] = 10
y_grid[0, 0, 10, 10, 10] = 10
z_grid[0, 0, 10, 10, 10] = 10
vz_grid[0, 0, 10, 10, 10] = 10
mask[0, 0, 10, 10, 10] = 1

x_grid[0, 0, 1, 10, 10] = 10
y_grid[0, 0, 1, 10, 10] = 10
z_grid[0, 0, 1, 10, 10] = 1
vz_grid[0, 0, 1, 10, 10] = -10
mask[0, 0, 1, 10, 10] = 1

x_grid[0, 0, 10, 10, 10] = 10
y_grid[0, 0, 10, 10, 10] = 10
z_grid[0, 0, 10, 10, 10] = 10
vx_grid[0, 0, 10, 10, 10] = 10
mask[0, 0, 10, 10, 10] = 1
'''

'''
x_grid[0, 0, 1, 1, 1] = 1
y_grid[0, 0, 1, 1, 1] = 1
z_grid[0, 0, 1, 1, 1] = 1
vx_grid[0, 0, 1, 1, 1] = 0
vy_grid[0, 0, 1, 1, 1] = 0
mask[0, 0, 1, 1, 1] = 1
'''
'''
x_grid[0, 0, 1, 10, 10] = 10*0.05
y_grid[0, 0, 1, 10, 10] = 10*0.05
z_grid[0, 0, 1, 10, 10] = 1*0.05
vx_grid[0, 0, 1, 10, 10] = 5*0.05
vy_grid[0, 0, 1, 10, 10] = 5*0.05
vz_grid[0, 0, 1, 10, 10] = 0
mask[0, 0, 1, 10, 10] = 1
'''
'''
x_grid[0, 0, 12, 10, 10] = 10
y_grid[0, 0, 12, 10, 10] = 10
z_grid[0, 0, 12, 10, 10] = 12
vx_grid[0, 0, 12, 10, 10] = 10
vy_grid[0, 0, 12, 10, 10] = 0
vz_grid[0, 0, 12, 10, 10] = -10
mask[0, 0, 12, 10, 10] = 1
'''



# initialise particles distribution (more efficient)
i, j = np.meshgrid(np.arange(1, half_domain_size-1), np.arange(1, half_domain_size-1))
x_grid[0, 0, domain_height-3, j*2, i*2] = i * cell_size * 2
y_grid[0, 0, domain_height-3, j*2, i*2] = j * cell_size * 2
z_grid[0, 0, domain_height-3, j*2, i*2] = (domain_height - 3) * cell_size
vz_grid[0, 0, domain_height-3, j*2, i*2] = -5*0.1
vx_grid[0, 0, domain_height-3, j*2, i*2] = random.uniform(-1.0,1.0)*0.1
vy_grid[0, 0, domain_height-3, j*2, i*2] = random.uniform(-1.0,1.0)*0.1

mask[0, 0, domain_height-3, j*2, i*2] = 1
del i,j
''''''
''''''
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

    def forward(self, x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, d, R_p, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape, filter_size):
        cell_xold = x_grid / d # 2.4 / 0.5 
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
        
        # fx_grid_friction = torch.zeros(input_shape, device=device) 
        # fy_grid_friction = torch.zeros(input_shape, device=device) 
        # fz_grid_friction = torch.zeros(input_shape, device=device) 
        
        # fn_grid_collision = torch.zeros(input_shape, device=device) 
        
        # diffvx_Vn = torch.zeros(input_shape, device=device) 
        # diffvy_Vn = torch.zeros(input_shape, device=device) 
        # diffvz_Vn = torch.zeros(input_shape, device=device) 
        
        # diffv_Vn  = torch.zeros(input_shape, device=device) 
        
        # diffv_Vn_x = torch.zeros(input_shape, device=device) 
        # diffv_Vn_y = torch.zeros(input_shape, device=device) 
        # diffv_Vn_z = torch.zeros(input_shape, device=device) 
        
        # diffv_Vt_x = torch.zeros(input_shape, device=device) 
        # diffv_Vt_y = torch.zeros(input_shape, device=device) 
        # diffv_Vt_z = torch.zeros(input_shape, device=device) 
        
        # diffv_Vt = torch.zeros(input_shape, device=device) 
        
        for i in range(filter_size):
            for j in range(filter_size):
                for k in range(filter_size):
                    # calculate distance between the two particles
                    diffx = self.detector(x_grid, i, j, k) # individual
                    diffy = self.detector(y_grid, i, j, k) # individual
                    diffz = self.detector(z_grid, i, j, k) # individual
                    dist = torch.sqrt(diffx**2 + diffy**2 + diffz**2)  

                    # calculate collision force between the two particles
                    fx_grid_collision =  fx_grid_collision + torch.where(torch.lt(dist,2*R_p), kn * (dist - 2 * d ) * diffx / torch.maximum(eplis, dist), zeros) # individual
                    fy_grid_collision =  fy_grid_collision + torch.where(torch.lt(dist,2*R_p), kn * (dist - 2 * d ) * diffy / torch.maximum(eplis, dist), zeros) # individual
                    fz_grid_collision =  fz_grid_collision + torch.where(torch.lt(dist,2*R_p), kn * (dist - 2 * d ) * diffz / torch.maximum(eplis, dist), zeros) # individual            
                    
                    # fn_grid_collision = torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) , zeros)
                    
                    # calculate nodal velocity difference between the two particles
                    # diffv_Vn = self.detector(vx_grid, i, j, k) * diffx / torch.maximum(eplis, dist) + self.detector(vy_grid, i, j, k) * diffy / torch.maximum(eplis, dist) + self.detector(vz_grid, i, j, k)  * diffz / torch.maximum(eplis, dist)
                    # diffvx = self.detector(vx_grid, i, j, k) # individual
                    # diffvy = self.detector(vy_grid, i, j, k) # individual
                    # diffvz = self.detector(vz_grid, i, j, k) # individual 
                    
                    diffvx_Vn =  self.detector(vx_grid, i, j, k) * diffx /  torch.maximum(eplis, dist)
                    diffvy_Vn =  self.detector(vy_grid, i, j, k) * diffy /  torch.maximum(eplis, dist)
                    diffvz_Vn =  self.detector(vz_grid, i, j, k) * diffz /  torch.maximum(eplis, dist) 
                    
                    diffv_Vn = diffvx_Vn + diffvy_Vn + diffvz_Vn
                    
                    # fn_grid_damping =  torch.where(torch.lt(dist, 2*d), - damping_coefficient_Eta * diffv_Vn, zeros)
                    
                    # calculate the damping force between the two particles
                    diffv_Vn_x =  diffv_Vn * diffx /  torch.maximum(eplis, dist)
                    diffv_Vn_y =  diffv_Vn * diffy /  torch.maximum(eplis, dist)
                    diffv_Vn_z =  diffv_Vn * diffz /  torch.maximum(eplis, dist)         
                    
                    fx_grid_damping =  fx_grid_damping + torch.where(torch.lt(dist,2*R_p), damping_coefficient_Eta * diffv_Vn_x, zeros) # individual   
                    fy_grid_damping =  fy_grid_damping + torch.where(torch.lt(dist,2*R_p), damping_coefficient_Eta * diffv_Vn_y, zeros) # individual 
                    fz_grid_damping =  fz_grid_damping + torch.where(torch.lt(dist,2*R_p), damping_coefficient_Eta * diffv_Vn_z, zeros) # individual 
                    
                    # fn_grid =torch.abs(fn_grid_collision + fn_grid_damping)
                    
                    # calculate the friction force between the two particles
                    
                    # diffv_Vt_x = diffvx * (diffy**2 + diffz**2) / torch.maximum(eplis, dist**2)
                    # diffv_Vt_y = diffvy * (diffx**2 + diffz**2) / torch.maximum(eplis, dist**2)
                    # diffv_Vt_z = diffvz * (diffx**2 + diffy**2) / torch.maximum(eplis, dist**2)
                    
                    # diffv_Vt = torch.sqrt (diffv_Vt_x**2 + diffv_Vt_y**2 + diffv_Vt_z**2)
                    '''
                    fx_grid_friction =  fx_grid_friction - torch.where(torch.lt(dist, 2*d), friction_coefficient * fn_grid * diffv_Vt_x /  torch.maximum(eplis, diffv_Vt), zeros) # fx_grid_friction - torch.where(torch.lt(dist,2*d), friction_coefficient * fn_grid * diffv_Vt_x / torch.maximum(eplis,diffv_Vt), zeros)
                    fy_grid_friction =  fy_grid_friction - torch.where(torch.lt(dist, 2*d), friction_coefficient * fn_grid * diffv_Vt_y /  torch.maximum(eplis, diffv_Vt), zeros)
                    fz_grid_friction =  fz_grid_friction - torch.where(torch.lt(dist, 2*d), friction_coefficient * fn_grid * diffv_Vt_z /  torch.maximum(eplis, diffv_Vt), zeros)
                    '''
        del diffx, diffy, diffz, diffvx_Vn, diffvy_Vn, diffvz_Vn, diffv_Vn, diffv_Vn_x, diffv_Vn_y, diffv_Vn_z

        # judge whether the particle is colliding the boundaries
        is_left_overlap     = torch.ne(x_grid, 0.0000) & torch.lt(x_grid, R_p) # Overlap with bottom wall
        is_right_overlap    = torch.gt(x_grid, domain_size*cell_size-2*R_p)# Overlap with bottom wall
        is_bottom_overlap   = torch.ne(y_grid, 0.0000) & torch.lt(y_grid, R_p) # Overlap with bottom wall
        is_top_overlap      = torch.gt(y_grid, domain_size*cell_size-2*R_p ) # Overlap with bottom wall
        is_forward_overlap  = torch.ne(z_grid, 0.0000) & torch.lt(z_grid, R_p) # Overlap with bottom wall
        is_backward_overlap = torch.gt(z_grid, domain_size*cell_size-2*R_p ) # Overlap with bottom wall             
        
        # is_left_overlap     = is_left_overlap.to(device)
        # is_right_overlap    = is_right_overlap.to(device)
        # is_bottom_overlap   = is_bottom_overlap.to(device)
        # is_top_overlap      = is_top_overlap.to(device)
        # is_forward_overlap  = is_forward_overlap.to(device)
        # is_backward_overlap = is_backward_overlap.to(device)
        
        # calculate the elastic force from the boundaries
        fx_grid_boundary_left     = kn * torch.where(is_left_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (R_p - x_grid)
        fx_grid_boundary_right    = kn * torch.where(is_right_overlap,   torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (x_grid - domain_size*cell_size + 2*R_p)
        fy_grid_boundary_bottom   = kn * torch.where(is_bottom_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (R_p - y_grid)
        fy_grid_boundary_top      = kn * torch.where(is_top_overlap,     torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (y_grid - domain_size*cell_size + 2*R_p)
        fz_grid_boundary_forward  = kn * torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (R_p - z_grid)
        fz_grid_boundary_backward = kn * torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (z_grid - domain_size*cell_size + 2*R_p)
        '''
        print(fx_grid_boundary_right[fx_grid_boundary_right!=0].cpu())
        print(fx_grid_boundary_left[fx_grid_boundary_left!=0].cpu())
        print(fy_grid_boundary_bottom[fy_grid_boundary_bottom!=0].cpu())
                
        print(fx_grid_friction[fx_grid_friction!=0].cpu())
        print(fy_grid_friction[fy_grid_friction!=0].cpu())
        print(fz_grid_friction[fz_grid_friction!=0].cpu())
        
        print(diffx[diffx!=0])
        print(diffy[diffx!=0])
        print(diffy[diffx!=0])

        print(fx_grid_collision[fx_grid_collision!=0].cpu())
        print(fy_grid_collision[fy_grid_collision!=0].cpu())
        print(fz_grid_collision[fz_grid_collision!=0].cpu())
        
        print(fx_grid_damping[fx_grid_damping!=0].cpu())
        print(fy_grid_damping[fy_grid_damping!=0].cpu())
        print(fz_grid_friction[fz_grid_friction!=0].cpu())
        '''
        # calculate damping force from the boundaries
        fx_grid_left_damping      = damping_coefficient_Eta_wall * vx_grid *torch.where(is_left_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fx_grid_right_damping     = damping_coefficient_Eta_wall * vx_grid *torch.where(is_right_overlap,   torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fy_grid_bottom_damping    = damping_coefficient_Eta_wall * vy_grid *torch.where(is_bottom_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fy_grid_top_damping       = damping_coefficient_Eta_wall * vy_grid *torch.where(is_top_overlap,     torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fz_grid_forward_damping   = damping_coefficient_Eta_wall * vz_grid *torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
        fz_grid_backward_damping  = damping_coefficient_Eta_wall * vz_grid *torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask

        '''
        print(fx_grid_left_damping[fx_grid_left_damping!=0].cpu())
        print(fx_grid_right_damping[fx_grid_right_damping!=0].cpu())
        print(fy_grid_bottom_damping[fy_grid_bottom_damping!=0].cpu())
        print(fy_grid_top_damping[fy_grid_top_damping!=0].cpu())
        print(fz_grid_forward_damping[fz_grid_forward_damping!=0].cpu())
        print(fz_grid_backward_damping[fz_grid_backward_damping!=0].cpu())
        '''
        '''
        # calculate friction force from the boundaries
        fx_grid_left_friction_y      = - torch.where(is_left_overlap,    friction_coefficient * torch.abs (  fx_grid_boundary_left     - fx_grid_left_damping    )   * vy_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fx_grid_left_friction_z      = - torch.where(is_left_overlap,    friction_coefficient * torch.abs (  fx_grid_boundary_left     - fx_grid_left_damping    )   * vz_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))

        fx_grid_right_friction_y     = - torch.where(is_right_overlap,   friction_coefficient * torch.abs (- fx_grid_boundary_right    - fx_grid_right_damping   )   * vy_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fx_grid_right_friction_z     = - torch.where(is_right_overlap,   friction_coefficient * torch.abs (- fx_grid_boundary_right    - fx_grid_right_damping   )   * vz_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))

        fy_grid_bottom_friction_x    = - torch.where(is_bottom_overlap,  friction_coefficient * torch.abs (  fy_grid_boundary_bottom   - fy_grid_bottom_damping  )   * vx_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fy_grid_bottom_friction_z    = - torch.where(is_bottom_overlap,  friction_coefficient * torch.abs (  fy_grid_boundary_bottom   - fy_grid_bottom_damping  )   * vz_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
                
        fy_grid_top_friction_x       = - torch.where(is_top_overlap,     friction_coefficient * torch.abs (- fy_grid_boundary_top      - fy_grid_top_damping     )   * vx_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fy_grid_top_friction_z       = - torch.where(is_top_overlap,     friction_coefficient * torch.abs (- fy_grid_boundary_top      - fy_grid_top_damping     )   * vz_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
    
        fz_grid_forward_friction_x   = - torch.where(is_forward_overlap, friction_coefficient * torch.abs (  fz_grid_boundary_forward  - fz_grid_forward_damping )   * vx_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fz_grid_forward_friction_y   = - torch.where(is_forward_overlap, friction_coefficient * torch.abs (  fz_grid_boundary_forward  - fz_grid_forward_damping )   * vy_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        
        fz_grid_backward_friction_x  = - torch.where(is_backward_overlap,friction_coefficient * torch.abs (- fz_grid_boundary_backward - fz_grid_backward_damping)   * vx_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        fz_grid_backward_friction_y  = - torch.where(is_backward_overlap,friction_coefficient * torch.abs (- fz_grid_boundary_backward - fz_grid_backward_damping)   * vy_grid / torch.maximum(eplis, torch.sqrt(vx_grid**2+vy_grid**2+vz_grid**2)), torch.tensor(0.0, device=device))
        '''
        
        '''
        print(fz_grid_forward_friction_x[fz_grid_forward_friction_x!=0].cpu())
        print(fz_grid_forward_friction_y[fz_grid_forward_friction_y!=0].cpu())
        print(fz_grid_backward_friction_x[fz_grid_backward_friction_x!=0].cpu())
        print(fz_grid_backward_friction_y[fz_grid_backward_friction_y!=0].cpu())
        
        print(fy_grid_top_friction_x[fy_grid_top_friction_x!=0].cpu())
        print(fy_grid_top_friction_z[fy_grid_top_friction_z!=0].cpu())
        print(fy_grid_bottom_friction_x[fy_grid_bottom_friction_x!=0].cpu())
        print(fy_grid_bottom_friction_z[fy_grid_bottom_friction_z!=0].cpu())
        
        print(fx_grid_left_friction_y[fx_grid_left_friction_y!=0].cpu())
        print(fx_grid_left_friction_z[fx_grid_left_friction_z!=0].cpu())
        print(fx_grid_right_friction_y[fx_grid_right_friction_y!=0].cpu())
        print(fx_grid_right_friction_z[fx_grid_right_friction_z!=0].cpu())
        '''
        # calculate the new velocity with acceleration calculated by forces
        vx_grid = vx_grid  +  (dt / particle_mass) * ( - 0   * particle_mass) * mask + (dt / particle_mass) * ( - fx_grid_boundary_right    + fx_grid_boundary_left    - fx_grid_collision - fx_grid_damping - fx_grid_left_damping    - fx_grid_right_damping    ) * mask # + fx_grid_friction + fz_grid_forward_friction_x + fz_grid_backward_friction_x + fy_grid_bottom_friction_x + fy_grid_top_friction_x  ) * mask
        vy_grid = vy_grid  +  (dt / particle_mass) * ( - 0   * particle_mass) * mask + (dt / particle_mass) * ( - fy_grid_boundary_top      + fy_grid_boundary_bottom  - fy_grid_collision - fy_grid_damping - fy_grid_bottom_damping  - fy_grid_top_damping      ) * mask # + fy_grid_friction + fz_grid_forward_friction_y + fz_grid_backward_friction_y + fx_grid_left_friction_y   + fx_grid_right_friction_y) * mask 
        vz_grid = vz_grid  +  (dt / particle_mass) * ( -9.8  * particle_mass) * mask + (dt / particle_mass) * ( - fz_grid_boundary_backward + fz_grid_boundary_forward - fz_grid_collision - fz_grid_damping - fz_grid_forward_damping - fz_grid_backward_damping ) * mask # + fz_grid_friction + fy_grid_bottom_friction_z  + fy_grid_top_friction_z      + fx_grid_left_friction_z   + fx_grid_right_friction_z )* mask 
        
        # del fx_grid_collision, fy_grid_collision, fz_grid_collision, fx_grid_damping, fy_grid_damping, fz_grid_damping         
        del fx_grid_boundary_left, fx_grid_boundary_right, fy_grid_boundary_bottom, fy_grid_boundary_top, fz_grid_boundary_forward, fz_grid_boundary_backward
        del fx_grid_left_damping, fx_grid_right_damping, fy_grid_bottom_damping, fy_grid_top_damping, fz_grid_forward_damping, fz_grid_backward_damping
      
        # print(vx_grid[vx_grid!=0].cpu())
        # print(vy_grid[vy_grid!=0].cpu())
        # print(vz_grid[vz_grid!=0].cpu())
        # print(x_grid[x_grid!=0].cpu())
        # print(y_grid[y_grid!=0].cpu())
        # print(z_grid[z_grid!=0].cpu())
      
        '''
        print(fx_grid_boundary_right[fx_grid_boundary_right!=0].cpu())
        print(fx_grid_boundary_left[fx_grid_boundary_left!=0].cpu())
        print(fy_grid_boundary_bottom[fy_grid_boundary_bottom!=0].cpu())
        print(fy_grid_boundary_top[fy_grid_boundary_top!=0].cpu())
        print(fz_grid_boundary_forward[fz_grid_boundary_forward!=0].cpu())
        print(fz_grid_boundary_backward[fz_grid_boundary_backward!=0].cpu())

        print(fx_grid_left_damping[fx_grid_left_damping!=0].cpu())
        print(fx_grid_right_damping[fx_grid_right_damping!=0].cpu())
        print(fy_grid_bottom_damping[fy_grid_bottom_damping!=0].cpu())
        print(fy_grid_top_damping[fy_grid_top_damping!=0].cpu())
        print(fz_grid_forward_damping[fz_grid_forward_damping!=0].cpu())
        print(fz_grid_backward_damping[fz_grid_backward_damping!=0].cpu())
        
        print(fz_grid_forward_friction_x[fz_grid_forward_friction_x!=0].cpu())
        print(fz_grid_forward_friction_y[fz_grid_forward_friction_y!=0].cpu())
        print(fz_grid_backward_friction_x[fz_grid_backward_friction_x!=0].cpu())
        print(fz_grid_backward_friction_y[fz_grid_backward_friction_y!=0].cpu())
        
        print(fy_grid_top_friction_x[fy_grid_top_friction_x!=0].cpu())
        print(fy_grid_top_friction_z[fy_grid_top_friction_z!=0].cpu())
        print(fy_grid_bottom_friction_x[fy_grid_bottom_friction_x!=0].cpu())
        print(fy_grid_bottom_friction_z[fy_grid_bottom_friction_z!=0].cpu())
        
        print(fx_grid_left_friction_y[fx_grid_left_friction_y!=0].cpu())
        print(fx_grid_left_friction_z[fx_grid_left_friction_z!=0].cpu())
        print(fx_grid_right_friction_y[fx_grid_right_friction_y!=0].cpu())
        print(fx_grid_right_friction_z[fx_grid_right_friction_z!=0].cpu())
        '''
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
        
        # update new values based on new index         
        mask[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = 1
        x_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = x_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        y_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = y_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        z_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = z_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 

        vx_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vx_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
        vy_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vy_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
        vz_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vz_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
        
        return x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, (fx_grid_collision+fx_grid_damping), (fy_grid_collision+fy_grid_damping), (fz_grid_collision+fz_grid_damping)

model = AI4DEM().to(device)
# model = torch.compile(model, mode="reduce-overhead")
# Module 2: Contact detection and force calculation
t = 0
dt = 0.001  # 0.0001
ntime = 20000
# Convert np.array into torch.tensor and transfer it to GPU
filter_size = 5 

# Initialize tensors
# diffx = torch.zeros(input_shape_global, device=device)
# diffy = torch.zeros(input_shape_global, device=device)
# diffz = torch.zeros(input_shape_global, device=device)

# diffvx = torch.zeros(input_shape_global, device=device)
# diffvy = torch.zeros(input_shape_global, device=device)
# diffvz = torch.zeros(input_shape_global, device=device)

zeros = torch.zeros(input_shape_global, device=device)
eplis = torch.ones(input_shape_global, device=device) * 1e-04

# fx_grid_collision = torch.zeros(input_shape_global, device=device)
# fy_grid_collision = torch.zeros(input_shape_global, device=device)
# fz_grid_collision = torch.zeros(input_shape_global, device=device)

# fx_grid_damping = torch.zeros(input_shape_global, device=device)
# fy_grid_damping = torch.zeros(input_shape_global, device=device)
# fz_grid_damping = torch.zeros(input_shape_global, device=device)

mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)

vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)
vz_grid = torch.from_numpy(vz_grid).float().to(device)

# Main simulation loop
start = time.time()
with torch.no_grad():
    for itime in range(1, ntime + 1):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            [x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, Fx, Fy, Fz] = model(x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, cell_size, R_p, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape_global, filter_size)
            print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
            save_path = 'DEM_no_friction_2'
            if itime % 60 == 0:
                np.save(save_path+"/xp"+str(itime), arr=x_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/yp"+str(itime), arr=y_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/zp"+str(itime), arr=z_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/up"+str(itime), arr=vx_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/vp"+str(itime), arr=vy_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/wp"+str(itime), arr=vz_grid.cpu().detach().numpy()[0,0,:,:])     
                np.save(save_path+"/Fx"+str(itime), arr=Fx.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/Fy"+str(itime), arr=Fy.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/Fz"+str(itime), arr=Fz.cpu().detach().numpy()[0,0,:,:])     


            # # Visualize particles
            # xp = x_grid[x_grid!=0].cpu() 
            # yp = y_grid[y_grid!=0].cpu() 
            # zp = z_grid[z_grid!=0].cpu() 

            # fig = plt.figure(figsize=(20, 10))
            # ax = fig.add_subplot(111, projection="3d")
            # sc = ax.scatter(xp,yp,zp, c=vz_grid[vz_grid != 0].cpu(), cmap="turbo", s=S_graph, vmin=-10, vmax=10)
            # cbar = plt.colorbar(sc, orientation='horizontal', shrink=0.35)
            # cbar.set_label('$V_{z}$')
            # ax = plt.gca()

            # ax.set_xlim([0, domain_size-cell_size])
            # ax.set_ylim([0, domain_size-cell_size])
            # ax.set_zlim([0, domain_size-cell_size])
            # ax.view_init(elev=45, azim=45)
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')            
            # ax.set_zlabel('z')

            # print(vx_grid[vx_grid!=0].cpu())

            # # Save visualization
            # if itime < 10:
            #     save_name = "3D_new2/"+str(itime)+".jpg"
            # elif itime >= 10 and itime < 100:
            #     save_name = "3D_new2/"+str(itime)+".jpg"
            # elif itime >= 100 and itime < 1000:
            #     save_name = "3D_new2/"+str(itime)+".jpg"
            # elif itime >= 1000 and itime < 10000:
            #     save_name = "3D_new2/"+str(itime)+".jpg"
            # else:
            #     save_name = "3D_new2/"+str(itime)+".jpg"
            # plt.savefig(save_name, dpi=1000, bbox_inches='tight')
            # plt.close()
            if itime % 250 == 0 and itime < 7500:
                for i in range(1, half_domain_size-1):
                    for j in range(1, half_domain_size-1):
                        x_grid[0, 0, domain_height-3, j*2, i*2] = i * cell_size * 2
                        y_grid[0, 0, domain_height-3, j*2, i*2] = j * cell_size * 2
                        z_grid[0, 0, domain_height-3, j*2, i*2] = (domain_height - 3) * cell_size
                        vz_grid[0, 0, domain_height-3, j*2, i*2] = -5*0.1
                        vx_grid[0, 0, domain_height-3, j*2, i*2] = random.uniform(-1.0,1.0)*0.1
                        vy_grid[0, 0, domain_height-3, j*2, i*2] = random.uniform(-1.0,1.0)*0.1

end = time.time()
print('Elapsed time:', end - start)
