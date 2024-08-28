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

domain_size_x = 14
domain_size_y = 68
domain_size_z = 134

half_domain_size_x = int(domain_size_x/2)+1
half_domain_size_y = int(domain_size_y/2)+1
half_domain_size_z = int(domain_size_z/2)+1

# domain_size = 1500  # Size of the square domain
# domain_height = 100
# half_domain_size = int(domain_size/2)+1
cell_size =0.003   # Cell size and particle radius
simulation_time = 1
kn = 10000#0#000  # Normal stiffness of the spring
dn = 0.5  # Normal damping coefficient
rho_p = 1592 
particle_mass = 4/3*3.1415*cell_size**3*rho_p #4188.7902

print('mass of particle:', particle_mass)

K_graph = 57*100000*1
S_graph = K_graph * (cell_size / domain_size_x) ** 2
restitution_coefficient = 0.5  # coefficient of restitution
friction_coefficient = 0.4  # coefficient of friction
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
input_shape_global = (1, 1, domain_size_z, domain_size_y, domain_size_x)

# Generate particles
# npt = int(domain_size ** 3)

x_grid = np.zeros(input_shape_global)
y_grid = np.zeros(input_shape_global)
z_grid = np.zeros(input_shape_global)

vx_grid = np.zeros(input_shape_global)
vy_grid = np.zeros(input_shape_global)
vz_grid = np.zeros(input_shape_global)
mask = np.zeros(input_shape_global)


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

    def forward(self, x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, d, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape, filter_size):
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
        
        for i in range(filter_size):
            for j in range(filter_size):
                for k in range(filter_size):
                    # calculate distance between the two particles
                    diffx = self.detector(x_grid, i, j, k) # individual
                    diffy = self.detector(y_grid, i, j, k) # individual
                    diffz = self.detector(z_grid, i, j, k) # individual
                    dist = torch.sqrt(diffx**2 + diffy**2 + diffz**2)  

                    # calculate collision force between the two particles
                    fx_grid_collision =  fx_grid_collision + torch.where(torch.lt(dist,2 * d), kn * (dist - 2 * d ) * diffx / torch.maximum(eplis, dist), zeros) # individual
                    fy_grid_collision =  fy_grid_collision + torch.where(torch.lt(dist,2 * d), kn * (dist - 2 * d ) * diffy / torch.maximum(eplis, dist), zeros) # individual
                    fz_grid_collision =  fz_grid_collision + torch.where(torch.lt(dist,2 * d), kn * (dist - 2 * d ) * diffz / torch.maximum(eplis, dist), zeros) # individual            
                                       
                    # calculate nodal velocity difference between the two particles
                    
                    diffvx_Vn =  self.detector(vx_grid, i, j, k) * diffx /  torch.maximum(eplis, dist)
                    diffvy_Vn =  self.detector(vy_grid, i, j, k) * diffy /  torch.maximum(eplis, dist)
                    diffvz_Vn =  self.detector(vz_grid, i, j, k) * diffz /  torch.maximum(eplis, dist) 
                    
                    diffv_Vn = diffvx_Vn + diffvy_Vn + diffvz_Vn
                    
                    
                    # calculate the damping force between the two particles
                    diffv_Vn_x =  diffv_Vn * diffx /  torch.maximum(eplis, dist)
                    diffv_Vn_y =  diffv_Vn * diffy /  torch.maximum(eplis, dist)
                    diffv_Vn_z =  diffv_Vn * diffz /  torch.maximum(eplis, dist)         
                    
                    fx_grid_damping =  fx_grid_damping + torch.where(torch.lt(dist, 2*d), damping_coefficient_Eta * diffv_Vn_x, zeros) # individual   
                    fy_grid_damping =  fy_grid_damping + torch.where(torch.lt(dist, 2*d), damping_coefficient_Eta * diffv_Vn_y, zeros) # individual 
                    fz_grid_damping =  fz_grid_damping + torch.where(torch.lt(dist, 2*d), damping_coefficient_Eta * diffv_Vn_z, zeros) # individual 

        del diffx, diffy, diffz, diffvx_Vn, diffvy_Vn, diffvz_Vn, diffv_Vn, diffv_Vn_x, diffv_Vn_y, diffv_Vn_z

        # judge whether the particle is colliding the boundaries
        is_left_overlap     = torch.ne(x_grid, 0.0000) & torch.lt(x_grid, d) # Overlap with bottom wall
        is_right_overlap    = torch.gt(x_grid, domain_size_x*cell_size-2*d)# Overlap with bottom wall
        is_bottom_overlap   = torch.ne(y_grid, 0.0000) & torch.lt(y_grid, d) # Overlap with bottom wall
        is_top_overlap      = torch.gt(y_grid, domain_size_y*cell_size-2*d ) # Overlap with bottom wall
        is_forward_overlap  = torch.ne(z_grid, 0.0000) & torch.lt(z_grid, 3*d) # Overlap with bottom wall
        is_backward_overlap = torch.gt(z_grid, domain_size_z*cell_size-2*d ) # Overlap with bottom wall             

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

        # calculate the new velocity with acceleration calculated by forces
        vx_grid = vx_grid  +  (dt / particle_mass) * ( - 0   * particle_mass) * mask + (dt / particle_mass) * ( - fx_grid_boundary_right    + fx_grid_boundary_left    - fx_grid_collision - fx_grid_damping - fx_grid_left_damping    - fx_grid_right_damping    ) * mask # + fx_grid_friction + fz_grid_forward_friction_x + fz_grid_backward_friction_x + fy_grid_bottom_friction_x + fy_grid_top_friction_x  ) * mask
        vy_grid = vy_grid  +  (dt / particle_mass) * ( - 0   * particle_mass) * mask + (dt / particle_mass) * ( - fy_grid_boundary_top      + fy_grid_boundary_bottom  - fy_grid_collision - fy_grid_damping - fy_grid_bottom_damping  - fy_grid_top_damping      ) * mask # + fy_grid_friction + fz_grid_forward_friction_y + fz_grid_backward_friction_y + fx_grid_left_friction_y   + fx_grid_right_friction_y) * mask 
        vz_grid = vz_grid  +  (dt / particle_mass) * ( -9.8  * particle_mass) * mask + (dt / particle_mass) * ( - fz_grid_boundary_backward + fz_grid_boundary_forward - fz_grid_collision - fz_grid_damping - fz_grid_forward_damping - fz_grid_backward_damping ) * mask # + fz_grid_friction + fy_grid_bottom_friction_z  + fy_grid_top_friction_z      + fx_grid_left_friction_z   + fx_grid_right_friction_z )* mask 
        
        # del fx_grid_collision, fy_grid_collision, fz_grid_collision, fx_grid_damping, fy_grid_damping, fz_grid_damping         
        del fx_grid_boundary_left, fx_grid_boundary_right, fy_grid_boundary_bottom, fy_grid_boundary_top, fz_grid_boundary_forward, fz_grid_boundary_backward
        del fx_grid_left_damping, fx_grid_right_damping, fy_grid_bottom_damping, fy_grid_top_damping, fz_grid_forward_damping, fz_grid_backward_damping
      

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
        cell_x = x_grid / cell_size 
        cell_y = y_grid / cell_size     
        cell_z = z_grid / cell_size     
                
        cell_x = torch.round(cell_x).long()
        cell_y = torch.round(cell_y).long()    
        cell_z = torch.round(cell_z).long()  
        
        # extract index (previous and new) from sparse index tensor (previous and new)
        cell_x =cell_x[cell_x!=0]
        cell_y =cell_y[cell_y!=0]
        cell_z =cell_z[cell_z!=0]
        
        cell_xold =cell_xold[cell_xold!=0]
        cell_yold =cell_yold[cell_yold!=0]
        cell_zold =cell_zold[cell_zold!=0]
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
        
        x_grid_next = x_grid + dt * vx_grid
        y_grid_next = y_grid + dt * vy_grid
        z_grid_next = z_grid + dt * vz_grid
        is_left_out   = torch.lt(x_grid_next, 0.5*d) # Overlap with bottom wall
        is_right_out  = torch.lt(y_grid_next, 0.5*d) # Overlap with bottom wall
        is_bottom_out = torch.lt(z_grid_next, 0.5*d) # Overlap with bottom wall
        
        combined_condition = is_left_out |is_right_out|is_bottom_out

        x_grid = torch.where(combined_condition, torch.tensor(0, dtype=x_grid.dtype).cuda(), x_grid)
        y_grid = torch.where(combined_condition, torch.tensor(0, dtype=y_grid.dtype).cuda(), y_grid)
        z_grid = torch.where(combined_condition, torch.tensor(0, dtype=z_grid.dtype).cuda(), z_grid)

        return x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, (fx_grid_collision+fx_grid_damping), (fy_grid_collision+fy_grid_damping), (fz_grid_collision+fz_grid_damping)

model = AI4DEM().to(device)
# model = torch.compile(model, mode="reduce-overhead")
# Module 2: Contact detection and force calculation
t = 0
dt = 0.0001  # 0.0001
ntime = 400000
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
            [x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, Fx, Fy, Fz] = model(x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, cell_size, kn, damping_coefficient_Eta, friction_coefficient, dt, input_shape_global, filter_size)
            print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
            if itime % 100 == 0:
                xp = x_grid[x_grid!=0].cpu() 
                yp = y_grid[y_grid!=0].cpu() 
                zp = z_grid[z_grid!=0].cpu() 
    
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(xp,yp,zp)
                cbar = plt.colorbar(sc, orientation='horizontal', shrink=0.35)
                cbar.set_label('$AngularV_{x}$')
                ax = plt.gca()
    
                ax.set_xlim([0, domain_size_x*cell_size])
                ax.set_ylim([0, domain_size_y*cell_size])
                ax.set_zlim([0, domain_size_z*cell_size])
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
            if itime % 600 == 0:
                save_name = "hopper/"+str(itime)
                torch.save(x_grid, save_name + 'x_grid.pt')
                torch.save(y_grid, save_name + 'y_grid.pt')
                torch.save(z_grid, save_name + 'z_grid.pt')
                torch.save(vx_grid, save_name + 'vx_grid.pt')
                torch.save(vy_grid, save_name + 'vy_grid.pt')
                torch.save(vz_grid, 'vz_grid.pt')
                torch.save(torch.count_nonzero(mask), save_name + 'Number_of_particles.pt')
            if itime == 1:
                for i in range(1, half_domain_size_x-1):
                    for j in range(1, half_domain_size_y-1):
                        for k in range(1,  half_domain_size_z-1):
                            x_grid[0, 0, k*2, j*2, i*2] = i * cell_size * 2
                            y_grid[0, 0, k*2, j*2, i*2] = j * cell_size * 2
                            z_grid[0, 0, k*2, j*2, i*2] = k * cell_size * 2
                            
                            vx_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.001
                            vy_grid[0, 0, k*2, j*2, i*2] = random.uniform(-1.0,1.0)*0.001
                            vz_grid[0, 0, k*2, j*2, i*2] = -0.0003

     

end = time.time()
print('Elapsed time:', end - start)
