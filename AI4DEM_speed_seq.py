#!/usr/bin/env python

import os
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

# Function to generate a structured grid
def create_grid(l, d):
    num_cells = int(l / (d))
    grid = np.zeros((num_cells, num_cells), dtype=int)
    return grid

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
# Generate particles **************** Random ************************ less particles 
# npt = int(l*l*0.2)
# x = np.random.uniform(l/10, l/10*9, npt)
# y = np.random.uniform(l/10, l/10*9, npt)
# input_shape_global = (1,1,grid_shape[0],grid_shape[1])
# x_grid = np.zeros(input_shape_global)
# y_grid = np.zeros(input_shape_global)
# print('Numeber of elements nodes:',x_grid.shape)
# for i in range(npt):
#   cell_x = int(x[i] / d)
#   cell_y = int(y[i] / d)
#   x_grid[0,0,cell_x,cell_y] = x[i]
#   y_grid[0,0,cell_x,cell_y] = y[i]
# mask = np.where(x_grid != 0, 1, 0)
# print('Number of particles:',np.count_nonzero(mask))
# Generate particles **************** maximum ************************ more particles 
npt = int(l*l)
x = np.arange(2, l)
y = np.arange(2, l)
input_shape_global = (1,1,grid_shape[0],grid_shape[1])
x_grid = np.zeros(input_shape_global)
y_grid = np.zeros(input_shape_global)
print('Numeber of elements nodes:',x_grid.shape)
for i in range(l):
  cell_x = np.round(x / d)
  cell_y = np.round(y / d)
  x_grid[0,0,i,1:l-1] = cell_x
  y_grid[0,0,1:l-1,i] = cell_y
mask = np.where(x_grid != 0, 1, 0)
print('Number of particles:',np.count_nonzero(mask))

# ***************** design filters to detect particel-to-particle interaction ****************
class AI4DEM(nn.Module):
    """docstring for AI4DEM"""
    def __init__(self):
        super(AI4DEM, self).__init__()
        # self.arg = arg

    def detector(self, grid, i, j):
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
        shifts=(0, 1)   !shifts=(0, -2)   !shifts=(-2, -2)  !shifts=(1, -2)
        0 0 0 0 0     !0 0 0 0 0      !0 0 0 0 1      !0 0 0 0 0
        0 0 0 0 0     !0 0 0 0 0      !0 0 0 0 0      !0 0 0 0 0 
        0 1 1 0 0     !0 0 1 0 1      !0 0 1 0 0      !0 0 1 0 0
        0 0 0 0 0     !0 0 0 0 0      !0 0 0 0 0      !0 0 0 0 1
        0 0 0 0 0       !0 0 0 0 0      !0 0 0 0 0      !0 0 0 0 0 
        **************************************************************************
        """

        diff = grid - torch.roll(grid, shifts=(j-2, i-2), dims=(2, 3))  
        return diff

    def forward(self, x_grid, y_grid, vx_grid, vy_grid, fx_grid, fy_grid, mask, d, kn, diffx, diffy, dt, input_shape, filter_size):
        # store previous index tensor 
        cell_xold = x_grid / d
        cell_yold = y_grid / d   
        cell_xold = torch.Tensor.int(cell_xold)
        cell_yold = torch.Tensor.int(cell_yold) 
        # calculate distance between the two particles         
        fx_grid = torch.zeros(input_shape, device=device) 
        fy_grid = torch.zeros(input_shape, device=device)
        for i in range(filter_size):
            for j in range(filter_size):
                diffx = self.detector(x_grid, i, j) # individual
                diffy = self.detector(y_grid, i, j) # individual
                dist = torch.sqrt(diffx**2 + diffy**2)   
                fx_grid = fx_grid + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffx / torch.maximum(eplis, dist), zeros) # individual
                fy_grid = fy_grid + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffy / torch.maximum(eplis, dist), zeros) # individual       
        # Update velocity: Vel(tt+1) = Vel(tt) + F/particle_mass * delta_t
        vx_grid = vx_grid - (dt / particle_mass) * fx_grid * mask
        vy_grid = vy_grid - (dt / particle_mass) * fy_grid * mask 
        # Update particle coordniates
        x_grid = x_grid + dt * vx_grid
        y_grid = y_grid + dt * vy_grid
        # update new index tensor 
        cell_x = x_grid / d 
        cell_y = y_grid / d     
        cell_x = torch.Tensor.int(cell_x)
        cell_y = torch.Tensor.int(cell_y)       
        # extract index (previous and new) from sparse index tensor (previous and new)
        cell_x = cell_x[cell_x!=0]
        cell_y = cell_y[cell_y!=0]         
        cell_xold = cell_xold[cell_xold!=0]
        cell_yold = cell_yold[cell_yold!=0]   
        # get rid of values at previous index 
        mask[0,0,cell_y, cell_x] = 1
        x_grid[0,0,cell_y, cell_x] = x_grid[0,0,cell_yold,cell_xold] 
        y_grid[0,0,cell_y, cell_x] = y_grid[0,0,cell_yold,cell_xold] 
        # update new values based on new index 
        mask[0,0,cell_yold, cell_xold] = 0
        x_grid[0,0,cell_yold,cell_xold] = 0 
        y_grid[0,0,cell_yold,cell_xold] = 0 
        return x_grid, y_grid, mask

model = AI4DEM().to(device)

# Module 2: Contact detection using 5x5 filter, and contact force calculation *************
t = 0
dt = 1e-5 #2*np.pi*np.sqrt(particle_mass/kn)
ntime = 1000 
# print(dt)

# ***************** Convert np.array into torch.tensor and transfer it to GPU ****************
filter_size = 5 
# input_shape_local = (1,filter_size**2,grid_shape[0],grid_shape[1])
input_shape_global = (1,1,grid_shape[0],grid_shape[1])

diffx = torch.zeros(input_shape_global, device=device)
diffy = torch.zeros(input_shape_global, device=device)
zeros = torch.zeros(input_shape_global, device=device)
eplis = torch.ones(input_shape_global, device=device)*1e-04
fx_grid = torch.zeros(input_shape_global, device=device)
fy_grid = torch.zeros(input_shape_global, device=device)
vx_grid = torch.zeros(input_shape_global, device=device)
vy_grid = torch.zeros(input_shape_global, device=device)

mask = torch.reshape(torch.tensor(mask), input_shape_global).to(device)
x_grid = torch.reshape(torch.tensor(x_grid), input_shape_global).to(device)
y_grid = torch.reshape(torch.tensor(y_grid), input_shape_global).to(device)

# # # ################################### # # #
# # # #########  AI4DEM MAIN ############ # # #
# # # ################################### # # #

start = time.time()
with torch.no_grad():
  for itime in range(1,ntime+1):
    [x_grid, y_grid, mask] = model(x_grid, y_grid, vx_grid, vy_grid, fx_grid, fy_grid, mask, d, kn, diffx, diffy, dt, input_shape_global, filter_size)
    print('Time step:',itime)
  end = time.time()
  print('time',(end-start))
