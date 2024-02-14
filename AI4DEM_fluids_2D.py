import os
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import gstools as gs
import random

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
dt = 0.01
dx = 1
dy = 1
Re = 1
ub = 1
nx = 128
ny = 128
lx = dx * nx ; ly = dy * ny 
nlevel = int(math.log(ny, 2)) + 1 
print(nlevel)
# # # ################################### # # #
# # # ######    Linear Filter   ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0],dtype=torch.double)
w1 = torch.tensor([[[[1/3*Re/dx**2], 
         [1/3*Re/dx**2],
         [1/3*Re/dx**2]],

        [[1/3*Re/dx**2],
         [-8/3*Re/dx**2],
         [1/3*Re/dx**2]],

        [[1/3*Re/dx**2],
         [1/3*Re/dx**2],
         [1/3*Re/dx**2]]]],dtype=torch.double)

w2 = torch.tensor([[[[1/(12*dx)],  # Central differencing for y-advection and second-order time scheme
         [0.0],
         [-1/(12*dx)]],

        [[1/(3*dx)],
         [0.0],
         [-1/(3*dx)]],

        [[1/(12*dx)],
         [0.0],
         [-1/(12*dx)]]]],dtype=torch.double)

w3 = torch.tensor([[[[-1/(12*dx)],  # Central differencing for y-advection and second-order time scheme
         [-1/(3*dx)],
         [-1/(12*dx)]],

        [[0.0],
         [0.0],
         [0.0]],

        [[1/(12*dx)],
         [1/(3*dx)],
         [1/(12*dx)]]]],dtype=torch.double)

wA = torch.tensor([[[[-1/3/dx**2],        # A matrix for Jacobi
         [-1/3/dx**2],
         [-1/3/dx**2]],

        [[-1/3/dx**2],
         [8/3/dx**2],
         [-1/3/dx**2]],

        [[-1/3/dx**2],
         [-1/3/dx**2],
         [-1/3/dx**2]]]],dtype=torch.double)

w1 = torch.reshape(w1, (1,1,3,3))
w2 = torch.reshape(w2, (1,1,3,3))
w3 = torch.reshape(w3, (1,1,3,3))
wA = torch.reshape(wA, (1,1,3,3))
w_res = torch.zeros([1,1,2,2],dtype=torch.double)
w_res[0,0,:,:] = 0.25

wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
        [[0.11] ,  [0.44], [0.11]],
        [[0.028], [0.11] , [0.028]]]],dtype=torch.double)
wm = torch.reshape(wm,(1,1,3,3))


# # # ################################### # # #
# # # ############   AI4CFD   ########### # # #
# # # ################################### # # #
class AI4CFD(nn.Module):
    """docstring for AI4CFD"""
    def __init__(self):
        super(AI4CFD, self).__init__()
        # self.arg = arg
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.A = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.diff.weight.data = w1
        self.cmm.weight.data = wm
        self.A.weight.data = wA
        self.res.weight.data = w_res

        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer
        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
        
    def boundary_condition_u(self, values_v, values_vv):                   
        ny = values_v.shape[2]
        nx = values_v.shape[3]
        nny = values_vv.shape[2]
        nnx = values_vv.shape[3]

        values_vv[0,0,1:nny-1,1:nnx-1] = values_v[0,0,:,:]
        values_vv[0,0,:,0].fill_(0.0)
        values_vv[0,0,:,nx+1].fill_(0.0)
        values_vv[0,0,0,:].fill_(0.0)
        values_vv[0,0,ny+1,:].fill_(0.0)
        return values_vv
    
    def boundary_condition_v(self, values_v, values_vv):                   
        ny = values_v.shape[2]
        nx = values_v.shape[3]
        nny = values_vv.shape[2]
        nnx = values_vv.shape[3]

        values_vv[0,0,1:nny-1,1:nnx-1] = values_v[0,0,:,:]
        values_vv[0,0,:,0].fill_(0.0)
        values_vv[0,0,:,nx+1].fill_(0.0)
        values_vv[0,0,0,:].fill_(0.0)
        values_vv[0,0,ny+1,:].fill_(0.0)
        return values_vv

    def boundary_condition_p(self, values_p, values_pp):                   
        ny = values_p.shape[2]
        nx = values_p.shape[3]
        nny = values_pp.shape[2]
        nnx = values_pp.shape[3]
        
        values_pp[0,0,1:nny-1,1:nnx-1] = values_p[0,0,:,:]
        values_pp[0,0,:,0] =  values_pp[0,0,:,1] 
        values_pp[0,0,:,nx+1] = values_pp[0,0,:,nx]
        values_pp[0,0,0,:] = values_pp[0,0,1,:]
        values_pp[0,0,ny+1,:] = values_pp[0,0,ny,:]
        return values_pp

    def boundary_condition_k(self, k_u, k_uu):                            
        ny = k_u.shape[2]
        nx = k_u.shape[3]
        nny = k_uu.shape[2]
        nnx = k_uu.shape[3]

        k_uu[0,0,1:nny-1,1:nnx-1] = k_u[0,0,:,:]
        k_uu[0,0,:,0] =  k_uu[0,0,:,1]*0 
        k_uu[0,0,:,nx+1] = k_uu[0,0,:,nx]*0 
        k_uu[0,0,0,:] = k_uu[0,0,1,:]*0
        k_uu[0,0,ny+1,:] = k_uu[0,0,ny,:]*0
        return k_uu

    def boundary_condition_cw(self, w):                                    
        ny = w.shape[2]
        nx = w.shape[3]
        ww = F.pad(w, (1, 1, 1, 1), mode='constant', value=0)
    
        ww[0,0,:,0] =  ww[0,0,:,1]*0 
        ww[0,0,:,nx+1] = ww[0,0,:,nx]*0 
        ww[0,0,0,:] = ww[0,0,1,:]*0
        ww[0,0,ny+1,:] = ww[0,0,ny,:]*0
        return ww
        
    def PG_vector(self, values_u, values_v, k1, k_uu, k_vv, ADx_u, ADy_u, ADx_v, ADy_v, AD2_u, AD2_v):
        k_u = 0.25 * dx * torch.abs(1/2 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy) * AD2_u) / \
            (1e-03 + (torch.abs(ADx_u) * dx**-3 + torch.abs(ADy_u) * dx**-3) / 2)

        k_v = 0.25 * dy * torch.abs(1/2 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy) * AD2_v) / \
            (1e-03 + (torch.abs(ADx_v) * dx**-3 + torch.abs(ADy_v) * dx**-3) / 2)

        k_u = torch.minimum(k_u, k1) 
        k_v = torch.minimum(k_v, k1) 
        k_uu = self.boundary_condition_k(k_u,k_uu)    
        k_vv = self.boundary_condition_k(k_v,k_vv)    

        k_x = 3*0.5 * (k_u * AD2_u + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 3*0.5 * (k_v * AD2_v + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        return k_x, k_y

    def F_cycle_MG(self, values_uu, values_vv, values_p, values_pp, iteration, diag, dt, nlevel):
        b = -(self.xadv(values_uu) + self.yadv(values_vv)) / dt
        for MG in range(iteration):
            w = torch.zeros((1,1,2*1,2*1), device=device).double()
            r = self.A(self.boundary_condition_p(values_p, values_pp)) - b 
            r_s = []  
            r_s.append(r)
            for i in range(1,nlevel-1):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel-1)):
                ww = self.boundary_condition_cw(w)
                w = w - self.A(ww) / diag + r_s[i] / diag
                w = self.prol(w)         
            values_p = values_p - w
            values_p = values_p - self.A(self.boundary_condition_p(values_p, values_pp)) / diag + b / diag
        return values_p, w, r

    def forward(self, values_u, values_uu, values_v, values_vv, values_p, values_pp, b_uu, b_vv, k1, dt, iteration, k_uu, k_vv, Fx, Fy):        
        values_uu = self.boundary_condition_u(values_u,values_uu) 
        values_vv = self.boundary_condition_v(values_v,values_vv)  
        values_pp = self.boundary_condition_p(values_p,values_pp)   
        Grapx_p = self.xadv(values_pp) * dt ; Grapy_p = self.yadv(values_pp) * dt 
        ADx_u = self.xadv(values_uu) ; ADy_u = self.yadv(values_uu) 
        ADx_v = self.xadv(values_vv) ; ADy_v = self.yadv(values_vv) 
        AD2_u = self.diff(values_uu) ; AD2_v = self.diff(values_vv) 
    # First step for solving uvw
        [k_x,k_y] = self.PG_vector(values_u, values_v, k1, k_uu, k_vv, ADx_u, ADy_u, ADx_v, ADy_v, AD2_u, AD2_v)  

        b_u = values_u + 0.5 * (Re * k_x * dt - values_u * ADx_u * dt - values_v * ADy_u * dt) - Grapx_p - Fx * dt*0
        b_v = values_v + 0.5 * (Re * k_y * dt - values_u * ADx_v * dt - values_v * ADy_v * dt) - Grapy_p - Fy * dt*0
    # Padding velocity vectors 
        b_uu = self.boundary_condition_u(b_u,b_uu) 
        b_vv = self.boundary_condition_v(b_v,b_vv) 

        ADx_u = self.xadv(b_uu) ; ADy_u = self.yadv(b_uu) 
        ADx_v = self.xadv(b_vv) ; ADy_v = self.yadv(b_vv) 
        AD2_u = self.diff(b_uu) ; AD2_v = self.diff(b_vv) 

        [k_x,k_y] = self.PG_vector(b_u, b_v, k1, k_uu, k_vv, ADx_u, ADy_u, ADx_v, ADy_v, AD2_u, AD2_v)   
    # Second step for solving uvw   
        values_u = values_u + Re * k_x * dt - b_u * ADx_u * dt - b_v * ADy_u * dt - Grapx_p - Fx * dt 
        values_v = values_v + Re * k_y * dt - b_u * ADx_v * dt - b_v * ADy_v * dt - Grapy_p - Fy * dt
    # pressure
        values_uu = self.boundary_condition_u(values_u,values_uu) 
        values_vv = self.boundary_condition_v(values_v,values_vv)  
        [values_p, w ,r] = self.F_cycle_MG(values_uu, values_vv, values_p, values_pp, iteration, diag, dt, nlevel)
    # Pressure gradient correction    
        values_pp = self.boundary_condition_p(values_p, values_pp)  
        values_u = values_u - self.xadv(values_pp) * dt ; values_v = values_v - self.yadv(values_pp) * dt 
        return values_u, values_v, values_p, w, r

# # # ################################### # # #
# # # ############   AI4DEM   ########### # # #
# # # ################################### # # #
class AI4DEM(nn.Module):
    """docstring for AI4DEM"""
    def __init__(self):
        super(AI4DEM, self).__init__()
        # self.arg = arg
        self.cmm = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm.weight.data = wm
        self.cmm.bias.data = bias_initializer

    def coupling_forward(self, xp_grid, yp_grid, x_grid, y_grid, alpha, d, input_shape):
        """
        Interpolating velocities at the nodes to the velocity at the centre of a particle 
        x_grid      : x coordinate in Eulerian nodes -> Fluids (1,1,nz,ny,nx)
        vx_grid     : x velocity in Eulerian nodes -> Fluids
        xp_grid     : x coordinate in Lagrangian nodes -> Particles 
        d           : grid spacing (assume it as uniform)
        dims(2,3,4) : 4 --> x axis (nx)     ; 3 --> y axis (ny)     ; 2 --> z axis (nz)
        rolling     : 2 --> filter size = 5 ; 1 --> filter size = 3
        **************************************************************************
        """
        alpha_p = torch.zeros(input_shape, device=device) 
        for i in range(3):
            for j in range(3):
                xii = (xp_grid - x_grid)/d
                phi = (yp_grid - y_grid)/d 
                if i == 0:
                    Q_xii = -0.5 * xii * (1 - xii)
                elif i == 1:
                    Q_xii = 1 - xii**2
                else:
                    Q_xii = 0.5 * xii * (1 - xii)
                if j == 0:
                    Q_phi = -0.5 * phi * (1 - phi)
                elif j == 1:
                    Q_phi = 1 - phi**2
                else:
                    Q_phi = 0.5 * phi * (1 - phi)
                alpha_p = alpha_p + Q_xii * Q_phi * torch.roll(alpha, shifts=(j-1, i-1), dims=(2, 3))
        return alpha_p

    def coupling_backward(self, xp_grid, yp_grid, x_grid, y_grid, vx_grid, vy_grid, Fx_grid, Fy_grid, mask, d, input_shape, one, zero):
        """
        Interpolating volume fractions and velocities of the particle to the nodes 
        x_grid      : x coordinate in Eulerian nodes -> Fluids (1,1,nz,ny,nx)
        vx_grid     : x velocity in Eulerian nodes -> Fluids
        xp_grid     : x coordinate in Lagrangian nodes -> Particles 
        d           : grid spacing (assume it as uniform)
        dims(2,3,4) : 4 --> x axis (nx)     ; 3 --> y axis (ny)     ; 2 --> z axis (nz)
        rolling     : 2 --> filter size = 5 ; 1 --> filter size = 3
        **************************************************************************
        """

        alpha = torch.zeros(input_shape, device=device) 
        alpha_u = torch.zeros(input_shape, device=device) 
        alpha_v = torch.zeros(input_shape, device=device) 
        Fx = torch.zeros(input_shape, device=device) 
        Fy = torch.zeros(input_shape, device=device) 
        for i in range(3):
            for j in range(3):
                sgn_x = torch.roll(xp_grid, shifts=(j-1, i-1), dims=(2, 3)) - x_grid 
                sgn_y = torch.roll(yp_grid, shifts=(j-1, i-1), dims=(2, 3)) - y_grid

                Sx = torch.where(torch.ge(sgn_x,0), one, one*-1)
                Sy = torch.where(torch.ge(sgn_y,0), one, one*-1)

                xii = (2 * sgn_x - Sx * dx) / dx    
                phi = (2 * sgn_y - Sy * dy) / dy

                N_xii = 0.5 * (1 - Sx * xii) * torch.where(torch.logical_and(torch.ge(xii,-1), torch.le(xii,1)), one, zero)
                N_phi = 0.5 * (1 - Sy * phi) * torch.where(torch.logical_and(torch.ge(phi,-1), torch.le(phi,1)), one, zero)

                alpha = alpha + N_xii * N_phi * mask * Vp / (dx*dy)
                alpha_u = alpha_u + N_xii * N_phi * mask * Vp * torch.roll(vx_grid, shifts=(j-1, i-1), dims=(2, 3)) / (dx*dy)
                alpha_v = alpha_v + N_xii * N_phi * mask * Vp * torch.roll(vy_grid, shifts=(j-1, i-1), dims=(2, 3)) / (dx*dy)
                Fx = Fx + N_xii * N_phi * mask * Vp * torch.roll(Fx_grid, shifts=(j-1, i-1), dims=(2, 3)) / (dx*dy)
                Fy = Fy + N_xii * N_phi * mask * Vp * torch.roll(Fy_grid, shifts=(j-1, i-1), dims=(2, 3)) / (dx*dy)
        return alpha, alpha_u, alpha_v, Fx, Fy

    def cal_force(self, vx_grid, vy_grid, values_u, values_v):
        """
        Calculating forces for particle (drag, gravity, contacting force, etc )
        values_u, values_v : Two velocities in Eul nodes -> Fluids 	  (1,1,nz,ny,nx)
        vx_grid, vy_grid   : Two velocities in Lag nodes -> particles (1,1,nz,ny,nx)
        **************************************************************************
        Drag force               : F_{D} = (0.5 * rho * A * |u_{slip}| * u_{slip} * C_{D}) / m_{p}  [m/s^{2}]
        Gravity force (bouyancy) : F_{b} = g          												[m/s^{2}]
        ***************** Adding contacting force ********************************



        ***************** Adding contacting force ********************************

        """    	
        Fx_grid = -1 * ((vx_grid - values_u)**2 + (vy_grid - values_v)**2)**0.5 * (vx_grid - values_u) * 0.44 * (3/2)
        Fy_grid = -1 * ((vx_grid - values_u)**2 + (vy_grid - values_v)**2)**0.5 * (vy_grid - values_v) * 0.44 * (3/2) - 9.81
        vx_grid = vx_grid + Fx_grid * dt
        vy_grid = vy_grid + Fy_grid * dt
        return vx_grid * mask, vy_grid * mask, Fx_grid * mask, Fy_grid * mask

    def forward(self, x_grid, y_grid, vx_grid, vy_grid, values_u, values_v, alpha, mask, time_rel, source, source_no, d, input_shape, filter_size, one, half, zero):        
        # store previous Global index tensor
        cell_xold = x_grid / d ; cell_xold = torch.round(cell_xold).long()
        cell_yold = y_grid / d ; cell_yold = torch.round(cell_yold).long()  
        # [vx_grid, vy_grid, alpha_p] = self.coupling_forward(x_grid, y_grid, X, Y, values_u, values_v, alpha, d, input_shape)
        [vx_grid, vy_grid, Fx_grid, Fy_grid] = self.cal_force(vx_grid, vy_grid, values_u, values_v)
        # update particle coordinates
        x_grid = x_grid + dt * vx_grid 
        y_grid = y_grid + dt * vy_grid 
        # alpha_p = alpha_p * mask 
        # Merge particles
        x_grid_merge = x_grid
        y_grid_merge = y_grid
        vx_grid_merge = vx_grid
        vy_grid_merge = vy_grid
        Fx_grid_merge = Fx_grid
        Fy_grid_merge = Fy_grid
        # check if particle is outside domain
        x_bc = torch.logical_or(torch.ge(x_grid_merge,nx-5), torch.le(x_grid_merge,0+5))
        y_bc = torch.logical_or(torch.ge(y_grid_merge,nx-5), torch.le(y_grid_merge,0+5))
        xy_bc = torch.logical_or(x_bc,y_bc)
        # remove the particles that are outside domain        
        x_grid_merge = x_grid_merge * torch.where(xy_bc, zero, one)
        y_grid_merge = y_grid_merge * torch.where(xy_bc, zero, one)
        vx_grid_merge = vx_grid_merge * torch.where(xy_bc, zero, one)
        vy_grid_merge = vy_grid_merge * torch.where(xy_bc, zero, one)
        Fx_grid_merge = Fx_grid_merge * torch.where(xy_bc, zero, one)
        Fy_grid_merge = Fy_grid_merge * torch.where(xy_bc, zero, one)
        # update new global index tensor (sparse)
        cell_x = x_grid_merge / d ; cell_x = torch.round(cell_x).long()
        cell_y = y_grid_merge / d ; cell_y = torch.round(cell_y).long()   
        # remove the index about particles that are outside domain from previous index tensor (sparse)
        cell_xxold = torch.where(xy_bc, zero, cell_xold) ; cell_xxold = torch.round(cell_xxold).long()
        cell_yyold = torch.where(xy_bc, zero, cell_yold) ; cell_yyold = torch.round(cell_yyold).long()
        # extract new index from sparse index tensor (global)
        cell_xxold = cell_xxold[cell_xxold!=0]
        cell_yyold = cell_yyold[cell_yyold!=0]            
        cell_xold = cell_xold[cell_xold!=0]
        cell_yold = cell_yold[cell_yold!=0]   
        cell_x = cell_x[cell_x!=0]
        cell_y = cell_y[cell_y!=0]  
        # print(cell_x.shape,cell_y.shape)
        if (cell_x.shape != cell_y.shape):
            print('wrong')
            # break
        # get rid of values at previous index (global)
        mask[0,0,cell_yold,cell_xold] = 0
        x_grid[0,0,cell_yold,cell_xold] = 0 
        y_grid[0,0,cell_yold,cell_xold] = 0 
        vx_grid[0,0,cell_yold,cell_xold] = 0 
        vy_grid[0,0,cell_yold,cell_xold] = 0     
        Fx_grid[0,0,cell_yold,cell_xold] = 0 
        Fy_grid[0,0,cell_yold,cell_xold] = 0  
        # update new values based on new index (global)
        mask[0,0,cell_y,cell_x] = 1
        x_grid[0,0,cell_y,cell_x] = x_grid_merge[0,0,cell_yyold,cell_xxold] 
        y_grid[0,0,cell_y,cell_x] = y_grid_merge[0,0,cell_yyold,cell_xxold]    
        vx_grid[0,0,cell_y,cell_x] = vx_grid_merge[0,0,cell_yyold,cell_xxold] 
        vy_grid[0,0,cell_y,cell_x] = vy_grid_merge[0,0,cell_yyold,cell_xxold] 
        Fx_grid[0,0,cell_y,cell_x] = Fx_grid_merge[0,0,cell_yyold,cell_xxold] 
        Fy_grid[0,0,cell_y,cell_x] = Fy_grid_merge[0,0,cell_yyold,cell_xxold]  
        # time_rel[0,0,cell_yold,cell_xold] = 0
        [alpha, alpha_u, alpha_v, Fx, Fy] = self.coupling_backward(x_grid, y_grid, X, Y, vx_grid, vy_grid, Fx_grid, Fy_grid, mask, d, input_shape, one, zero)

        return x_grid, y_grid, vx_grid, vy_grid, Fx_grid, Fy_grid, mask, alpha, alpha_u, alpha_v, Fx, Fy

Lag = AI4DEM().to(device)
Eul = AI4CFD().to(device)

################# Release particles ################
def release_pt(x_grid, y_grid, mask, xp_s, yp_s):
    x_idx = torch.round(xp_s/d).to(torch.int)
    y_idx = torch.round(yp_s/d).to(torch.int)
    x_grid[0, 0, y_idx, x_idx] = xp_s.to(x_grid.dtype)
    y_grid[0, 0, y_idx, x_idx] = yp_s.to(y_grid.dtype)
    mask[0, 0, y_idx, x_idx] = 1
    return x_grid, y_grid, mask
################# Fluids parameters ################
ntime = 10000                       # Time steps
n_out = 1000                        # Results output
iteration = 5                       # Multigrid iteration
nrestart = 0                        # Last time step for restart
eplsion_k = 1e-04                   # Stablisatin factor in Petrov-Galerkin for velocity
diag = np.array(wA)[0,0,1,1]        # Diagonal component
################# particle parameters ################
d = 1
Vp = 3.1415 / 6 * 1**3
filter_size = 3
source_no = 9
xp_s = torch.tensor([0.2*nx,0.2*nx,0.2*nx,0.5*nx,0.5*nx,0.5*nx,0.8*nx,0.8*nx,0.8*nx]).to(device)
yp_s = torch.tensor([0.2*nx,0.5*nx,0.8*nx,0.2*nx,0.5*nx,0.8*nx,0.2*nx,0.5*nx,0.8*nx]).to(device)
# # # ################################### # # #
# # # ###  Create tensor in Eulerian  ### # # #
# # # ################################### # # #
input_shape = (1,1,ny,nx)
values_u = torch.zeros(input_shape, device=device).double()
values_v = torch.zeros(input_shape, device=device).double()
values_p = torch.zeros(input_shape, device=device).double()
alpha = torch.zeros(input_shape, device=device).double()
k1 = (torch.ones(input_shape, device=device)*1.0*dx**2/dt).double()
input_shape_pad = (1,1,ny+2,nx+2)
values_uu = torch.zeros(input_shape_pad, device=device).double()
values_vv = torch.zeros(input_shape_pad, device=device).double()
values_pp = torch.zeros(input_shape_pad, device=device).double()
b_uu = torch.zeros(input_shape_pad, device=device).double()
b_vv = torch.zeros(input_shape_pad, device=device).double()
k_uu = torch.zeros(input_shape_pad, device=device).double()
k_vv = torch.zeros(input_shape_pad, device=device).double()
# # # ################################### # # #
# # # ###  Create tensor in Eulerian  ### # # #
# # # ################################### # # #
# storing position of particles 
mask = torch.zeros(input_shape, device=device)
x_grid = torch.zeros(input_shape, device=device)
y_grid = torch.zeros(input_shape, device=device)
vx_grid = torch.zeros(input_shape, device=device)
vy_grid = torch.zeros(input_shape, device=device)
# storing time since released 
time_rel = torch.zeros(input_shape, device=device)
# storing position of the source 
source = torch.zeros(input_shape, device=device)
alpha = torch.zeros(input_shape, device=device)
# limitation tensors 
one = torch.ones(input_shape, device=device)
half = torch.ones(input_shape, device=device)*0.5
zero = torch.zeros(input_shape, device=device)
values_u = torch.zeros(input_shape, device=device)
values_v = torch.zeros(input_shape, device=device)

y, x = torch.meshgrid(torch.arange(0, ly, dx), torch.arange(0, lx, dx))
X = torch.reshape(x, input_shape).to(device)
Y = torch.reshape(y, input_shape).to(device)

# # # ################################### # # #
# # # ###########  Main loop  ########### # # #
# # # ################################### # # #
with torch.no_grad():
    for itime in range(1,1000+1):
        if itime ==1 or itime % 100 == 0:
            [x_grid, y_grid, mask] = release_pt(x_grid, y_grid, mask, xp_s, yp_s)

        [x_grid,y_grid,vx_grid,vy_grid,Fx_grid,Fy_grid,mask,alpha,alpha_u,alpha_v,Fx,Fy] = Lag(x_grid, y_grid, vx_grid, vy_grid,
                                                                                   values_u, values_v, alpha, mask, time_rel,
                                                                                   source, source_no, d, input_shape, filter_size, one, half, zero)
        [values_u,values_v,values_p,w,r] = Eul(values_u, values_uu, values_v, values_vv, values_p, values_pp, b_uu, b_vv, k1, dt, iteration, k_uu, k_vv, Fx, Fy)

        print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
        print('Pressure error:', np.max(np.abs(w.cpu().detach().numpy())), 'cty equation residual:', np.max(np.abs(r.cpu().detach().numpy())))

        if itime % 10 == 0:
            plt.figure(figsize=(16, 4))
            plt.subplot(131)
            yp, xp = torch.where(mask.cpu()[0,0,:,:] == 1)
            plt.scatter(xp, yp, c=vy_grid.cpu()[0,0,yp,xp], cmap='turbo', s=20, vmin=-4,vmax=-3)
            cbar = plt.colorbar()
            cbar.set_label('$V_{p}$')
            ax = plt.gca()
            # ax.set_aspect('equal')
            ax.set_xlim([0, nx])
            ax.set_ylim([0, ny])
            # plt.grid(which='both')
            # plt.minorticks_on()
            plt.subplot(132)
            plt.imshow((values_v.cpu())[0,0,:,:],cmap='RdBu',vmin=-0.05,vmax=0.25)
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.title('nodal velocity in y')
            plt.subplot(133)
            plt.imshow((values_u.cpu())[0,0,:,:],cmap='RdBu',vmin=-0.05,vmax=0.05)
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.title('nodal velocity in x')

            if itime < 10:
                save_name = "Force_model/00"+str(itime)+".jpg"
            elif itime >= 10 and itime < 100:
                save_name = "Force_model/0"+str(itime)+".jpg"
            else:
                save_name = "Force_model/"+str(itime)+".jpg"
            plt.savefig(save_name, dpi=200, bbox_inches='tight')
            plt.close() 