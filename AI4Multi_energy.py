#!/usr/bin/env python

#  Copyright (C) 2023
#  
#  Boyang Chen, Claire Heaney, Christopher Pain
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#
#  boyang.chen16@imperial.ac.uk
#  
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation,
#  version 3.0 of the License.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.

#-- Import general libraries
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' ## enable xla devices # Comment out this line if runing on GPU cluster
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
dt = 0.002
dx = 0.04
dy = 0.04
dz = 0.04
nx = 512
ny = 64
nz = 64
ratio = int(max(nx, ny, nz) / min(nx, ny, nz))
nlevel = int(math.log(nz, 2)) + 1 
print('How many levels in multigrid:', nlevel)
print('Aspect ratio:', ratio)
# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0])
# Laplacian filters
pd1 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
pd2 = torch.tensor([[3/26, 6/26, 3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26, 3/26]])
pd3 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
w1 = torch.zeros([1, 1, 3, 3, 3])
wA = torch.zeros([1, 1, 3, 3, 3])
w1[0, 0, 0,:,:] = pd1/dx**2
w1[0, 0, 1,:,:] = pd2/dx**2
w1[0, 0, 2,:,:] = pd3/dx**2
wA[0, 0, 0,:,:] = -pd1/dx**2
wA[0, 0, 1,:,:] = -pd2/dx**2
wA[0, 0, 2,:,:] = -pd3/dx**2
# Gradient filters
p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056],
       [-0.22, 0.0, 0.22],
       [-0.056, 0.0, 0.056]])
p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_y1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_y2 = torch.tensor([[0.056, 0.22, 0.056],
       [0.0, 0.0, 0.0],
       [-0.056, -0.22, -0.056]])
p_div_y3 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.056, 0.22, 0.056],
       [0.014, 0.056, 0.014]])
p_div_z2 = torch.tensor([[0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0]])
p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014],
       [-0.056, -0.22, -0.056],
       [-0.014, -0.056, -0.014]])
w2 = torch.zeros([1,1,3,3,3])
w3 = torch.zeros([1,1,3,3,3])
w4 = torch.zeros([1,1,3,3,3])
w2[0,0,0,:,:] = -p_div_x1/dx
w2[0,0,1,:,:] = -p_div_x2/dx
w2[0,0,2,:,:] = -p_div_x3/dx
w3[0,0,0,:,:] = -p_div_y1/dx
w3[0,0,1,:,:] = -p_div_y2/dx
w3[0,0,2,:,:] = -p_div_y3/dx
w4[0,0,0,:,:] = -p_div_z1/dx 
w4[0,0,1,:,:] = -p_div_z2/dx
w4[0,0,2,:,:] = -p_div_z3/dx
# Curvature Laplacian filters
curvature_x1 = torch.tensor([[-0.1875, 0.375,  -0.1875],
       [-0.75, 1.5,  -0.75],
       [-0.1875, 0.375,  -0.1875]])
curvature_x2= torch.tensor([[-0.75, 1.5,  -0.75],
       [-3.0, 6.0,  -3.0],
       [-0.75, 1.5,  -0.75]])
curvature_x3 = torch.tensor([[-0.1875, 0.375,  -0.1875],
       [-0.75, 1.5,  -0.75],
       [-0.1875, 0.375,  -0.1875]])
curvature_y1 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [0.375, 1.5,  0.375],
       [-0.1875, -0.75,  -0.1875]])
curvature_y2= torch.tensor([[-0.75, -3.0,  -0.75],
       [1.5, 6.0,  1.5],
       [-0.75, -3.0,  -0.75]])
curvature_y3 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [0.375, 1.5,  0.375],
       [-0.1875, -0.75,  -0.1875]])
curvature_z1 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [-0.75, -3.0,  -0.75],
       [-0.1875, -0.75,  -0.1875]])
curvature_z2= torch.tensor([[0.375, 1.5,  0.375],
       [1.5, 6.0,  1.5],
       [0.375, 1.5,  0.375]])
curvature_z3 = torch.tensor([[-0.1875, -0.75,  -0.1875],
       [-0.75, -3.0,  -0.75],
       [-0.1875, -0.75,  -0.1875]])
AD2_x = torch.zeros([1,1,3,3,3])
AD2_y = torch.zeros([1,1,3,3,3])
AD2_z = torch.zeros([1,1,3,3,3])
AD2_x[0,0,0,:,:] = -curvature_x1/dx**2
AD2_x[0,0,1,:,:] = -curvature_x2/dx**2
AD2_x[0,0,2,:,:] = -curvature_x3/dx**2
AD2_y[0,0,0,:,:] = -curvature_y1/dx**2
AD2_y[0,0,1,:,:] = -curvature_y2/dx**2
AD2_y[0,0,2,:,:] = -curvature_y3/dx**2
AD2_z[0,0,0,:,:] = -curvature_z1/dx**2
AD2_z[0,0,1,:,:] = -curvature_z2/dx**2
AD2_z[0,0,2,:,:] = -curvature_z3/dx**2
# Restriction filters
w_res = torch.zeros([1,1,2,2,2])
w_res[0,0,:,:,:] = 0.125
# Detecting filters
wxu = torch.zeros([1,1,3,3,3])
wxd = torch.zeros([1,1,3,3,3])
wyu = torch.zeros([1,1,3,3,3])
wyd = torch.zeros([1,1,3,3,3])
wzu = torch.zeros([1,1,3,3,3])
wzd = torch.zeros([1,1,3,3,3])
wxu[0,0,1,1,1] = 1.0
wxu[0,0,1,1,0] = -1.0
wxd[0,0,1,1,1] = -1.0
wxd[0,0,1,1,2] = 1.0
wyu[0,0,1,1,1] = 1.0
wyu[0,0,1,0,1] = -1.0
wyd[0,0,1,1,1] = -1.0
wyd[0,0,1,2,1] = 1.0
wzu[0,0,1,1,1] = 1.0
wzu[0,0,0,1,1] = -1.0
wzd[0,0,1,1,1] = -1.0
wzd[0,0,2,1,1] = 1.0
################# Numerical parameters ################
ntime = 20000                     # Time steps
n_out = 500                       # Results output
iteration = 10                    # Multigrid iteration
nrestart = 0                      # Last time step for restart
ctime_old = 0                     # Last ctime for restart
LSCALAR = True                    # Scalar transport 
LMTI = True                       # Non density for multiphase flows
LIBM = True                      # Immersed boundary method 
ctime = 0                         # Initialise ctime   
save_fig = True                   # Save results
Restart = False                   # Restart
eplsion_k = 1e-04                 # Stablisatin factor in Petrov-Galerkin for velocity
################# Physical parameters #################
rho_l = 1000                      # Density of liquid phase 
rho_g = 1.0                       # Density of gas phase 
g_x = 0;g_y = 0;g_z = -10         # Gravity acceleration (m/s2) 
diag = np.array(wA)[0,0,1,1,1]    # Diagonal component
#######################################################
# # # ################################### # # #
# # # ######    Create tensor      ###### # # #
# # # ################################### # # #
input_shape = (1,1,nz,ny,nx)
values_u = torch.zeros(input_shape, device=device)
values_v = torch.zeros(input_shape, device=device)
values_w = torch.zeros(input_shape, device=device)
values_ph = torch.zeros(input_shape, device=device)
values_pd = torch.zeros(input_shape, device=device)
alpha = torch.zeros(input_shape, device=device)
b_u = torch.zeros(input_shape, device=device)
b_v = torch.zeros(input_shape, device=device)
b_w = torch.zeros(input_shape, device=device)
k1 = torch.ones(input_shape, device=device)
k2 = torch.zeros(input_shape, device=device)
k3 = torch.ones(input_shape, device=device)*-1.0
k4 = torch.ones(input_shape, device=device)*dx**2*0.25/dt
k5 = torch.ones(input_shape, device=device)*dx**2*0.05/dt
k6 = torch.ones(input_shape, device=device)*dx**2*-0.0001/dt
input_shape_pad = (1,1,nz+2,ny+2,nx+2)
values_uu = torch.zeros(input_shape_pad, device=device)
values_vv = torch.zeros(input_shape_pad, device=device)
values_ww = torch.zeros(input_shape_pad, device=device)
values_phh = torch.zeros(input_shape_pad, device=device)
values_pdd = torch.zeros(input_shape_pad, device=device)
alphaa = torch.zeros(input_shape_pad, device=device)
rhoo = torch.zeros(input_shape_pad, device=device)
b_uu = torch.zeros(input_shape_pad, device=device)
b_vv = torch.zeros(input_shape_pad, device=device)
b_ww = torch.zeros(input_shape_pad, device=device)
k7 = torch.ones(input_shape_pad, device=device)
k8 = torch.zeros(input_shape_pad, device=device)
#######################################################
print('============== Numerical parameters ===============')
print('Mesh resolution:', values_v.shape)
print('Time step:', ntime)
print('Initial time:', ctime)
print('Diagonal componet:', diag)
#######################################################
################# Only for restart ####################
if Restart == True:
    nrestart = 8000
    ctime_old = nrestart*dt
    print('Restart solver!')
#######################################################    
################# Only for scalar #####################
if LSCALAR == True and Restart == False:
    alpha = torch.zeros(input_shape, dtype=torch.float32, device=device) 
    alpha[0,0,0:12,:,:].fill_(1.0)
    alpha[0,0,0:32,:,0:64].fill_(1.0)
    print('Switch on scalar filed solver!')
#######################################################
################# Only for scalar #####################
if LMTI == True and Restart == False:
    # rho = rho_l * torch.ones(input_shape, device=device)
    rho = torch.zeros(input_shape, device=device)
    # rho = alpha*rho_l + (1-alpha)*rho_g*50
    print('Solving multiphase flows!')
else:
    print('Solving single-phase flows!')
################# Only for IBM ########################
if LIBM == True:
    sigma = torch.zeros(input_shape, dtype=torch.float32, device=device) 
    # sigma[0,0,7:22,:,1013:1014] = 1e08  
    # sigma[0,0,22:23,:,1014:1018] = 1e08  
    # sigma[0,0,22:23,:,1020:1024] = 1e08  
 # first bottom profile  
    sigma[0,0,8:24,:,501:502] = 1e08  
    sigma[0,0,23:24,:,502:506] = 1e08  
    sigma[0,0,23:24,:,508:512] = 1e08  
 # first bottom profile  
    for i in range(nx):
       for j in range(nz):
              if i*dx >= 20.1 and  j*dz <= i*dx - 20.1:
                     sigma[0,0,j,:,i] = 1e08  
                     sigma[0,0,j,:,i] = 1e08  
                     sigma[0,0,j,:,i] = 1e08  
    print('Switch on IBM solver!')
    print('===================================================')
    plt.figure(figsize=(10, 6))
    plt.imshow(sigma.cpu().detach().numpy()[0,0,:,32,:], cmap='jet')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('Energy/Initial_sigma.jpg')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.imshow(alpha.cpu().detach().numpy()[0,0,:,32,:], cmap='jet')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('Energy/Initial_alpha.jpg')
    plt.close()
#######################################################
# # # ################################### # # #
# # # #########  AI4MULTI MAIN ########## # # #
# # # ################################### # # #
class AI4MULTI(nn.Module):
    """docstring for AI4Multi"""
    def __init__(self):
        super(AI4MULTI, self).__init__()
        # self.arg = arg
        self.xadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.zadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.difx = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.dify = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.difz = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wxu = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wxd = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wyu = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wyd = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wzu = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wzd = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.A = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)

        self.A.weight.data = wA
        self.res.weight.data = w_res

        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.zadv.weight.data = w4
        self.difx.weight.data = AD2_x
        self.dify.weight.data = AD2_y
        self.difz.weight.data = AD2_z

        self.wxu.weight.data = wxu
        self.wyu.weight.data = wyu
        self.wzu.weight.data = wzu
        self.wxd.weight.data = wxd
        self.wyd.weight.data = wyd
        self.wzd.weight.data = wzd

        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer

        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.zadv.bias.data = bias_initializer
        self.difx.bias.data = bias_initializer
        self.dify.bias.data = bias_initializer
        self.difz.bias.data = bias_initializer

        self.wxu.bias.data = bias_initializer
        self.wxd.bias.data = bias_initializer
        self.wyu.bias.data = bias_initializer
        self.wyd.bias.data = bias_initializer
        self.wzu.bias.data = bias_initializer
        self.wzd.bias.data = bias_initializer

    def boundary_condition_u(self, values_u, values_uu):
        nz = values_u.shape[2]
        ny = values_u.shape[3]
        nx = values_u.shape[4]
        nnz = values_uu.shape[2]
        nny = values_uu.shape[3]
        nnx = values_uu.shape[4]

        values_uu[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_u[0,0,:,:,:]
        values_uu[0,0,:,:,0].fill_(0.0)
        values_uu[0,0,:,:,nx+1].fill_(0.0)
        values_uu[0,0,:,0,:] = values_uu[0,0,:,1,:] 
        values_uu[0,0,:,ny+1,:] = values_uu[0,0,:,ny,:]
        values_uu[0,0,0,:,:] = values_uu[0,0,1,:,:] 
        values_uu[0,0,nz+1,:,:] = values_uu[0,0,nz,:,:]
        return values_uu

    def boundary_condition_v(self, values_v, values_vv):
        nz = values_v.shape[2]
        ny = values_v.shape[3]
        nx = values_v.shape[4]
        nnz = values_vv.shape[2]
        nny = values_vv.shape[3]
        nnx = values_vv.shape[4]

        values_vv[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_v[0,0,:,:,:]
        values_vv[0,0,:,:,0] = values_vv[0,0,:,:,1] 
        values_vv[0,0,:,:,nx+1] = values_vv[0,0,:,:,nx]
        values_vv[0,0,:,0,:].fill_(0.0)
        values_vv[0,0,:,ny+1,:].fill_(0.0)
        values_vv[0,0,0,:,:] = values_vv[0,0,1,:,:] 
        values_vv[0,0,nz+1,:,:] = values_vv[0,0,nz,:,:]
        return values_vv

    def boundary_condition_w(self, values_w, values_ww):
        nz = values_w.shape[2]
        ny = values_w.shape[3]
        nx = values_w.shape[4]
        nnz = values_ww.shape[2]
        nny = values_ww.shape[3]
        nnx = values_ww.shape[4]

        values_ww[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_w[0,0,:,:,:]
        values_ww[0,0,:,:,0] =  values_ww[0,0,:,:,1] 
        values_ww[0,0,:,:,nx+1] = values_ww[0,0,:,:,nx]
        values_ww[0,0,:,0,:] = values_ww[0,0,:,1,:]
        values_ww[0,0,:,ny+1,:] = values_ww[0,0,:,ny,:]
        values_ww[0,0,0,:,:].fill_(0.0)
        values_ww[0,0,nz+1,:,:].fill_(0.0)
        return values_ww

    def solid_body(self, values_u, values_v, values_w, sigma, dt):
        values_u = values_u / (1+dt*sigma) 
        values_v = values_v / (1+dt*sigma) 
        values_w = values_w / (1+dt*sigma) 
        return values_u, values_v, values_w

    def alpha_to_rho(self, alpha, rho, k1, rho_l, rho_g):
        alpha = torch.minimum(alpha,k1)
        alpha = torch.maximum(alpha,k1*0.05)
        rho_old = rho.clone()
        rho = alpha * rho_l + (1 - alpha) * rho_g
        return rho, rho_old

    def boundary_condition_pd(self, values_pd, values_pdd):
        nz = values_pd.shape[2]
        ny = values_pd.shape[3]
        nx = values_pd.shape[4]
        nnz = values_pdd.shape[2]
        nny = values_pdd.shape[3]
        nnx = values_pdd.shape[4]

        values_pdd[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_pd[0,0,:,:,:]

        values_pdd[0,0,:,:,0] =  values_pdd[0,0,:,:,1] 
        values_pdd[0,0,:,:,nx+1] = values_pdd[0,0,:,:,nx]
        values_pdd[0,0,:,0,:] = values_pdd[0,0,:,1,:]
        values_pdd[0,0,:,ny+1,:] = values_pdd[0,0,:,ny,:]
        values_pdd[0,0,0,:,:] = values_pdd[0,0,1,:,:]
        values_pdd[0,0,nz+1,:,:].fill_(0.0)
        return values_pdd

    def boundary_condition_ph(self, values_ph, values_phh, rho):
        nz = values_ph.shape[2]
        ny = values_ph.shape[3]
        nx = values_ph.shape[4]
        nnz = values_phh.shape[2]
        nny = values_phh.shape[3]
        nnx = values_phh.shape[4]

        values_phh[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_ph[0,0,:,:,:]

        values_phh[0,0,:,:,0] =  values_phh[0,0,:,:,1] 
        values_phh[0,0,:,:,nx+1] = values_phh[0,0,:,:,nx]
        values_phh[0,0,:,0,:] = values_phh[0,0,:,1,:]
        values_phh[0,0,:,ny+1,:] = values_phh[0,0,:,ny,:]
        values_phh[0,0,0,:,:] = values_phh[0,0,1,:,:] + dz * 10.0 * rho[0,0,1,:,:]
        values_phh[0,0,nz+1,:,:].fill_(0.0)
        return values_phh

    def boundary_condition_scalar(self, values, valuesS):  # alpha, alphaa
        """ 
        values --> alpha, rho
        valuesS --> alphaa, rhoo
        """ 
        nz = values.shape[2]
        ny = values.shape[3]
        nx = values.shape[4]
        nnz = valuesS.shape[2]
        nny = valuesS.shape[3]
        nnx = valuesS.shape[4]

        valuesS[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values[0,0,:,:,:]

        valuesS[0,0,:,:,0] =  valuesS[0,0,:,:,1] 
        valuesS[0,0,:,:,nx+1] = valuesS[0,0,:,:,nx]
        valuesS[0,0,:,0,:] = valuesS[0,0,:,1,:]
        valuesS[0,0,:,ny+1,:] = valuesS[0,0,:,ny,:] 
        valuesS[0,0,0,:,:] = valuesS[0,0,1,:,:]
        valuesS[0,0,nz+1,:,:] = valuesS[0,0,nz,:,:]
        return valuesS

    def F_cycle_MG_pd(self, values_uu, values_vv, values_ww, rho, rho_old, rhoo, values_pd, values_pdd, iteration, diag, dt, nlevel, ratio):
        rhoo = self.boundary_condition_scalar(rho, rhoo)
        b = -(-self.xadv(values_uu * rhoo) - self.yadv(values_vv * rhoo) - self.zadv(values_ww * rhoo) - (rho - rho_old) / dt) / dt
        for MG in range(iteration):
            w = torch.zeros((1,1,2,2,2*ratio), device=device)
            r = self.A(self.boundary_condition_pd(values_pd, values_pdd)) - b 
            r_s = []  
            r_s.append(r)
            for i in range(1,nlevel-1):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel-1)):
                w = w - self.A(F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)) / diag + r_s[i] / diag
                w = self.prol(w)         
            values_pd = values_pd - w 
            values_pd = values_pd - self.A(self.boundary_condition_pd(values_pd, values_pdd)) / diag + b / diag
        return values_pd, w, r

    def F_cycle_MG_ph(self, values_ph, values_phh, rhoo, iteration, diag, nlevel, ratio):
        b = self.zadv(rhoo*int(abs(g_z)))
        for MG in range(iteration):  
            w = torch.zeros((1,1,2,2,2*ratio), device=device)
            r = self.A(self.boundary_condition_ph(values_ph, values_phh, rhoo)) - b 
            r_s = [] 
            r_s.append(r)
            for i in range(1,nlevel-1):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel-1)):
                w = w - self.A(F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)) / diag + r_s[i] / diag
                w = self.prol(w) 
            values_ph = values_ph - w
            values_ph = values_ph - self.A(self.boundary_condition_ph(values_ph, values_phh, rhoo)) / diag + b / diag
        return values_ph

    def PG_vector(self, values_uu, values_vv, values_ww, values_u, values_v, values_w, rho, k4):
        k_u = 0.25 * dx * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * self.diff(values_uu)) / \
            (1e-04  + (torch.abs(self.xadv(values_uu)) * dx**-3 + torch.abs(self.yadv(values_uu)) * dx**-3 + torch.abs(self.zadv(values_uu)) * dx**-3) / 3)

        k_v = 0.25 * dy * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * self.diff(values_vv)) / \
            (1e-04  + (torch.abs(self.xadv(values_vv)) * dx**-3 + torch.abs(self.yadv(values_vv)) * dx**-3 + torch.abs(self.zadv(values_vv)) * dx**-3) / 3)

        k_w = 0.25 * dz * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * self.diff(values_ww)) / \
            (1e-04  + (torch.abs(self.xadv(values_ww)) * dx**-3 + torch.abs(self.yadv(values_ww)) * dx**-3 + torch.abs(self.zadv(values_ww)) * dx**-3) / 3)

        k_u = torch.minimum(k_u, k4) * rho
        k_v = torch.minimum(k_v, k4) * rho
        k_w = torch.minimum(k_w, k4) * rho

        k_uu = F.pad(k_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(k_v, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_ww = F.pad(k_w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        k_x = 0.5 * (k_u * self.diff(values_uu) + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * self.diff(values_vv) + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        k_z = 0.5 * (k_w * self.diff(values_ww) + self.diff(values_ww * k_ww) - values_w * self.diff(k_ww))
        return k_x, k_y, k_z

    def PG_compressive_scalar(self, alphaa, alpha, values_u, values_v, values_w, k1, k2, k3, k5, k6):
        factor_S = 1 # Negative factor to be mutiplied by S (detecting variable)
        factor_P = 1 # Postive factor to be mutiplied by S (detecting variable)
        factor_beta = 0.1        
        
        temp1 = self.xadv(alphaa)
        temp2 = self.yadv(alphaa)
        temp3 = self.zadv(alphaa)

        temp4 = temp1*(values_u*temp1+values_v*temp2+values_w*temp3)/\
        (eplsion_k+temp1**2+temp2**2+temp3**2)
        temp5 = temp2*(values_u*temp1+values_v*temp2+values_w*temp3)/\
        (eplsion_k+temp1**2+temp2**2+temp3**2)
        temp6 = temp3*(values_u*temp1+values_v*temp2+values_w*temp3)/\
        (eplsion_k+temp1**2+temp2**2+temp3**2)

        k_u = torch.minimum(k1, torch.maximum(k2, factor_S * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), k3, self.wxu(alphaa) * self.wxd(alphaa)))) * \
               -factor_beta * (values_u**2 + values_v**2 + values_w**2) / (eplsion_k + (temp1**2 + temp2**2 + temp3**2) * (torch.abs(temp4) + torch.abs(temp5) + torch.abs(temp6))/3)/dx + \
               -torch.maximum(k3, torch.minimum(k2, factor_P * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), k3, self.wxu(alphaa) * self.wxd(alphaa)))) * \
                3 * dx * torch.abs(1/3 * (dx**-3) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx + torch.abs(values_w) * dx) * self.diff(alphaa)) / \
                (eplsion_k + (torch.abs(temp1 * (dx**-3)) + torch.abs(temp2 * (dx**-3)) + torch.abs(temp3 * (dx**-3))) / 3)  

        k_v = torch.minimum(k1, torch.maximum(k2, factor_S * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), k3, self.wyu(alphaa) * self.wyd(alphaa)))) * \
               -factor_beta * (values_u**2 + values_v**2 + values_w**2) / (eplsion_k + (temp1**2 + temp2**2 + temp3**2) * (torch.abs(temp4) + torch.abs(temp5) + torch.abs(temp6))/3)/dx + \
               -torch.maximum(k3, torch.minimum(k2, factor_P * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), k3, self.wyu(alphaa) * self.wyd(alphaa)))) * \
                3 * dx * torch.abs(1/3 * (dx**-3) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx + torch.abs(values_w) * dx) * self.diff(alphaa)) / \
                (eplsion_k + (torch.abs(temp1 * (dx**-3)) + torch.abs(temp2 * (dx**-3)) + torch.abs(temp3 * (dx**-3))) / 3)  

        k_w = torch.minimum(k1, torch.maximum(k2, factor_S * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), k3, self.wzu(alphaa) * self.wzd(alphaa)))) * \
               -factor_beta * (values_u**2 + values_v**2 + values_w**2) / (eplsion_k + (temp1**2 + temp2**2 + temp3**2) * (torch.abs(temp4) + torch.abs(temp5) + torch.abs(temp6))/3)/dx + \
               -torch.maximum(k3, torch.minimum(k2, factor_P * torch.where(torch.logical_or(torch.gt(alpha,1), torch.lt(alpha,0)), k3, self.wzu(alphaa) * self.wzd(alphaa)))) * \
                3 * dx * torch.abs(1/3 * (dx**-3) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx + torch.abs(values_w) * dx) * self.diff(alphaa)) / \
                (eplsion_k + (torch.abs(temp1 * (dx**-3)) + torch.abs(temp2 * (dx**-3)) + torch.abs(temp3 * (dx**-3))) / 3)  

        k_u = torch.where(torch.gt(k_u,0.0),torch.minimum(k_u,k5),torch.maximum(k_u,k6))
        k_v = torch.where(torch.gt(k_v,0.0),torch.minimum(k_v,k5),torch.maximum(k_v,k6))
        k_w = torch.where(torch.gt(k_w,0.0),torch.minimum(k_w,k5),torch.maximum(k_w,k6))

        k_uu = F.pad(k_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(k_v, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        k_ww = F.pad(k_w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        k_x = 0.5 * (k_u * self.difx(alphaa) + self.difx(alphaa * k_uu) - alpha * self.difx(k_uu)) + \
              0.5 * (k_v * self.dify(alphaa) + self.dify(alphaa * k_vv) - alpha * self.dify(k_vv)) + \
              0.5 * (k_w * self.difz(alphaa) + self.difz(alphaa * k_ww) - alpha * self.difz(k_ww))  
        return k_x

    def forward(self, values_u, values_uu, values_v, values_vv, values_w, values_ww, values_pd, values_pdd, values_ph, values_phh, alpha, alphaa, rho, rhoo, b_uu, b_vv, b_ww, k1, k2, k3, k4, k5, k6, k7, k8, dt, rho_l, rho_g, iteration):
    # Hydrostatic pressure 
        [rho, rho_old] = self.alpha_to_rho(alpha, rho, k1, rho_l, rho_g)   
        rhoo = self.boundary_condition_scalar(rho,rhoo)
        values_ph = self.F_cycle_MG_ph(values_ph, values_phh, rhoo, iteration, diag, nlevel, ratio)
    # Padding velocity vectors 
        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)
        values_ww = self.boundary_condition_w(values_w,values_ww)
    # First step for solving u
        b_u = values_u + 0.5 * (self.PG_vector(values_uu, values_vv, values_ww, values_u, values_v, values_w, rho, k4)[0] * dt / rho - \
        values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt - values_w * self.zadv(values_uu) * dt) 
    # First step for solving v
        b_v = values_v + 0.5 * (self.PG_vector(values_uu, values_vv, values_ww, values_u, values_v, values_w, rho, k4)[1] * dt / rho - \
        values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt - values_w * self.zadv(values_vv) * dt) 
    # First step for solving w
        b_w = values_w + 0.5 * (self.PG_vector(values_uu, values_vv, values_ww, values_u, values_v, values_w, rho, k4)[2] * dt / rho - \
        values_u * self.xadv(values_ww) * dt - values_v * self.yadv(values_ww) * dt - values_w * self.zadv(values_ww) * dt) + g_z * dt
    # Pressure gradient correction - hydrostatic 
        b_u = b_u - self.xadv(self.boundary_condition_ph(values_ph, values_phh, rhoo)) / rho * dt  
        b_v = b_v - self.yadv(self.boundary_condition_ph(values_ph, values_phh, rhoo)) / rho * dt     
        b_w = b_w - self.zadv(self.boundary_condition_ph(values_ph, values_phh, rhoo)) / rho * dt 
    # Solid body
        [b_u, b_v, b_w] = self.solid_body(b_u, b_v, b_w, sigma, dt)
    # Padding velocity vectors 
        b_uu = self.boundary_condition_u(b_u,b_uu)
        b_vv = self.boundary_condition_v(b_v,b_vv)
        b_ww = self.boundary_condition_w(b_w,b_ww)
    # Second step for solving u   
        values_u = values_u + self.PG_vector(b_uu, b_vv, b_ww, b_u, b_v, b_w, rho, k4)[0] * dt / rho - b_u * self.xadv(b_uu) * dt - \
        b_v * self.yadv(b_uu) * dt - b_w * self.zadv(b_uu) * dt   
    # Second step for solving v   
        values_v = values_v + self.PG_vector(b_uu, b_vv, b_ww, b_u, b_v, b_w, rho, k4)[1] * dt / rho - b_u * self.xadv(b_vv) * dt - \
        b_v * self.yadv(b_vv) * dt - b_w * self.zadv(b_vv) * dt   
    # Second step for solving w   
        values_w = values_w + self.PG_vector(b_uu, b_vv, b_ww, b_u, b_v, b_w, rho, k4)[2] * dt / rho - b_u * self.xadv(b_ww) * dt - \
        b_v * self.yadv(b_ww) * dt - b_w * self.zadv(b_ww) * dt  + g_z * dt 
    # Pressure gradient correction - hydrostatic 
        values_u = values_u - self.xadv(self.boundary_condition_ph(values_ph, values_phh, rhoo)) / rho * dt    
        values_v = values_v - self.yadv(self.boundary_condition_ph(values_ph, values_phh, rhoo)) / rho * dt      
        values_w = values_w - self.zadv(self.boundary_condition_ph(values_ph, values_phh, rhoo)) / rho * dt  
    # Solid body
        [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
    # Transport indicator field 
        alphaa = self.boundary_condition_scalar(alpha,alphaa)
        b_u = alpha + 0.5 * (self.PG_compressive_scalar(alphaa, alpha, values_u, values_v, values_w, k1, k2, k3, k5, k6) * dt - \
        values_u * self.xadv(alphaa) * dt - values_v * self.yadv(alphaa) * dt - values_w * self.zadv(alphaa) * dt)
    # 
        b_uu = self.boundary_condition_scalar(b_u,b_uu)
        b_u = torch.maximum(torch.minimum(b_u,k1),k2)
        b_uu = torch.maximum(torch.minimum(self.boundary_condition_scalar(b_u,b_uu),k7),k8)
    # 
        alpha = alpha + (self.PG_compressive_scalar(b_uu, b_u, values_u, values_v, values_w, k1, k2, k3, k5, k6) * dt - \
        values_u * self.xadv(b_uu) * dt - values_v * self.yadv(b_uu) * dt - values_w * self.zadv(b_uu) * dt) 
    # Avoid sharp interfacing    
        [rho, rho_old] = self.alpha_to_rho(alpha, rho, k1, rho_l, rho_g)
    # non-hydrostatic pressure
        rhoo = self.boundary_condition_scalar(rho,rhoo)
        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)
        values_ww = self.boundary_condition_w(values_w,values_ww)   
        [values_pd, w ,r] = self.F_cycle_MG_pd(values_uu, values_vv, values_ww, rho, rho_old, rhoo, values_pd, values_pdd, iteration, diag, dt, nlevel, ratio)
    # Pressure gradient correction - non-hydrostatic     
        values_u = values_u + self.xadv(self.boundary_condition_pd(values_pd, values_pdd)) / rho * dt 
        values_v = values_v + self.yadv(self.boundary_condition_pd(values_pd, values_pdd)) / rho * dt 
        values_w = values_w + self.zadv(self.boundary_condition_pd(values_pd, values_pdd)) / rho * dt   
    # Solid body
        [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
        return values_u, values_v, values_w, values_ph, values_pd, alpha, rho, w, r

model = AI4MULTI().to(device)

start = time.time()
with torch.no_grad():
    for itime in range(1,ntime+1):
        [values_u, values_v, values_w, values_ph, values_pd, alpha, rho, w, r] = model(values_u, values_uu, values_v, values_vv, values_w, values_ww,
            values_pd, values_pdd, values_ph, values_phh, alpha, alphaa, rho, rhoo, b_uu, b_vv, b_ww, k1, k2, k3, k4, k5, k6, k7, k8, dt, rho_l, rho_g, iteration)
# output   
        print('Time step:', itime) 
        print('Pressure error:', np.max(np.abs(w.cpu().detach().numpy())), 'cty equation residual:', np.max(np.abs(r.cpu().detach().numpy())))
        print('========================================================')
        if np.max(np.abs(w.cpu().detach().numpy())) > 80000.0:
            np.save("temp1/dbug_alpha"+str(itime), arr=alpha.cpu().detach().numpy()[0,0,:,:])              
            np.save("temp1/dbug_w"+str(itime), arr=values_w.cpu().detach().numpy()[0,0,:,:])
            np.save("temp1/dbug_v"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
            np.save("temp1/dubg_u"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])
            print('Not converged !!!!!!')
            break
        if save_fig == True and itime % n_out == 0:
            np.save("temp1/alpha"+str(itime), arr=alpha.cpu().detach().numpy()[0,0,:,:])              
            np.save("temp1/w"+str(itime), arr=values_w.cpu().detach().numpy()[0,0,:,:])
            np.save("temp1/v"+str(itime), arr=values_v.cpu().detach().numpy()[0,0,:,:])
            np.save("temp1/u"+str(itime), arr=values_u.cpu().detach().numpy()[0,0,:,:])
    end = time.time()
    print('time',(end-start))