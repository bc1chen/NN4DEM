#!/usr/bin/env python

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
import csv

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(is_gpu)

class AI4DEM(nn.Module):
	"""docstring for AI4DEM"""
	def __init__(self):
		super(AI4DEM, self).__init__()
		# self.arg = arg

    def detect_scalar_333(self, values, axis):
        """ 
        Assume that a tensor is defined as (1,1,nz,ny,nx)
        axis: 4 --> x axis (nx) ; 3 --> y axis (ny) ; 2 --> z axis (nz)
        """ 
        values = (values - torch.roll(values, 1, axis)) * (torch.roll(values, -1, axis) - values)
        nz = values.shape[2]
        ny = values.shape[3]
        nx = values.shape[4]        
        if axis == 4:
              values[0,0,:,:,0].fill_(0.0)
              values[0,0,:,:,nx-1].fill_(0.0)
        elif axis == 3:
              values[0,0,:,0,:].fill_(0.0)
              values[0,0,:,ny-1,:].fill_(0.0)
        elif axis == 2:
              values[0,0,0,:,:].fill_(0.0)
              values[0,0,nz-1,:,:].fill_(0.0)
        return values

    def detect_scalar_555(self, values, axis):
        """ 
        Assume that a tensor is defined as (1,1,nz,ny,nx)
        axis: 4 --> x axis (nx) ; 3 --> y axis (ny) ; 2 --> z axis (nz)
        """ 
        values = (values - torch.roll(values, 2, axis)) * (torch.roll(values, -2, axis) - values)
        nz = values.shape[2]
        ny = values.shape[3]
        nx = values.shape[4]        
        if axis == 4:
              values[0,0,:,:,0:2].fill_(0.0)
              values[0,0,:,:,nx-3:nx-1].fill_(0.0)
        elif axis == 3:
              values[0,0,:,0:2,:].fill_(0.0)
              values[0,0,:,ny-3:ny-1,:].fill_(0.0)
        elif axis == 2:
              values[0,0,0:2,:,:].fill_(0.0)
              values[0,0,nz-3:nz-1,:,:].fill_(0.0)
        return values

	def forward(self, C):
        Sx = self.detect_scalar(C,4) 
        Sy = self.detect_scalar(C,3) 
        Sz = self.detect_scalar(C,2) 
		return Sx, Sy, Sz

model = AI4DEM().to(device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #          AI4DEM                # #       AI4MULTIPHASE                # #
# # C # #  could be velocity, force etc  # #  volume fraction (indicator)       # #
# # s # #          ? ? ?                 # #  oscillation detecting variable    # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# torch.roll(input, shifts, dims=None) → Tensor
# input (Tensor) – the input tensor.
# shifts (int or tuple of ints) – The number of places by which the elements of the tensor are shifted. 
# If shifts is a tuple, dims must be a tuple of the same size, and each dimension will be rolled by the corresponding value
# dims (int or tuple of ints) – Axis along which to roll

def main():
	ntime = 10 # time propogation
    input_shape = (1, 1, ny, nx)
	with torch.no_grad():
	    for itime in range(1,ntime+1): # time loop
            for l in range(1,np):      # particle loop
                [Sx,Sy,Sz] = model(C) 

 if __name__ == '__main__':
 	main()






























