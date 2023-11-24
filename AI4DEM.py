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

bias_initializer = torch.tensor([0.0])

# Detecting filters (3x3x3) Chris suggested to use 1x1x3 for computing efficiency 
# 0  0  0          0  0  0
# -1 0  1          1  0  -1
# 0  0  0          0  0  0 
wxu_3 = torch.zeros([1,1,3,3,3]) ; wxd_3 = torch.zeros([1,1,3,3,3])
wyu_3 = torch.zeros([1,1,3,3,3]) ; wyd_3 = torch.zeros([1,1,3,3,3])
wzu_3 = torch.zeros([1,1,3,3,3]) ; wzd_3 = torch.zeros([1,1,3,3,3])
wxu_3[0,0,1,1,1] = 1.0 ; wxu_3[0,0,1,1,0] = -1.0 ; wxd_3[0,0,1,1,1] = -1.0 ; wxd_3[0,0,1,1,2] = 1.0
wyu_3[0,0,1,1,1] = 1.0 ; wyu_3[0,0,1,0,1] = -1.0 ; wyd_3[0,0,1,1,1] = -1.0 ; wyd_3[0,0,1,2,1] = 1.0
wzu_3[0,0,1,1,1] = 1.0 ; wzu_3[0,0,0,1,1] = -1.0 ; wzd_3[0,0,1,1,1] = -1.0 ; wzd_3[0,0,2,1,1] = 1.0

# Detecting filters (5x5x5) Chris suggested to use 1x1x5 for computing efficiency 
# 0  0  0  0  0        0  0  0  0  0 
# 0  0  0  0  0        0  0  0  0  0 
# -1 0  0  0  1        1  0  0  0  -1
# 0  0  0  0  0        0  0  0  0  0 
# 0  0  0  0  0        0  0  0  0  0 
wxu_5 = torch.zeros([1,1,5,5,5]) ; wxd_5 = torch.zeros([1,1,5,5,5])
wyu_5 = torch.zeros([1,1,5,5,5]) ; wyd_5 = torch.zeros([1,1,5,5,5])
wzu_5 = torch.zeros([1,1,5,5,5]) ; wzd_5 = torch.zeros([1,1,5,5,5])
wxu_5[0,0,2,2,2] = 1.0 ; wxu_5[0,0,2,2,0] = -1.0 ; wxd_5[0,0,2,2,2] = -1.0 ; wxd_5[0,0,2,2,4] = 1.0
wyu_5[0,0,2,2,2] = 1.0 ; wyu_5[0,0,2,0,2] = -1.0 ; wyd_5[0,0,2,2,2] = -1.0 ; wyd_5[0,0,2,4,2] = 1.0
wzu_5[0,0,2,2,2] = 1.0 ; wzu_5[0,0,0,2,2] = -1.0 ; wzd_5[0,0,2,2,2] = -1.0 ; wzd_5[0,0,4,2,2] = 1.0


class AI4DEM(nn.Module):
	"""docstring for AI4DEM"""
	def __init__(self):
		super(AI4DEM, self).__init__()
		# self.arg = arg
        self.wxu_3 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wxd_3 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wyu_3 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wyd_3 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wzu_3 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.wzd_3 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.wxu_5 = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=0)
        self.wxd_5 = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=0)
        self.wyu_5 = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=0)
        self.wyd_5 = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=0)
        self.wzu_5 = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=0)
        self.wzd_5 = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=0)

        self.wxu_3.weight.data = wxu_3
        self.wxd_3.weight.data = wxd_3
        self.wyu_3.weight.data = wyu_3
        self.wyd_3.weight.data = wyd_3
        self.wzu_3.weight.data = wzu_3
        self.wzd_3.weight.data = wzd_3

        self.wxu_5.weight.data = wxu_5
        self.wxd_5.weight.data = wxd_5
        self.wyu_5.weight.data = wyu_5
        self.wyd_5.weight.data = wyd_5
        self.wzu_5.weight.data = wzu_5
        self.wzd_5.weight.data = wzd_5

        self.wxu_3.bias.data = bias_initializer
        self.wxd_3.bias.data = bias_initializer
        self.wyu_3.bias.data = bias_initializer
        self.wyd_3.bias.data = bias_initializer
        self.wzu_3.bias.data = bias_initializer
        self.wzd_3.bias.data = bias_initializer

        self.wxu_5.bias.data = bias_initializer
        self.wxd_5.bias.data = bias_initializer
        self.wyu_5.bias.data = bias_initializer
        self.wyd_5.bias.data = bias_initializer
        self.wzu_5.bias.data = bias_initializer
        self.wzd_5.bias.data = bias_initializer

		def forward(self, C):
			s1 = self.wxu_5(C) * self.wxd_5(C)
			s2 = self.wxu_3(C) * self.wxd_3(C)
			return torch.minmum(s1,s2)

model = AI4DEM().to(device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #          AI4DEM                # #       AI4MULTIPHASE                # #
# # C # #  could be velocity, force etc  # #  volume fraction (indicator)       # #
# # s # #          ? ? ?                 # #  oscillation detecting variable    # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# wxu_3(c) -> C_{i-1} - C_{i}
# wxd_3(c) -> C_{i+1} - C_{i}

# wxu_5(c) -> C_{i-2} - C_{i}
# wxd_5(c) -> C_{i+2} - C_{i}

# s1 = wxu_3(c) * wxd_3(c) -> (C_{i-1} - C_{i}) x (C_{i+1} - C_{i}) eq(29)
# s2 = wxu_5(c) * wxd_5(c) -> (C_{i-2} - C_{i}) x (C_{i+2} - C_{i}) eq(32)

# torch.minimum(s1, s2)
def main():
	ntime = 10 # time propogation

    input_shape = (1, 1, ny, nx)

	with torch.no_grad():
	    for itime in range(1,ntime+1): # time loop
            for l in range(1,np):      # particle loop
                s = model(C) 

 if __name__ == '__main__':
 	main()






























