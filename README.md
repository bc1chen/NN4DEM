# NN4DEM (Neural Netwwork framework for Discrete Element Method)

This repository contains code devloped as part of a broader initiative to harness specialised AI hardware and software environments, marking a transition from traditional computational physics programming approaches.

#### Default filter size -> 5 x 5 in 2D and 5 x 5 x 5 in 3D 

#### Grid_particle_4.py            -> Initial testing 2D code for solving particle-to-particle interaction
#### Grid_particle_4_filter.py     -> Create filter function using shifting operation (simultanesouly - high memory)
#### Grid_particle_4_filter_seq.py -> Create filter function using shifting operation (sequentially   - low memory)

#### AI4DEM_speed.py     -> Initial AI4DEM code for testing the code (simultanesouly - high memory) ** do not consider boundary conditions **
#### AI4DEM_speed_seq.py -> Initial AI4DEM code for testing the code (sequentially   - low memory)  ** do not consider boundary conditions **
