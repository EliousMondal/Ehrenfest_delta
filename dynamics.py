import numpy as np
from numpy import random as rd
from mpi4py import MPI
import time 

import os
import sys

import initBath as iBth
import parameters as param
import method

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TrajDir     = sys.argv[1]
NTraj       = param.NTraj
NTasks      = NTraj//size
NRem        = NTraj - (NTasks*size)
TaskArray   = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
        
# compiling the code
# ixp_dum      = np.loadtxt(f"Data/1/iRP_1_lam50.txt")
R_dum, P_dum = iBth.initR() #ixp_dum[:, 0], ixp_dum[:, 1]
ψt_dum       = method.evolve(R_dum, P_dum, 2, param.NStates)
        
st_rank = time.time()
  
for iTraj in TaskArray:
    st_traj  = time.time()
    os.makedirs(f"Data/{iTraj+1}", exist_ok=True)
    # ixp      = np.loadtxt(f"Data/{iTraj+1}/iRP_{iTraj+1}_lam50.txt")
    R, P     = iBth.initR()     #ixp[:, 0], ixp[:, 1]
    ψt       = method.evolve(R, P, param.NSteps, param.NStates)
    
    ed_traj  = time.time()
    print(f"Time for trajectory {iTraj+1} = {np.round(ed_traj-st_traj, 8)}", flush=True)
    np.savetxt(f"Data/{iTraj+1}/psi_t_{iTraj+1}_model{param.model_no}.txt", ψt)
    
ed_rank = time.time()
print(f"Process {rank} took {np.round(ed_rank - st_rank, 8)} seconds for computing {len(TaskArray)} trajectories.")