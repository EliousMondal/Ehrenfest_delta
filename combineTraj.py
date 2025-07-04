import numpy as np
import numba as nb
from mpi4py import MPI
import time

import parameters as param

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NTraj = param.NTraj
NTasks = NTraj//size
NRem = NTraj - (NTasks*size)
TaskArray = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
  
        
ρt  = np.zeros((param.NSteps, param.NStates))
ρBf = np.zeros_like(ρt)

# γW = (2 / param.NStates) * (np.sqrt(param.NStates + 1) - 1)
@nb.njit
def combineR(ψTraj, ρt):
    ρt[:, :] += np.abs(ψTraj) ** 2
    # for iStep in range(param.NSteps):
    #     ρt[iStep, :] += np.outer(ψTraj[iStep, :], np.conjugate(ψTraj[iStep, :])).reshape(4)

count = 0
st = time.time()
for iTraj in TaskArray:
    # print(iTraj+1, flush=True)
    ψTraj  = np.loadtxt(f"Data/{iTraj+1}/psi_t_{iTraj+1}_model{param.model_no}.txt", dtype=np.complex128)
    # for iStep in range(param.NSteps):
    #     ρt[iStep, :] += np.outer(ψTraj[iStep, :], np.conjugate(ψTraj[iStep, :])).reshape(4)
    combineR(ψTraj, ρt)
    count += 1
print(f"# trajectories = {count} in rank {rank}", flush=True)

comm.Reduce(ρt, ρBf, op=MPI.SUM, root=0)
et = time.time()

if comm.rank == 0:
    ρBf /= param.NTraj
    np.savetxt(f"Data/rho00_t_model{param.model_no}.txt", ρBf)
    print(f"Time taken = {et-st} seconds\n")