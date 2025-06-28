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


@nb.njit
def ψUP(ψ):
    cmt = np.sum(ψ[1:param.NMol+1]) / np.sqrt(2 * param.NMol)
    cph = ψ[-1] / np.sqrt(2)
    return cph + cmt 

@nb.njit
def ψLP(ψ):
    cmt = np.sum(ψ[1:param.NMol+1]) / np.sqrt(2 * param.NMol)
    cph = ψ[-1] / np.sqrt(2)
    return cph - cmt

@nb.njit
def ψD(ψ):
    ψd = np.zeros(ψ.shape[0]-3, dtype=np.complex128)
    for iSt in range(ψd.shape[0]):
        exp_fct = np.array([np.exp(-2 * np.pi * 1j * imol * (iSt+1) / param.NMol) for imol in range(param.NMol)])
        ψd[iSt] = np.sum(exp_fct * ψ[1:param.NMol+1])
    return ψd / np.sqrt(param.NMol)

@nb.njit        
def ψAd(ψ):
    ψad     = np.zeros_like(ψ)
    ψad[1]  = ψLP(ψ)
    ψad[2:param.NMol+1] = ψD(ψ)[:]
    ψad[-1] = ψUP(ψ)
    return ψad 
    
    
# γW = (2 / param.NStates) * (np.sqrt(param.NStates + 1) - 1)
# @nb.njit
# def combineR(ψTraj, ρt):
#     ρt[:, :] += np.abs(ψTraj) ** 2
    
@nb.njit
def combineR_Ad(ψTraj, ρt):
    
    ψTraj_Ad  = np.zeros_like(ψTraj)
    for iStp in range(ψTraj.shape[0]):
        # print(ψTraj[iStp, :])
        ψTraj_Ad[iStp, :]  = ψAd(ψTraj[iStp, :])
        
    ρt[:, :] += np.abs(ψTraj_Ad) ** 2


count = 0
st = time.time()
for iTraj in TaskArray:
    ψTraj  = np.loadtxt(f"Data/{iTraj+1}/psi_t_{iTraj+1}_model{param.model_no}.txt", dtype=np.complex128)
    # combineR(ψTraj, ρt)
    combineR_Ad(ψTraj, ρt)
    
    count += 1
print(f"# trajectories = {count} in rank {rank}", flush=True)

comm.Reduce(ρt, ρBf, op=MPI.SUM, root=0)
et = time.time()

if comm.rank == 0:
    ρBf /= param.NTraj
    np.savetxt(f"Data/rho00_t_model{param.model_no}_ad.txt", ρBf)
    print(f"Time taken = {et-st} seconds\n")