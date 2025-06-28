import numpy as np
import numba as nb

import parameters as param


@nb.jit(nopython=True)
def H_sb(R):
    
    ε_sb    = np.zeros(param.NMol, dtype=np.complex128)
    for imol in range(1, param.NMol+1):
        sIn  = (imol-1) * param.Modes
        eIn  = imol * param.Modes
        ε_sb[imol-1] = np.sum(param.c_nν * R[sIn: eIn])
    
    return ε_sb


@nb.jit(nopython=True)
def H_sys(R):
    
    Hij        = np.zeros((param.NStates, param.NStates), dtype=np.complex128)
    ε_sb       = H_sb(R)
    
    for imol in range(1, param.NMol+1):
        Hij[imol, imol] = param.ω0 + ε_sb[imol-1]
        Hij[imol, -1]   = param.gc 
        Hij[-1, imol]   = param.gc

    Hij[-1, -1] = param.ωc
    
    return Hij


@nb.jit(nopython=True)
def F_nν(R, ψ):
    
    F    = -param.ω_all * R
    ψ2   = np.absolute(ψ)**2
    
    for imol in range(1, param.NMol+1):
        sIn  = (imol-1) * param.Modes
        eIn  = imol * param.Modes
        F[sIn: eIn] += param.c_nν * ψ2[imol]
    
    return F