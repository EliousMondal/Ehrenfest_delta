import numpy as np
import numba as nb

import parameters as param
import model
    

@nb.jit(nopython=True)
def evolve_ψ(R, ψt, δt):
    Hτ     = model.H_sys(R)
    E, U   = np.linalg.eigh(Hτ)
    U_Hτ   = U @ np.diag(np.exp(-1j * E * δt)) @ np.conjugate(U.T)
    return U_Hτ @ ψt


@nb.jit(nopython=True)
def evolve_R(R, P, ψ, δt):
    F1       = model.F_nν(R, ψ)
    R_δt     = R + (P * δt) #+ (0.5 * F1 * δt ** 2)
    
    F2       = model.F_nν(R_δt, ψ) 
    P_δt     = P + 0.5 * (F1 + F2) * δt
    return R_δt, P_δt


@nb.jit(nopython=True)
def evolve_ψR(R, P, ψ, δt):
    
    ψ_δt_hf    = evolve_ψ(R, ψ, δt/2)          # half-step system evolution
    R_δt, P_δt = evolve_R(R, P, ψ_δt_hf, δt)   # Bath evolution
    ψ_δt       = evolve_ψ(R_δt, ψ_δt_hf, δt/2) # half-step system evolution
    
    return ψ_δt, R_δt, P_δt


@nb.jit(nopython=True)
def evolve(R, P, nSteps, nStates):
    ψt       = np.zeros((nSteps, nStates), dtype=np.complex128)
    ψt[0, :] = param.ψ0
    
    for iStep in range(1, nSteps):
        ψt[iStep, :], R, P = evolve_ψR(R, P, ψt[iStep-1, :], param.dtN)
    
    return ψt
    