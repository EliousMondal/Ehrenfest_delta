import numpy as np

# Fundamental constant conversions
fs2au      = 41.341374575751                   # fs to au
cminv2au   = 4.55633*1e-6                      # cm⁻¹ to au
eV2au      = 0.036749405469679                 # eV to au
K2au       = 0.00000316678                     # K to au
Kb         = 8.617333262*1e-5 * eV2au / K2au   # Boltzmann constant in au

# Trajectory parameters
NTraj      = 10000                             # Total number oftrajectories
SimTime    = 600                               # Total simulation time (fs) 
# δt_fs      = 0.5                               # Bath time step (fs)
dtN        = 3#δt_fs * fs2au     
δt_fs      = dtN / fs2au                                       
NSteps     = int(SimTime / δt_fs) + 1

ω0         = (2.0 + (28.09941 / 1000)) * eV2au
ωc         = 2.0 * eV2au
gc         = (68.1 / 1000) * eV2au
NMol       = 5
Ωr         = 2 * np.sqrt(NMol) * gc
NStates    = 1 + NMol + 1


ψ0         = np.zeros(NStates, dtype=np.complex128) 
ψ0[-1]     = 1 / np.sqrt(2)
ψ0[1:-1]   = 1 / np.sqrt(2 * NMol)


model_no   = 1
c_nν       = np.loadtxt(f"../BathParams/cj_model{model_no}.txt")
ω_nν       = np.loadtxt(f"../BathParams/ωj_model{model_no}.txt")

Modes      = c_nν.shape[0]
NModes     = NMol * Modes
# T          = 1000   #300 * K2au                        # Temperature in au
β          = 1000

ω_all      = np.kron(np.ones(NMol), ω_nν) ** 2