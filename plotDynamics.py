import numpy as np
import matplotlib.pyplot as plt

import parameters as param


# ρt      = np.loadtxt(f"Data/rho00_t_model{param.model_no}.txt")
ρ_ad    = np.loadtxt(f"Data/rho00_t_model{param.model_no}_ad.txt")
    
ρ_plot  = np.zeros((ρ_ad.shape[0], 3))
ρ_plot[:, 0] = ρ_ad[:, 1]
ρ_plot[:, 1] = np.sum(ρ_ad[:, 2:param.NMol+1], axis=1)
ρ_plot[:, 2] = ρ_ad[:, -1]

print(np.sum(ρ_plot, axis=1))

τ_array = np.linspace(0, param.SimTime, param.NSteps) / 1000

plt.plot(τ_array, ρ_plot[:, 0],  lw=4, ls="-", color="#e67e22", label=f"ρ₋₋")
plt.plot(τ_array, ρ_plot[:, 1],  lw=4, ls="-", color="#3498db", label=r"ρ$_{\mathrm{DD}}$")
plt.plot(τ_array, ρ_plot[:, 2],  lw=4, ls="-", color="#e74c3c", label=f"ρ₊₊")


# colors  = [ "#3498db", "#2ecc71", "#1abc9c", "#e67e22", "#e74c3c", "#f1c40f" ]
# for imol in range(1, param.NMol+1):
#     plt.plot(τ_array, ρt[:, imol],  lw=4, ls="-", color=colors[imol-1], label=f"E{imol}")
# plt.plot(τ_array, ρt[:, -1], lw=4, ls="-", color=colors[-1], label="G1")


plt.ylim(0.0 ,1.00)
plt.xlim(0, param.SimTime / 1000)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
# plt.xticks([0, 4, 8, 12], fontsize=15)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=15)
plt.legend(frameon=False, fontsize=15, handlelength=0.75, ncols=3, columnspacing=1.0, handletextpad=0.25)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f"pop_model{param.model_no}_ad.png", dpi=300, bbox_inches="tight")
# plt.savefig("sigma_z.pdf", dpi=300, bbox_inches="tight")