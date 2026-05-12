import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl

from BeamformingModel import BeamformingModel, BeamformingPlot
from BeamformingArray import BeamformingArray, ElementDirectivity
from ArrayShading import ArrayShading

# initialize model
opt_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_20260511-143126.npz'))
accepted_states = opt_data['arr_1']
accepted_energies = opt_data['arr_2']
min_energy = opt_data['arr_3']
pareto_states = opt_data['arr_4']
pareto_values = np.array(opt_data['arr_5'])

state_index = 26
state = pareto_states[state_index]
num_elements = len(state) // 2 + 2
num_elements = len(state) // 2 + 2
n = num_elements
freqs = state[:(n-1)]
angles = state[(n-1):]
cum_angles = np.cumsum(angles)
dx = np.zeros(n-1, dtype=np.float32)
dy = np.zeros(n-1, dtype=np.float32)
dy[0] = (1460 / freqs[0]) / 2
dx[1:] = (1460 / freqs[1:]) / 2 * np.cos(cum_angles)
dy[1:] = (1460 / freqs[1:]) / 2 * np.sin(cum_angles)
coords = np.zeros((n,2), dtype = np.float32)
coords[1:,0] = np.cumsum(dx)
coords[1:,1] = np.cumsum(dy)
coords = np.round(coords, decimals=2)

Y = coords[:, 0]
Z = coords[:, 1]
X = np.zeros_like(Y)

bf_array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
bf_model = BeamformingModel(bf_array, c = 1460)
plotter = BeamformingPlot(bf_model)

f = np.logspace(1, 2, 20)
di = np.zeros(len(f))
hpbw = np.zeros((len(f), 3))
msll = np.zeros(len(f))

bp_list = []
for i, freq in enumerate(f):   
    di[i], (hpbw[i,0], hpbw[i,1], hpbw[i,2]), msll[i] =  bf_model.get_beamforming_performance_measures(frequency=freq, c=1460)

    az, de, bp = bf_model.compute_beampattern(frequency=freq, c=1460)
    bp_list.append(bp[az == 0, :].flatten())
bp_list = np.array(bp_list)
msll[msll == 0] = np.nan

fig, ax = plt.subplots()
F, DE = np.meshgrid(f, np.degrees(de))
cmap = mpl.cm.turbo
bounds = np.arange(-21, 3, 3)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='min')
im = ax.pcolormesh(F, DE, 20*np.log10(np.abs(bp_list.T)/np.max(np.abs(bp_list.T))), cmap=cmap, norm=norm)
ax.set_xlabel('Frequency (Hz)')
ax.set_xscale('log')
ax.set_ylabel('Elevation Angle (deg)')
fig.colorbar(im, ax=ax, label='Normalized Gain (dB)')

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(f, di)
ax[0].set_xlim([10, 100])
ax[0].set_ylim(bottom=0)
ax[0].set_xscale('log')
ax[0].set_xlabel('Frequency (Hz))')
ax[0].set_ylabel('Directivity Index (dB)')
ax[0].set_title('Directivity Index vs Frequency')
ax[0].grid(True, which='both', linestyle='--', alpha=0.5)

ax[1].plot(f, np.degrees(hpbw[:, 0]), 'b-', label='Min HPBW')
ax[1].plot(f, np.degrees(hpbw[:, 1]), 'k--', label='Mean HPBW')
ax[1].plot(f, np.degrees(hpbw[:, 2]), 'r-', label='Max HPBW')
ax[1].set_xlim([10, 100])
ax[1].set_ylim([0, 90])
ax[1].set_xscale('log')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('HPBW (degrees)')
ax[1].set_title('HPBW vs Frequency')
ax[1].legend()
ax[1].grid(True, which='both', linestyle='--', alpha=0.5)

ax[2].plot(f, msll)
ax[2].set_xlim([10, 100])
ax[2].set_ylim([-3, 0])
ax[2].set_xscale('log')
ax[2].set_xlabel('Frequency (Hz)')
ax[2].set_ylabel('Max Side Lobe Level (dB)')
ax[2].set_title('MSLL vs. Frequency')
ax[2].grid(True, which='both',linestyle='--', alpha=0.5)

plt.show()