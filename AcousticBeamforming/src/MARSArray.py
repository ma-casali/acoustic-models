import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import seaborn as sns

from BeamformingModel import BeamformingModel, BeamformingPlot
from BeamformingArray import BeamformingArray, ElementDirectivity
from ArrayShading import ArrayShading

# initialize model
opt_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_20260429-194831.npz'))
coords = opt_data['arr_0']
accepted_states = opt_data['arr_1']
accepted_energies = opt_data['arr_2']
penalties = np.array(opt_data['arr_4'])
num_elements = np.shape(coords)[0]

X = np.zeros(num_elements)

Y = coords[:,0] - np.mean(coords[:, 0])
Z = coords[:,1] - np.mean(coords[:, 1])

array = BeamformingArray(X, Y, Z, design_frequency=44, element_directivity=ElementDirectivity.DIPOLE)
bf_model = BeamformingModel(array)
plotter = BeamformingPlot(bf_model)

print(np.std(penalties, axis = 0))

labels = ['total_size', 'count', 'aperture_min', 'f_range']

res = scipy.stats.spearmanr(penalties, axis = 0)
norm = mpl.colors.CenteredNorm(vcenter=0)
ax = sns.heatmap(res.statistic, annot=True, cmap='RdBu', norm=norm, fmt='.2f')
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

f = 10 # Hz
# steer_de = np.radians(np.arange(0, 90, 10))
# di = np.zeros(len(steer_de))
# hpbw = np.zeros((len(steer_de), 3))
# msll = np.zeros(len(steer_de))
# for i, de in enumerate(steer_de):   
#     di[i], (hpbw[i,0], hpbw[i,1], hpbw[i,2]), msll[i] =  bf_model.get_beamforming_performance_measures(frequency=f, c=1460, steer_az=np.array([[np.radians(90)]]), steer_de=np.array([[de]]))

bands = bf_model.active_elements
unique_subarrays, first_occurrence, mapping = np.unique(bands, axis=1, return_index=True, return_inverse=True)

subarray_mask = np.sum(unique_subarrays, axis = 0) > 2
valid_inds = np.where(subarray_mask)[0]

for i in valid_inds: 
        
        f_hi = bf_model.f_cutoff[first_occurrence[i]+1]
        f_lo = bf_model.f_cutoff[np.where(mapping == i)[0][-1] + 1]
        
        print(f"[{f_lo:.2f}, {f_hi:.2f}], {f_hi-f_lo:.2e} Hz BW : {np.sum(unique_subarrays[:, i])} elements")

az, de, bp = bf_model.compute_beampattern(frequency=f, c = 1460, use_primary_filter=True)
plotter.plot_beampattern_image(f, az, de, bp, method= 'polar')

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].plot(np.degrees(steer_de), di)
# ax[0].set_xscale('linear')
# ax[0].set_xlabel('Steer Elevation (deg)')
# ax[0].set_ylabel('Directivity Index (dB)')
# ax[0].set_title('Directivity Index vs Frequency')
# ax[0].grid()

# ax[1].plot(np.degrees(steer_de), np.degrees(hpbw[:, 0]), 'b-', label='Min HPBW')
# ax[1].plot(np.degrees(steer_de), np.degrees(hpbw[:, 1]), 'k--', label='Mean HPBW')
# ax[1].plot(np.degrees(steer_de), np.degrees(hpbw[:, 2]), 'r-', label='Max HPBW')
# ax[1].set_xscale('linear')
# ax[1].set_xlabel('Steer Elevation (deg)')
# ax[1].set_ylabel('HPBW (degrees)')
# ax[1].set_title('HPBW vs Elevation')
# ax[1].legend()
# ax[1].grid()

# ax[2].plot(np.degrees(steer_de), msll)
# ax[2].set_xscale('linear')
# ax[2].set_xlabel('Steer Elevation (deg)')
# ax[2].set_ylabel('Max Side Lobe Level (dB)')
# ax[2].set_title('MSLL vs. Elevation')
# ax[2].grid()

plt.show()