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
opt_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_20260505-110011.npz'))
accepted_states = opt_data['arr_1']
accepted_energies = opt_data['arr_2']
min_energy = opt_data['arr_3']
pareto_states = opt_data['arr_4']
pareto_values = np.array(opt_data['arr_5'])

labels = ['total_size', 'count', 'aperture_min', 'f_dist']

# Assuming pareto_values and min_energy are available from your optimization
num_obj = pareto_values.shape[1]
fig = plt.figure(figsize=(10, 7))

gs = fig.add_gridspec(num_obj, num_obj + 2) # Extra columns for the array plot
axes = [[fig.add_subplot(gs[i, j]) for j in range(num_obj)] for i in range(num_obj)]
ax_array = fig.add_subplot(gs[:, num_obj:]) # Large plot on the right for geometry
scatters = []

# 1. Create the plots with 'picker' enabled
for i in range(num_obj):
	for j in range(num_obj):
		ax = axes[i][j]
		if i == j:
				counts, bins = np.histogram(pareto_values[:, i])
				ax.stairs(counts, bins)
				ax.set_xlabel(labels[j], fontsize = 8)
		else:
				sc = ax.scatter(pareto_values[:, j], pareto_values[:, i], s=15, c='black', picker=True, pickradius=5)
				ax.set_ylabel(labels[i], fontsize = 8)
				ax.set_xlabel(labels[j], fontsize = 8)
				ax.set_ylim([0, 1])
				scatters.append(sc)

		ax.set_xlim([0, 1])
		ax.set_xticklabels([])
		ax.set_xticks([])
		ax.set_yticklabels([])
		ax.set_yticks([])

# Initial empty plot for the array
array_scatter = ax_array.scatter([], [], c='blue', s=50)
ax_array.set_title("Array Geometry (Selected)")
ax_array.set_aspect('equal')

def on_pick(event):

	if len(event.ind) == 0:
		return

	idx = event.ind[0]
	print(f"Selected Pareto Row Index: {idx}")

	for sc in scatters:
		colors = np.full(len(pareto_values), 'black', dtype=object)
		colors[idx] = 'red'

		sizes = np.full(len(pareto_values), 15)
		sizes[idx] = 100

		sc.set_color(colors)
		sc.set_sizes(sizes)

		fig.canvas.draw_idle()

	state = pareto_states[idx]
	num_elements = len(state) // 2 + 2
	coords = np.zeros((num_elements, 2), dtype=np.float32)
	coords[1, :] = [0, state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
	curr_angle = 0
	for i in range(2, num_elements):
		curr_angle += state[(num_elements - 1)+ (i - 2)]
		coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
		coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

	array_scatter.set_offsets(coords)
	ax_array.set_xlim([np.min(coords[:, 0]) - 1, np.max(coords[:, 0]) + 1])
	ax_array.set_ylim([np.min(coords[:, 1]) - 1, np.max(coords[:, 1]) + 1])

fig.canvas.mpl_connect('pick_event', on_pick)
plt.tight_layout()
# plt.show()

state_index = 143
state = pareto_states[state_index]
num_elements = len(state) // 2 + 2
coords = np.zeros((num_elements, 2), dtype=np.float32)
coords[1, :] = [0, state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
curr_angle = 0
for i in range(2, num_elements):
	curr_angle += state[(num_elements - 1)+ (i - 2)]
	coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
	coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

Y = coords[:, 0]
Z = coords[:, 1]
X = np.zeros_like(Y)

bf_array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
bf_model = BeamformingModel(bf_array, c = 1460)
plotter = BeamformingPlot(bf_model)

f = 70 # Hz
steer_de = np.radians(np.arange(0, 90, 10))
di = np.zeros(len(steer_de))
hpbw = np.zeros((len(steer_de), 3))
msll = np.zeros(len(steer_de))
for i, de in enumerate(steer_de):   
    di[i], (hpbw[i,0], hpbw[i,1], hpbw[i,2]), msll[i] =  bf_model.get_beamforming_performance_measures(frequency=f, c=1460, steer_az=np.array([[np.radians(90)]]), steer_de=np.array([[de]]))

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

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(np.degrees(steer_de), di)
ax[0].set_xscale('linear')
ax[0].set_xlabel('Steer Elevation (deg)')
ax[0].set_ylabel('Directivity Index (dB)')
ax[0].set_title('Directivity Index vs Frequency')
ax[0].grid()

ax[1].plot(np.degrees(steer_de), np.degrees(hpbw[:, 0]), 'b-', label='Min HPBW')
ax[1].plot(np.degrees(steer_de), np.degrees(hpbw[:, 1]), 'k--', label='Mean HPBW')
ax[1].plot(np.degrees(steer_de), np.degrees(hpbw[:, 2]), 'r-', label='Max HPBW')
ax[1].set_xscale('linear')
ax[1].set_xlabel('Steer Elevation (deg)')
ax[1].set_ylabel('HPBW (degrees)')
ax[1].set_title('HPBW vs Elevation')
ax[1].legend()
ax[1].grid()

ax[2].plot(np.degrees(steer_de), msll)
ax[2].set_xscale('linear')
ax[2].set_xlabel('Steer Elevation (deg)')
ax[2].set_ylabel('Max Side Lobe Level (dB)')
ax[2].set_title('MSLL vs. Elevation')
ax[2].grid()

plt.show()