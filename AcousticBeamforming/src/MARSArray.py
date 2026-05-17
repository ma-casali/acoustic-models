import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
import matplotlib as mpl

from BeamformingModel import BeamformingModel, BeamformingPlot
from BeamformingArray import BeamformingArray, ElementDirectivity
from ArrayShading import ArrayShading

sys.path.append(os.path.abspath('../acoustic-models'))
from SimulatedAnnealing import GridPoints

def get_hull_width(points):

    def is_colinear(points, tol = 1e-12):
        if points.shape[0] < 3:
            return True
        
        origin_points = points - points[0,:]
        rank = np.linalg.matrix_rank(origin_points, tol=tol)

        return rank <= 1
    
    if is_colinear(points):
        p1_ind = np.argmax(points[:,0])
        p2_ind = np.argmin(points[:,0])
        if p1_ind == p2_ind: # points lie along vertical axis
            return 0, np.max(points[:,1]) - np.min(points[:,1])
        else:
            return 0, np.linalg.norm(points[p1_ind,:]-points[p2_ind,:])

    hull = scipy.spatial.ConvexHull(points)
    hull_ptr = points[hull.vertices]
    hull_points = np.vstack([hull_ptr, hull_ptr[0]])
    min_width = np.inf

    dist_matrix = scipy.spatial.distance_matrix(hull_ptr, hull_ptr)
    max_width = np.max(dist_matrix)

    for i in range(len(hull_points) - 1):
        p1 = hull_points[i]
        p2 = hull_points[i+1]
        edge_vector = p2 - p1
        edge_unit = edge_vector / np.linalg.norm(edge_vector)
        normal_vec = np.array([-edge_unit[1], edge_unit[0]])

        projections = np.dot(hull_ptr - p1, normal_vec)
        current_width = np.max(np.abs(projections))

        if current_width < min_width:
            min_width = current_width

    return min_width, max_width


# initialize model
opt_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_20260515-091813.npz'))
accepted_states = opt_data['arr_0']
accepted_energies = opt_data['arr_1']
min_energy = opt_data['arr_2']
pareto_states = opt_data['arr_3']
pareto_values = np.array(opt_data['arr_4'])

# state_index = 182 # minimize of aperture size using epsilon constraint
# state_index = 122 # minimize distance variation using epsilon constraint
state_index = 469 # chebyshev highest weighting on roundness, dist cv, aperture, count
# state_index = 433 # weighted sum minimum with weighting on dist cv, aperture, roundness, count
state = pareto_states[state_index]

f_min = 10
f_max = 100
c = 1460

P = 2 # aperture in half-wavelengths
d_min = c / f_max / 2
d_max = c / f_min / 2

x = np.arange(0, P * d_max, d_min)
y = np.arange(0, P * d_max, 1.0)
X, Y = np.meshgrid(x, y)
points = np.vstack((X.flatten(), Y.flatten())).T
grid_points = GridPoints(points)

n = len(state)
available_indices = list(range(len(grid_points.points)))
final_indices = np.zeros(n, dtype = int)
for i, choice in enumerate(state):
    idx = available_indices.pop(int(choice))
    final_indices[i] = int(idx)

coords = grid_points.points[final_indices]
Y = coords[:,0]
Z = coords[:,1]
X = np.zeros_like(Y)

bf_array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
bf_model = BeamformingModel(bf_array, c = 1460)
plotter = BeamformingPlot(bf_model)

fig, ax = plt.subplots()
ax.scatter(bf_array.Y, bf_array.Z, s = 25, marker = 'o', c = 'k')
ax.set_xlabel('Y [m]')
ax.set_ylabel('Z [m]')
ax.set_aspect('equal')
# plt.show()

print("______________")
print("Optimized Array")

bands = bf_model.active_elements
unique_subarrays, first_occurrence, mapping = np.unique(bands, axis=1, return_index=True, return_inverse=True)
subarray_mask = np.sum(unique_subarrays, axis = 0) > 2
valid_inds = np.where((subarray_mask) & (bf_model.f_cutoff[first_occurrence] <= 100) & (bf_model.f_cutoff[np.unique(mapping)] >= 10))[0]
n_arrays = len(valid_inds)
aperture_size = np.zeros(n_arrays)
rows = int(np.floor(np.sqrt(n_arrays)))
cols = int(np.ceil(n_arrays/rows))
fig, ax = plt.subplots(rows, cols)
for i, valid_ind in enumerate(valid_inds): 
    ax_i = i // cols
    ax_j = i % cols
    mask = unique_subarrays[:, valid_ind].astype(bool)
    subarray_points = coords[mask]

    f_hi = bf_model.f_cutoff[first_occurrence[valid_ind]]   
    f_lo = bf_model.f_cutoff[np.where(mapping == valid_ind)[0][-1]]
    num_elements = np.sum(mask)
    ax[ax_i, ax_j].scatter(bf_array.Y, bf_array.Z, s = 25, c = 'k', marker = '.')
    ax[ax_i, ax_j].scatter(subarray_points[:,0], subarray_points[:,1], s = 25, c = 'r', marker = 'o')
    ax[ax_i, ax_j].set_xlabel("Y [m]")
    ax[ax_i, ax_j].set_ylabel("Z [m]")
    # ax[ax_i, ax_j].set_aspect('equal')
    ax[ax_i, ax_j].grid(True, which='both', linestyle='--', alpha=0.5)
    if f_hi - f_lo > 0:
        ax[ax_i, ax_j].set_title(f"[{f_lo:.1f} -> {f_hi:.1f}] Hz")
    else:
        ax[ax_i, ax_j].set_title(f"{f_lo:.1f} Hz")

    aperture_size[i], _ = get_hull_width(subarray_points) / (1460 / f_lo / 2)
    print(f" Aperture: {aperture_size[i]:.2f} for [{f_lo:.2f}, {f_hi:.2f}] Hz")

for i in range(rows*cols - len(valid_inds)):
    ax_i = (len(valid_inds) + i) // cols
    ax_j = (len(valid_inds) + i) % cols
    ax[ax_i, ax_j].remove()

plt.tight_layout(h_pad = -0.5, w_pad = -0.5)

f = np.logspace(1, 2, 20)
di = np.zeros(len(f))
hpbw = np.zeros((len(f), 3))
msll = np.zeros(len(f))
bp_list = []
for i, freq in enumerate(f):   
    di[i], (hpbw[i,0], hpbw[i,1], hpbw[i,2]), msll[i] =  bf_model.get_beamforming_performance_measures(frequency=freq, c=1460)

#     az, de, bp = bf_model.compute_beampattern(frequency=freq, c=1460)
#     bp_list.append(bp[az == 0, :].flatten())
# bp_list = np.array(bp_list)
msll[msll == 0] = np.nan

# fig, ax = plt.subplots()
# F, DE = np.meshgrid(f, np.degrees(de))
# cmap = mpl.cm.turbo
# bounds = np.arange(-21, 3, 3)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='min')
# im = ax.pcolormesh(F, DE, 20*np.log10(np.abs(bp_list.T)/np.max(np.abs(bp_list.T))), cmap=cmap, norm=norm)
# ax.set_xlabel('Frequency (Hz)')
# ax.set_xscale('log')
# ax.set_ylabel('Elevation Angle (deg)')
# fig.colorbar(im, ax=ax, label='Normalized Gain (dB)')

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(f, di)
ax[0].set_xlim([10, 100])
ax[0].set_ylim([0, 15])
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
ax[2].set_ylim([-5, 0])
ax[2].set_xscale('log')
ax[2].set_xlabel('Frequency (Hz)')
ax[2].set_ylabel('Max Side Lobe Level (dB)')
ax[2].set_title('MSLL vs. Frequency')
ax[2].grid(True, which='both',linestyle='--', alpha=0.5)

plt.tight_layout()

# MARK: - Ward Array

c = 1460 # m/s
f_u = 100 # Hz
f_l = 10 # Hz
P = 2 # half-wavelengths

N = (P + 1) + int(np.ceil(np.log10(f_u/f_l) / np.log10(P/(P-1))))
x = np.zeros(N)
for i in range(P+1):
    x[i] = (c / f_u / 2) * i
for i in range(P+1,N-1):
    x[i] = P * (c / f_u / 2) * (P/(P-1))**(i - P)
x[N-1] = P * (c / f_l / 2)

y = x.copy()

Y, Z = np.meshgrid(x, y)
Y = Y.flatten()
Z = Z.flatten()
X = np.zeros_like(Y)

coords = np.vstack((Y, Z)).T

bf_array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
bf_model = BeamformingModel(bf_array, c = 1460)

print("______________")
print("Ward Array: ")

bands = bf_model.active_elements
unique_subarrays, first_occurrence, mapping = np.unique(bands, axis=1, return_index=True, return_inverse=True)
subarray_mask = np.sum(unique_subarrays, axis = 0) > 2
valid_inds = np.where((subarray_mask) & (bf_model.f_cutoff[first_occurrence] <= 100) & (bf_model.f_cutoff[np.unique(mapping)] >= 10))[0]
n_arrays = len(valid_inds)

aperture_size = np.zeros(n_arrays)
for i, valid_ind in enumerate(valid_inds): 
    mask = unique_subarrays[:, valid_ind].astype(bool)
    subarray_points = coords[mask]
    f_hi = bf_model.f_cutoff[first_occurrence[valid_ind]]  
    f_lo = bf_model.f_cutoff[np.where(mapping == valid_ind)[0][-1]]
    aperture_size[i], _ = get_hull_width(subarray_points) / (1460 / f_lo / 2)
    print(f" Aperture: {aperture_size[i]:.2f} for [{f_lo:.2f}, {f_hi:.2f}] Hz")

f = np.logspace(1, 2, 20)
di = np.zeros(len(f))
hpbw = np.zeros((len(f), 3))
msll = np.zeros(len(f))
bp_list = []
for i, freq in enumerate(f):   
    di[i], (hpbw[i,0], hpbw[i,1], hpbw[i,2]), msll[i] =  bf_model.get_beamforming_performance_measures(frequency=freq, c=1460)
msll[msll == 0] = np.nan

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(f, di)
ax[0].set_xlim([10, 100])
ax[0].set_ylim([0, 15])
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
ax[2].set_ylim([-5, 0])
ax[2].set_xscale('log')
ax[2].set_xlabel('Frequency (Hz)')
ax[2].set_ylabel('Max Side Lobe Level (dB)')
ax[2].set_title('MSLL vs. Frequency')
ax[2].grid(True, which='both',linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()