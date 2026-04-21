import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.special import factorial
import networkx as nx
import sys
import os
import cv2
from datetime import datetime

from BeamformingArray import BeamformingArray, ElementDirectivity
from BeamformingModel import BeamformingModel
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../acoustic-models'))
from SimulatedAnnealing import SAOptimization, SAParallel

# This is a problem that is perfect for simulated annealing, since the objective function is discrete valued and non-differentiable

num_elements = 10

now = datetime.now().strftime("%Y%m%d-%H%M%S")
save_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_'+now)

d_min = 1460 / 500 / 2 # half wavelength at 500 Hz
d_max = 1460 / 50 / 2 # half wavelength at 50 Hz

d_lims = np.tile([[d_min, d_max - 0.01]], (num_elements-1, 1)) # make sure an element point isn't rounded above d_max
theta_lims = np.tile([[np.radians(0), np.radians(359.5)]], (num_elements-2, 1))
theta_lims[0,:] = [np.radians(0), np.radians(180)] # limit the angle from the second element to avoid symmetric duplicates of the same array geometry

d_inc = np.tile(d_min, num_elements-1)
th_inc = np.tile(np.radians(0.5), num_elements-2)

d_0 = np.random.uniform(d_lims[:, 0], d_lims[:, 1], size = num_elements-1)
th_0 = np.random.uniform(theta_lims[:, 0], theta_lims[:, 1], size = num_elements-2) # element 2 is always on y-axis

initial_state = np.concatenate((d_0, th_0))
state_lims = np.concatenate((d_lims, theta_lims)).T
state_inc = np.concatenate((d_inc, th_inc))

circular_inds = np.full_like(initial_state, False, dtype = bool)
circular_inds[len(d_0) + 2:] = True # angles are circular variables

def objective_function_fast(state, optimal_state = False):

    def range_increase(x, a = 1, b = 0):
        if a * x < 1e-6:
            return 1e6
        else:
            return 1/(a * x) + b 
    
    gamma_number = 1
    gamma_std = 1
    gamma_min = 1
    gamma_balance = 1
    
    num_elements = len(state) // 2
    coords = np.zeros((num_elements, 2), dtype=np.float32)
    curr_angle = state[num_elements + 1]
    coords[1,:] = [state[0] * np.cos(curr_angle), state[0] * np.sin(curr_angle)]
    
    for i in range(2, num_elements):
        curr_angle += state[num_elements + i - 1]
        coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

    coords = np.round(coords, decimals = 2) # round to 2 decimals to align with centimeter spaced grid

    tree = cKDTree(coords)

    f = 3 * np.logspace(2, 3, num = 3) # generic frequencies (used in optimization)
    f_1 = np.random.uniform(f[0]*1.05, f[0]*0.95, size = 3)
    f_2 = np.random.uniform(f[1]*1.05, f[2]*0.95, size = 4)
    f = np.concatenate(([f[0]], f_1, [f[1]], f_2, [f[2]])) # add some randomization to the frequencies used in optimization to avoid overfitting to specific frequencies

    penalty_connected_elements = 0
    num_freq = 100
    height_data = []
    width_data = []
    for freq in f:
        spacing = 343 / freq / 2

        pairs = tree.query_pairs(r = spacing)
        filtered_pairs = set()
        for (i, j) in pairs:
            dist = np.linalg.norm(coords[i] - coords[j])
            # only show connections that are at least 1/5 of the wavelength apart to avoid apertures that are too small compared to a wavelength
            if dist >= spacing * 2 / 5: 
                filtered_pairs.add((i, j))

        G = nx.Graph()
        G.add_nodes_from(range(num_elements))
        G.add_edges_from(filtered_pairs)

        components = list(nx.connected_components(G))
        max_ind = np.argmax([len(c) for c in components])
        connected_points = coords[list(components[max_ind])]

        N = len(connected_points)
        penalty_connected_elements += 1 - np.atan(np.sqrt((num_elements - N)/num_elements) * np.pi / 2)
    
        rect = cv2.minAreaRect(connected_points)
        aperture_width = rect[1][0]
        aperture_height = rect[1][1]
        height_val = aperture_height / spacing
        width_val = aperture_width / spacing

        # cost increases rapidly as the aperture size decreases and 1 at num_elements
        height_data.append(height_val)
        width_data.append(width_val)

    penalty_connected_elements = range_increase(penalty_connected_elements / 10, b = -1) * gamma_number
    penalty_aperture_variance = range_increase(1 - np.std(height_data) + np.std(width_data), b = -1) * gamma_std 
    penalty_aperture_min = range_increase(np.min(height_data) + np.min(width_data), a = 1 / (num_elements * 2)) * gamma_min
    penalty_aperture_balance = range_increase(np.max(np.abs(np.array(height_data) - np.array(width_data))), b = -1) * gamma_balance
    
    if optimal_state:
        return [
            penalty_connected_elements,
            penalty_aperture_variance,
            penalty_aperture_min,
            penalty_aperture_balance
        ]

    penalty = penalty_connected_elements + penalty_aperture_variance + penalty_aperture_min + penalty_aperture_balance

    return penalty

def objective_function_slow(state, optimal_state = False):

    penalty_add = 0

    num_elements = len(state) // 2 + 2
    coords = np.zeros((num_elements, 2), dtype=np.float32)
    coords[1, :] = [0, state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
    
    curr_angle = 0
    for i in range(2, num_elements):
        curr_angle += state[(num_elements - 1)+ (i - 2)]
        coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

    coords = np.round(coords, decimals = 2) # round to 2 decimals to align with centimeter spaced grid

    f = 5 * np.logspace(1, 2, num = 3) # generic frequencies (used in optimization)
    f_1 = np.random.uniform(f[0]*1.05, f[0]*0.95, size = 3)
    f_2 = np.random.uniform(f[1]*1.05, f[2]*0.95, size = 4)
    f = np.concatenate(([f[0]], f_1, [f[1]], f_2, [f[2]])) # add some randomization to the frequencies used in optimization to avoid overfitting to specific frequencies

    if len(np.unique(coords, axis = 0)) < num_elements:
        unq_ind = np.unique(coords, axis = 0, return_index = True)[1]
        coords = coords[np.sort(unq_ind)]

    if len(coords) < 2:
        di = np.ones(len(f)) * 1.761 # di of a single dipole element
        hpbw = np.ones((len(f), 3)) * np.radians(30) # HPBW of a single dipole element
        msll = np.ones(len(f)) * -1 # include the main lobe in the side lobe
    else:
        X = np.zeros(coords.shape[0], dtype=np.float32)
        Y = coords[:, 0].flatten()
        Z = coords[:, 1].flatten()

        array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
        bf_model = BeamformingModel(array)
        
        di = np.zeros(len(f))
        msll = np.zeros(len(f))
        for i in range(len(f)):
            di[i], _, msll[i] = bf_model.get_beamforming_performance_measures(frequency=f[i], delta_az = 10, delta_de = 2)

    # want to optimize the following items (in order of importance):
    # 1. minimax di
    # 2. minimize variance of di across frequenciess
    # 3. minimax HPBW across frequencies
    # 4. minimax HPBW range across frequencies (max HPBW - min HPBW)

    penalty_di = np.exp(-np.min(di) * np.log(2) / 8) # [1 -> 0], higher is better, 0.5 at 8 dB  
    penalty_di += np.exp(-np.max(di) * np.log(2) / 8) # [1 -> 0], higher is better, 0.5 at 8 dB
    penalty_di_variance = 2 / ( 1 + np.exp(-np.std(di))) - 1 # [0 -> 1], lower is better
    penalty_side_lobe = np.exp(-np.min(np.abs(msll)) * np.log(2) / 3) # [1 -> 0], higher is better, 0.5 at 3 dB down

    # penalty_hpbw = 2 / (1 + np.exp(-np.max(hpbw[:, 2]))) - 1 # [0 -> 1], lower is better
    # penalty_hpbw_range = 2 / (1 + np.exp(-np.max(hpbw[:, 2] - hpbw[:, 0]))) - 1 # [0 -> 1], lower is better
        
    penalty = penalty_add + penalty_di + penalty_di_variance + penalty_side_lobe # + penalty_hpbw + penalty_hpbw_range # total penalty on [0, 4]

    if optimal_state:
        return [
            penalty_di,
            penalty_di_variance,
            penalty_side_lobe
        ]

    return penalty

def search_scaling(state, state_inc):

    # scaling occurs only for angles
    # each angle has an effect equivalent to how much extent that it rotates
    # scale this effect so that it's normalized to the effect of the last angle
    # in other words, each angle should have similar effect on the search space

    theta = state[len(state) // 2 + 1:] # for 16 elements, this should be 14 angles
    d = state[1:len(state) // 2 + 1]
    scaling = np.ones_like(state)
    r = d * np.cos(np.flip(np.cumsum(np.flip(theta))))
    theta_inc_prime = np.abs(np.arctan(0.01 / r))
    scaling[len(state) // 2 + 1:] = theta_inc_prime[-1] / theta_inc_prime

    return scaling

if __name__ == "__main__":
    np.random.seed(13)

    parallel_opt = SAParallel(n_opt=20, temp_ratio=10)
    min_opt = parallel_opt.run(func = objective_function_slow, ndim = np.shape(state_inc)[0], state_lims = state_lims, state_inc = state_inc, search_scaling_func = search_scaling)
    optimal_state = min_opt.global_min_state

    coords = np.zeros((num_elements, 2), dtype=np.float32)
    coords[1, :] = [0, optimal_state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
    curr_angle = 0
    for i in range(2, num_elements):
        curr_angle += optimal_state[(num_elements - 1)+ (i - 2)]
        coords[i, 0] = coords[i-1, 0] + optimal_state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + optimal_state[i-1] * np.sin(curr_angle)

    Y = coords[:, 0]
    Z = coords[:, 1]
    plt.scatter(Y, Z)

    np.savez(save_file, coords, min_opt.accepted_states, min_opt.accepted_energies, min_opt.global_min_energy)

    min_opt.plot_results()
    plt.show()

