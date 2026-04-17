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
from SimulatedAnnealing import SAOptimization

# This is a problem that is perfect for simulated annealing, since the objective function is discrete valued and non-differentiable

num_elements = 16

now = datetime.now().strftime("%Y%m%d-%H%M%S")
save_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_'+now)
print(save_file)

d_min = 343 / 3000 / 2 # half wavelength at 3 kHz
d_max = 343 / 300 / 2 # half wavelength at 300 Hz
d_lims = np.tile([[d_min, d_max - 0.01]], (num_elements-1, 1)) # make sure an element point isn't rounded above d_max
theta_lims = np.tile([[np.radians(-179.5), np.radians(180)]], (num_elements-1, 1))
d_inc = np.tile(d_min, num_elements-1)
th_inc = np.tile(np.radians(0.5), num_elements-1)
d_0 = np.random.uniform(d_lims[:, 0], d_lims[:, 1], size = num_elements-1)
th_0 = np.random.uniform(theta_lims[:, 0], theta_lims[:, 1], size = num_elements-1)
search_scaling = np.linspace(0.1, 1.0, num_elements-1) # search smaller changes for earlier elements and larger changes for later elements since earlier elements have a larger effect on the overall array geometry
search_scaling = np.concatenate((search_scaling, search_scaling)) # apply same scaling to both distance and angle parameters

initial_state = np.concatenate((d_0, th_0))
state_lims = np.concatenate((d_lims, theta_lims)).T
state_inc = np.concatenate((d_inc, th_inc))

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

    num_elements = len(state) // 2 + 1
    coords = np.zeros((num_elements, 2), dtype=np.float32)
    
    curr_angle = 0
    for i in range(1, num_elements):
        curr_angle += state[(num_elements - 1) + (i - 1)]
        coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

    coords = np.round(coords, decimals = 2) # round to 2 decimals to align with centimeter spaced grid

    f = 3 * np.logspace(2, 3, num = 3) # generic frequencies (used in optimization)
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
    penalty_side_lobe = 0.1 * np.exp(-np.min(np.abs(msll)) * np.log(2) / 3) # [0.1 -> 0], higher is better, 0.05 at 3 dB down, de-emphasized 
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

opt = SAOptimization(objective_function_slow, state_0 = initial_state, state_lims = state_lims, state_inc = state_inc, search_scaling = search_scaling)
optimal_state, optimal_value, accepted_states, accepted_energies, proposed_states = opt.optimize()

print("Penalty components: ", objective_function_slow(optimal_state, optimal_state = True))

coords = np.zeros((num_elements, 2), dtype=np.float32)
curr_angle = 0
for i in range(1, num_elements):
    curr_angle += optimal_state[(num_elements - 1) + (i - 1)]
    coords[i, 0] = coords[i-1, 0] + optimal_state[i-1] * np.cos(curr_angle)
    coords[i, 1] = coords[i-1, 1] + optimal_state[i-1] * np.sin(curr_angle)

Y = coords[:, 0]
Z = coords[:, 1]
plt.scatter(Y, Z)

np.savez(save_file, coords, accepted_states, accepted_energies, optimal_value)

print("Optimal state (Y): ", [(np.round(float(y), 2)) for y in Y])
print("Optimal state (Z): ", [(np.round(float(z), 2)) for z in Z])

opt.plot_results()
plt.show()

