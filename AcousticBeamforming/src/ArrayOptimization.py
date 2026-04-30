import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.special import factorial
import scipy
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

def objective_function_fast(state, optimal_state = False):

    def get_aperture_size(nodes):
        if len(nodes) < 2: return 0
        subset_points = points[list(nodes)]
        return np.max(scipy.spatial.distance.pdist(subset_points, metric='cityblock'))
    
    c = 1460 # m/s
    f_hi = 100 # Hz
    f_lo = 20 # Hz
    
    n = len(state) // 2 + 2
    coords = np.zeros((num_elements, 2), dtype=np.float32)
    coords[1, :] = [0, state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
    
    curr_angle = 0
    for i in range(2, num_elements):
        curr_angle += state[(num_elements - 1)+ (i - 2)]
        coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

    points = np.round(coords, decimals = 2) # round to 2 decimals to align with centimeter spaced grid

    total_height = (np.max(points[:, 1]) - np.min(points[:, 1])) / (5 * c / f_lo / 2)
    total_width = (np.max(points[:, 0]) - np.min(points[:, 0])) / (5 * c / f_lo / 2)

    # 0 @ extremely small size, 1 @ infinite size, 0.5 at cityblock diagonal of 5 * max_spacing
    penalty_total_size = 2 / (1 + np.exp(-(total_height + total_width))) - 1

    dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(points))
    unique_distances = np.sort(np.unique(dist_matrix)) 

    # each element has one or more active bands
    unique_distances[unique_distances < 0.01] = 0.01 - 1e-6
    f_cutoff = c / (2 * unique_distances) # sorted from highest to lowest
    active_elements = np.full((n, len(f_cutoff)-1), False)

    # 0 at val = 1, 1 as val approaches infinity
    penalty_f_range = 1 - (np.clip(f_cutoff[1], f_lo, f_hi) - np.clip(f_cutoff[-1], f_lo, f_hi))/(f_hi - f_lo)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for band_id in range(len(f_cutoff) - 1): 

        d = unique_distances[band_id + 1] # the larger separation
        # pick up any points with neighbors closer than the larger separation for the band
        rows, cols = np.where((dist_matrix <= d))
        edges = zip(rows, cols)
        G.add_edges_from(edges)

        subarrays = list(nx.connected_components(G))
        largest_subarray = max(subarrays, key=get_aperture_size) # largest by points

        active_elements[np.array(list(largest_subarray)), band_id] = True

    bands = active_elements.T 
    unique_configs, first_occurrence, mapping = np.unique(bands, axis=0, return_index=True, return_inverse=True)
    unique_subarrays = unique_configs.T

    height_data = np.zeros(unique_subarrays.shape[1])
    width_data = np.zeros(unique_subarrays.shape[1])
    f_array = np.zeros(unique_subarrays.shape[1])
    count_array = np.zeros(unique_subarrays.shape[1])
    for i in range(unique_subarrays.shape[1]): 
        
        f_array[i] = f_cutoff[np.where(mapping == i)[0][0] + 1]
        count_array[i] = np.sum(unique_subarrays[:, i])
        
        f_lo = f_cutoff[np.where(mapping == i)[0][-1] + 1]
        spacing = (c / f_lo) / 2

        element_mask = unique_subarrays[:, i].flatten()
        selected_elements = points[element_mask]

        aperture_width = np.max(selected_elements[:,0]) - np.min(selected_elements[:,0])
        aperture_height = np.max(selected_elements[:,1]) - np.min(selected_elements[:,1])
        height_data[i] = aperture_height / spacing
        width_data[i] = aperture_width / spacing

    if np.all(height_data == 0):
        penalty_aperture_min = 1
        penalty_aperture_balance = 1
    else:
        penalty_aperture_min = np.exp(-(np.min(height_data) + np.min(width_data))) # [1, 0]
        penalty_aperture_balance = 2 / (1 + np.exp(-np.max(np.abs(np.array(height_data) - np.array(width_data))))) - 1 # [0, 1]

    if len(count_array) > 2:
        med_ind = np.argmin(np.abs(np.diff(count_array - np.mean(count_array)))) # round down
        penalty_count = 1 - f_array[med_ind] / np.max(f_array) * count_array[med_ind] / np.max(count_array)
    else:
        penalty_count = 1

    penalty_aperture_variance = 2 / (1 + np.exp(-(np.std(height_data) + np.std(width_data)))) - 1 # [0, 1]
    
    # stds = np.array([2.78484950, 7.54512526e-1, 9.24490753e-1, 9.32192562e-4])
    stds = np.ones(4)

    return penalty_total_size/stds[0]*2, penalty_count/stds[1], penalty_aperture_min/stds[2], penalty_f_range/stds[3]


def objective_function_slow(state, optimal_state = False, penalty_start = 6):

    num_elements = len(state) // 2 + 2
    coords = np.zeros((num_elements, 2), dtype=np.float32)
    coords[1, :] = [0, state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
    
    curr_angle = 0
    for i in range(2, num_elements):
        curr_angle += state[(num_elements - 1)+ (i - 2)]
        coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

    coords = np.round(coords, decimals = 2) # round to 2 decimals to align with centimeter spaced grid

    f = np.floor(np.logspace(np.log10(50), np.log10(200), num = 3)) # generic frequencies (used in optimization)
    f_1 = np.random.choice(np.arange(f[0]+1, f[1], 1), size = 1) # don't need more resolution than 1 Hz. 
    f_2 = np.random.choice(np.arange(f[1]+1, f[2], 1), size = 1) # this should allow for ~100 function calls to cover the whole frequency range
    f = np.concatenate(([f[0]], f_1, [f[1]], f_2, [f[2]])) # add some randomization to the frequencies used in optimization to avoid overfitting to specific frequencies

    steer_de = np.radians(np.arange(0, 90, 10))

    # geometric calculations
    if np.isnan(coords).any() or (coords == np.inf).any():
        print(coords)
        print(state)
    tree = cKDTree(coords)
    penalty_connected_elements = 0
    height_data = []
    width_data = []
    for freq in f:
        spacing = 1460 / freq / 2

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

    penalty_connected_elements = penalty_connected_elements / len(f) # [0, 1]
    penalty_aperture_variance = 2 / (1 + np.exp(-(np.std(height_data) + np.std(width_data)))) # [0, 1]
    penalty_aperture_min = np.exp(-(np.min(height_data) + np.min(width_data))) # [1, 0]
    penalty_aperture_balance = 2 / (1 + np.exp(-np.max(np.abs(np.array(height_data) - np.array(width_data))))) # [0, 1]

    penalty_geometric = (penalty_connected_elements + penalty_aperture_variance + penalty_aperture_min + penalty_aperture_balance)/4

    # DI and MSLL calculations
    if len(np.unique(coords, axis = 0)) < num_elements:
        unq_ind = np.unique(coords, axis = 0, return_index = True)[1]
        coords = coords[np.sort(unq_ind)]

    if len(coords) < 2:
        di_f = np.ones(len(f)) * 1.761 # di of a single dipole element
        # hpbw = np.ones((len(f), 3)) * np.radians(30) # HPBW of a single dipole element
        di_de = np.ones(len(steer_de)) * 1.761
        msll_f = -np.ones(len(f))  # include the main lobe in the side lobe
        msll_de = -np.ones(len(steer_de))
    else:
        X = np.zeros(coords.shape[0], dtype=np.float32)
        Y = coords[:, 0].flatten()
        Z = coords[:, 1].flatten()

        array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
        bf_model = BeamformingModel(array, c = 1460)
        
        di_f = np.zeros(len(f))
        msll_f = np.zeros(len(f))
        for i in range(len(f)):
            di_f[i], _, msll_f[i] = bf_model.get_beamforming_performance_measures(frequency=f[i], delta_az = 10, delta_de = 2, use_primary_filter=True)
            msll_f[i] = min(-1, msll_f[i])

        
        di_de = np.zeros(len(steer_de))
        msll_de = np.zeros(len(steer_de))
        for i in range(len(steer_de)):
            di_de[i], _, msll_de[i] = bf_model.get_beamforming_performance_measures(frequency=44, delta_az = 10, delta_de = 2, steer_de=np.array([[steer_de[i]]]), use_primary_filter=True)
            msll_de[i] = min(-1, msll_de[i])

    # want to optimize the following items (in order of importance):
    # 1. minimax di
    # 2. minimize variance of di across frequenciess
    # 3. minimax HPBW across frequencies
    # 4. minimax HPBW range across frequencies (max HPBW - min HPBW)

    penalty_di_f = 0 # np.exp(-np.min(di_f) * np.log(2) / 8) # [1 -> 0], higher is better, 0.5 at 8 dB  
    penalty_di_f += 0 # np.exp(-np.max(di_f) * np.log(2) / 8) # [1 -> 0], higher is better, 0.5 at 8 dB
    penalty_di_f_variance = 2 / ( 1 + np.exp(-np.std(di_f))) - 1 # [0 -> 1], lower is better
    penalty_side_lobe_f = 1 / (np.max(np.abs(msll_f)) * 0.2 / 20) / 10 # [1 -> 0], higher is better, 0.5 at 20 dB down

    penalty_di_de = 0 # np.exp(-np.min(di_de) * np.log(1) / 8) * 0.5 # [1 -> 0], higher is better, 0.5 at 8 dB  
    penalty_di_de += 0 # np.exp(-np.max(di_de) * np.log(1) / 8) * 0.5 # [1 -> 0], higher is better, 0.5 at 8 dB
    penalty_di_de_variance = 2 / ( 1 + np.exp(-np.std(di_de))) - 1 # [0 -> 1], lower is better
    penalty_side_lobe_de = 1 / (np.max(np.abs(msll_de)) * 0.2 / 20) / 10 # [1 -> 0], higher is better, 0.5 at 20 dB down

    penalty_bf = penalty_di_f + penalty_di_de + penalty_di_f_variance + penalty_di_de_variance + penalty_side_lobe_f + penalty_side_lobe_de # + penalty_hpbw + penalty_hpbw_range # total penalty on [0, 6]

    gamma = 1.0 / 4.0
    penalty = penalty_geometric + gamma * penalty_bf

    if optimal_state:
        return [
            penalty_geometric, 
            penalty_bf
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
    theta_inc_prime = np.abs(np.arctan(0.01 / (r + 1e-9)))
    scaling[len(state) // 2 + 1:] = theta_inc_prime[-1] / theta_inc_prime

    return scaling

if __name__ == "__main__":
    np.random.seed(13)

    num_elements = 25

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_'+now)

    d_min = 1460 / 100 / 2 # half wavelength at 60 Hz
    d_max = 1460 / 20 / 2 # half wavelength at 20 Hz

    d_lims = np.tile([[d_min, d_min*2]], (num_elements-1, 1)) # make sure an element point isn't rounded above d_max
    theta_lims = np.tile([[np.radians(0), np.radians(359.5)]], (num_elements-2, 1))
    theta_lims[0,:] = [np.radians(0), np.radians(180)] # limit the angle from the second element to avoid symmetric duplicates of the same array geometry

    d_inc = np.tile(d_min / 10, num_elements-1)
    th_inc = np.tile(np.radians(15), num_elements-2)

    d_0 = np.random.uniform(d_lims[:, 0], d_lims[:, 1], size = num_elements-1)
    th_0 = np.random.uniform(theta_lims[:, 0], theta_lims[:, 1], size = num_elements-2) # element 2 is always on y-axis

    initial_state = np.concatenate((d_0, th_0))
    state_lims = np.concatenate((d_lims, theta_lims)).T
    state_inc = np.concatenate((d_inc, th_inc))

    circular_inds = np.full_like(initial_state, False, dtype = bool)
    circular_inds[len(d_0) + 2:] = True # angles are circular variables

    # parallel_opt = SAParallel(n_opt=10, temp_ratio=10)
    # opt = parallel_opt.run(func = objective_function_slow, ndim = np.shape(state_inc)[0], state_lims = state_lims, state_inc = state_inc, search_scaling_func = search_scaling)
    opt = SAOptimization(func = objective_function_fast, ndim = np.shape(state_inc)[0], state_lims = state_lims, state_inc = state_inc, search_scaling_func = search_scaling)
    opt.optimize()

    # opt.function = objective_function_slow
    # opt.r = np.array([0])
    # opt.reanneal_history = np.array([0])
    # opt.energy_reference = opt.global_min_energy
    # opt.state = opt.global_min_state
    # opt.energy = opt.global_min_energy
    # opt.optimize()

    fig, ax = plt.subplots()
    penalty_log = np.array(opt.penalty_log)
    sort_inds = np.argsort(penalty_log[:, 2])
    for i in range(penalty_log.shape[1]):
        ax.plot(penalty_log[sort_inds,i], label = f'penalty {i}')
    ax.grid()
    ax.legend()

    optimal_state = opt.global_min_state

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

    np.savez(save_file, coords, opt.accepted_states, opt.accepted_energies, opt.global_min_energy, opt.penalty_log)

    opt.plot_results()
    plt.show()

