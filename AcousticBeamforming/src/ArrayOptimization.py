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
from SimulatedAnnealing import SAOptimization

# This is a problem that is perfect for simulated annealing, since the objective function is discrete valued and non-differentiable

def objective_function_fast(state, optimal_state = False):

    """
    Description of objective functions:

    These objective functions rely heavily on the concept of "subarrays". These subarrays are
    subsets of elements within the larger collection of elements that are active for any given 
    frequency band. This concept is important, as only certain subarrays will be active for a 
    given frequency in order to reduce spatial aliasing due to over-large distances. 

    1. Total Size Penalty: this function attempts to minimize the city-block size of the array. 
        This comes mostly from a need from a logistical standpoint for the array to be the 
        smallest possible.
    2. Count Penalty: this function attempts to maximize the number of available elements for
        each subarray to increase the total array gain at any frequency. 
    3. Min. Aperture Penalty: This function attempts to maximize the size of each subarray's 
        aperture for the design frequency of that subarray. It does so by raising the floor. 
    4. Frequency Density Penalty: This function attempts to standardize the array so that each
        subarray has equivalent coverage, meaning that there are similar amounts of pairs of 
        elements from one subarray to another. 

    Each penalty is normalized to have absolute limits from 0 to 1. This helps with temperature scaling when using simulated annealing. 
    """

    def get_aperture_size(nodes):
        # measure city-block size of sub-arrays
        if len(nodes) < 2: return 0
        subset_points = points[list(nodes)]
        return np.max(scipy.spatial.distance.pdist(subset_points, metric='cityblock'))
    
    # constants
    c = 1460 # m/s
    f_hi = 100 # Hz
    f_lo = 10 # Hz
    df = 10 # Hz
    
    # calculation of point coordinates from state
    n = len(state) // 2 + 2

    freqs = state[:(n-1)]
    angles = state[(n-1):]
    cum_angles = np.cumsum(angles)
    dx = np.zeros(n-1, dtype=np.float32)
    dy = np.zeros(n-1, dtype=np.float32)
    dy[0] = (c / freqs[0]) / 2
    dx[1:] = (c / freqs[1:]) / 2 * np.cos(cum_angles)
    dy[1:] = (c / freqs[1:]) / 2 * np.sin(cum_angles)
    coords = np.zeros((n,2), dtype = np.float32)
    coords[1:,0] = np.cumsum(dx)
    coords[1:,1] = np.cumsum(dy)
    points = np.round(coords, decimals=2)

    # # get total size and apply to penalty
    # total_height = (np.max(points[:, 1]) - np.min(points[:, 1])) / (c / f_lo / 2)
    # total_width = (np.max(points[:, 0]) - np.min(points[:, 0])) / (c / f_lo / 2)
    # penalty_total_size = 2 / (1 + np.exp(-(total_height + total_width))) - 1

    # determine unique distances and therefore all possible band-limits
    dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(points))
    unique_distances = np.sort(np.unique(dist_matrix)) 
    unique_distances = unique_distances[unique_distances >= 0.01]
    f_cutoff = c / (2 * unique_distances) # sorted from highest to lowest

    # # use distances to determine the evenness of frequency coverage across the band of interest
    # bins = c / (2 * np.arange(f_lo, f_hi + df, df)) # include right-most edge
    # bins = np.flip(bins) # monotonically increasing
    # d_hist, _ = np.histogram(unique_distances, bins=bins)
    # density_value = np.std(d_hist)/np.mean(d_hist) if np.mean(d_hist) != 0 else 0
    # penalty_f_density = density_value / (density_value + 0.5)

    # minimize distances outside of desired range
    num_outside = np.sum((unique_distances < c / (2 * f_hi)) | (unique_distances > c / (2 * f_lo)))
    penalty_outside_range = num_outside / (num_outside + n/2)

    # determine subarrays for each band
    active_elements = np.full((n, len(f_cutoff)-1), False)
    for band_id in range(len(f_cutoff) - 1): 
        d = unique_distances[band_id + 1]
        adj_matrix = (dist_matrix <= d).astype(int)
        n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False)
        
        # find largest aperture size
        best_aperture = -1
        largest_subarray_mask = None
        for label_id in range(n_components):
            current_mask = (labels == label_id)
            nodes = np.where(current_mask)[0]
            ap_size = get_aperture_size(nodes)
            
            if ap_size > best_aperture:
                best_aperture = ap_size
                largest_subarray_mask = current_mask

        active_elements[:, band_id] = largest_subarray_mask

    unique_subarrays, _, mapping = np.unique(active_elements, axis=1, return_index=True, return_inverse=True)

    height_data = np.zeros(unique_subarrays.shape[1])
    width_data = np.zeros(unique_subarrays.shape[1])
    f_array = np.zeros(unique_subarrays.shape[1])
    count_array = np.zeros(unique_subarrays.shape[1])
    for i in range(unique_subarrays.shape[1]): 
        
        f_array[i] = f_cutoff[np.where(mapping == i)[0][0] + 1]
        count_array[i] = np.sum(unique_subarrays[:, i])
        
        f_lo = f_cutoff[np.where(mapping == i)[0][-1] + 1]
        element_mask = unique_subarrays[:, i].flatten()
        selected_elements = points[element_mask]

        aperture_width = np.max(selected_elements[:,0]) - np.min(selected_elements[:,0])
        aperture_height = np.max(selected_elements[:,1]) - np.min(selected_elements[:,1])
        height_data[i] = aperture_height / (c/f_lo)
        width_data[i] = aperture_width / (c/f_lo)

    if np.all(height_data == 0):
        penalty_aperture_size = 1
    else:
        size_value = np.min(height_data) + np.min(width_data)
        penalty_aperture_size = 1.0 / (size_value + 2.0)

    if len(count_array) > 2 and len(count_array[(f_array <= f_hi) & (f_array >= f_lo)]) > 0:
        penalty_count = 1 - np.min(count_array[(f_array <= f_hi) & (f_array >= f_lo)]) / np.max(count_array) 
    else:
        penalty_count = 1

    return np.array([penalty_outside_range, penalty_count, penalty_aperture_size])

def objective_function_slow(state):

    def get_aperture_size(nodes):
        # measure city-block size of sub-arrays
        if len(nodes) < 2: return 0
        subset_points = coords[list(nodes)]
        return np.max(scipy.spatial.distance.pdist(subset_points, metric='cityblock'))

    # constants
    c = 1460 # m/s
    f_hi = 200 # Hz
    f_lo = 10 # Hz
    df = 10 # Hz
    steer_de = np.radians(np.arange(0, 90, 10))

    num_elements = len(state) // 2 + 2
    coords = np.zeros((num_elements, 2), dtype=np.float32)
    coords[1, :] = [0, state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
    curr_angle = 0
    for i in range(2, num_elements):
        curr_angle += state[(num_elements - 1)+ (i - 2)]
        coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)
    coords = np.round(coords, decimals = 2) # round to 2 decimals to align with centimeter spaced grid

        # get total size and apply to penalty
    total_height = (np.max(coords[:, 1]) - np.min(coords[:, 1])) / (c / f_lo / 2)
    total_width = (np.max(coords[:, 0]) - np.min(coords[:, 0])) / (c / f_lo / 2)
    penalty_total_size = 2 / (1 + np.exp(-(total_height + total_width))) - 1

    # geometric penalty
    # determine unique distances and therefore all possible band-limits
    dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
    unique_distances = np.sort(np.unique(dist_matrix)) 
    unique_distances = unique_distances[unique_distances >= 0.01]
    f_cutoff = c / (2 * unique_distances) # sorted from highest to lowest

    # use distances to determine the evenness of frequency coverage across the band of interest
    bins = c / (2 * np.arange(f_lo, f_hi + df, df)) # include right-most edge
    bins = np.flip(bins) # monotonically increasing
    d_hist, _ = np.histogram(unique_distances, bins=bins)
    penalty_f_density = 2 / (1 + np.exp(-np.std(d_hist)/np.mean(d_hist))) - 1

        # determine subarrays for each band
    active_elements = np.full((num_elements, len(f_cutoff)-1), False)
    for band_id in range(len(f_cutoff) - 1): 
        d = unique_distances[band_id + 1]
        adj_matrix = (dist_matrix <= d).astype(int)
        n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False)
        
        # find largest aperture size
        best_aperture = -1
        largest_subarray_mask = None
        for label_id in range(n_components):
            current_mask = (labels == label_id)
            nodes = np.where(current_mask)[0]
            ap_size = get_aperture_size(nodes)
            
            if ap_size > best_aperture:
                best_aperture = ap_size
                largest_subarray_mask = current_mask

        active_elements[:, band_id] = largest_subarray_mask

    bands = active_elements.T 
    unique_configs, _, mapping = np.unique(bands, axis=0, return_index=True, return_inverse=True)
    unique_subarrays = unique_configs.T

    height_data = np.zeros(unique_subarrays.shape[1])
    width_data = np.zeros(unique_subarrays.shape[1])
    f_array = np.zeros(unique_subarrays.shape[1])
    count_array = np.zeros(unique_subarrays.shape[1])
    for i in range(unique_subarrays.shape[1]): 
        
        f_array[i] = f_cutoff[np.where(mapping == i)[0][0] + 1]
        count_array[i] = np.sum(unique_subarrays[:, i])
        
        f_lo = f_cutoff[np.where(mapping == i)[0][-1] + 1]
        spacing = max(0.01, (c / f_lo) / 2)

        element_mask = unique_subarrays[:, i].flatten()
        selected_elements = coords[element_mask]

        aperture_width = np.max(selected_elements[:,0]) - np.min(selected_elements[:,0])
        aperture_height = np.max(selected_elements[:,1]) - np.min(selected_elements[:,1])
        height_data[i] = aperture_height / spacing
        width_data[i] = aperture_width / spacing

    if np.all(height_data == 0):
        penalty_aperture_min = 1
    else:
        penalty_aperture_min = np.exp(-(np.min(height_data) + np.min(width_data))) # [1, 0]

    if len(count_array) > 2:
        med_ind = np.argmin(np.abs(np.diff(count_array - np.mean(count_array)))) # round down
        penalty_count = 1 - count_array[med_ind] / np.max(count_array) # * f_array[med_ind] / max(np.max(f_array), 1e-9)
    else:
        penalty_count = 1

    # DI and MSLL calculations
    if len(np.unique(coords, axis = 0)) < num_elements:
        unq_ind = np.unique(coords, axis = 0, return_index = True)[1]
        coords = coords[np.sort(unq_ind)]

    X = np.zeros(coords.shape[0], dtype=np.float32)
    Y = coords[:, 0].flatten()
    Z = coords[:, 1].flatten()
    array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
    bf_model = BeamformingModel(array, c = 1460)

    bands = bf_model.active_elements
    unique_subarrays, first_occurrence, mapping = np.unique(bands, axis=1, return_index=True, return_inverse=True)
    subarray_mask = np.sum(unique_subarrays, axis = 0) > 2
    valid_inds = np.where(subarray_mask)[0]
    f = []
    for i in valid_inds: 
        f_hi = bf_model.f_cutoff[first_occurrence[i]+1]
        f_lo = bf_model.f_cutoff[np.where(mapping == i)[0][-1] + 1]
        if f_hi != f_lo:
            f.append((f_hi - f_lo)/2 + f_lo)

    if len(coords) < 2:
        di_f = np.ones(len(f)) * 1.761 # di of a single dipole element
        # hpbw = np.ones((len(f), 3)) * np.radians(30) # HPBW of a single dipole element
        di_de = np.ones(len(steer_de)) * 1.761
        msll_f = -np.ones(len(f))  # include the main lobe in the side lobe
        msll_de = -np.ones(len(steer_de))
    else:
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

    penalty_di_f = np.exp(-np.min(di_f)) 
    penalty_side_lobe_f = np.exp(np.max(msll_f))
    penalty_di_de_variance = 2 / ( 1 + np.exp(-np.std(di_de))) - 1 # [0 -> 1], lower is better

    return np.array([penalty_f_density, penalty_di_f, penalty_side_lobe_f, penalty_di_de_variance, penalty_total_size, penalty_count, penalty_aperture_min])

def search_scaling(state, state_inc):

    # scaling occurs only for angles
    # each angle has an effect equivalent to how much extent that it rotates
    # scale this effect so that it's normalized to the effect of the last angle
    # in other words, each angle should have similar effect on the search space

    theta = state[len(state) // 2 + 1:] # for 16 elements, this should be 14 angles
    d = state[1:len(state) // 2 + 1]
    scaling = np.ones_like(state)
    r = d * np.cos(np.flip(np.cumsum(np.flip(theta))))
    theta_inc_prime = np.abs(np.arctan(0.01 / (np.maximum(np.abs(r), 1e-6))))
    scaling[len(state) // 2 + 1:] = theta_inc_prime[-1] / theta_inc_prime

    if not np.isfinite(scaling).all():
        print("CRITICAL: Scaling contains non-finite values!")
        print(f"r values: {r}")
        print(f"theta_inc_prime: {theta_inc_prime}")
        # Identify if the issue is a 0/0 or inf/inf case

    return scaling

if __name__ == "__main__":

    np.seterr(all='raise')
    np.random.seed(13)
    num_elements = 31

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_'+now)

    f_min = 10
    f_max = 100
    c = 1460

    f_lims = np.tile([[f_min, f_max]], (num_elements-1, 1)) # make sure an element point isn't rounded above d_max
    theta_lims = np.tile([[np.radians(0), np.radians(359.5)]], (num_elements-2, 1))
    theta_lims[0,:] = [np.radians(0), np.radians(180)] # limit the angle from the second element to avoid symmetric duplicates of the same array geometry

    f_inc = np.tile(f_min, num_elements-1)
    th_inc = np.tile(np.radians(5), num_elements-2)

    f_0 = np.random.uniform(f_lims[:, 0], f_lims[:, 1], size = num_elements-1)
    th_0 = np.random.uniform(theta_lims[:, 0], theta_lims[:, 1], size = num_elements-2) # element 2 is always on y-axis

    initial_state = np.concatenate((f_0, th_0))
    state_lims = np.concatenate((f_lims, theta_lims)).T
    state_inc = np.concatenate((f_inc, th_inc))

    circular_inds = np.full_like(initial_state, False, dtype = bool)
    circular_inds[len(f_0) + 2:] = True # angles are circular variables

    opt = SAOptimization(func = objective_function_fast, ndim = np.shape(state_inc)[0], state_lims = state_lims, state_inc = state_inc, search_scaling_func = search_scaling, circular_inds=circular_inds)
    opt.optimize()

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

    np.savez(save_file, coords, opt.accepted_states, opt.accepted_energies, opt.global_min_energy, opt.pareto_optimal_states, opt.pareto_optimal_values)

    plt.show()

