import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.special import factorial
import networkx as nx
import sys
import os
import cv2

from BeamformingArray import BeamformingArray, ElementDirectivity
from BeamformingModel import BeamformingModel
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../acoustic-models'))
from SimulatedAnnealing import SAOptimization

# This is a problem that is perfect for simulated annealing, since the objective function is discrete valued and non-differentiable

num_elements = 16
Y_lims = np.tile(np.array([-1, 1]), (num_elements, 1))
Z_lims = np.tile(np.array([-1, 1]), (num_elements, 1))
Y_0 = np.random.uniform(Y_lims[:, 0], Y_lims[:, 1], size = num_elements)
Z_0 = np.random.uniform(Z_lims[:, 0], Z_lims[:, 1], size = num_elements)

d_min = 343 / 3000 / 2 # half wavelength at 3 kHz
d_max = 343 / 300 / 2 # half wavelength at 300 Hz
d_lims = np.tile([[d_min, d_max]], (num_elements, 1))
theta_lims = np.tile([[np.radians(-179.5), np.radians(180)]], (num_elements, 1))
d_inc = np.tile(d_min/4, num_elements)
th_inc = np.tile(np.radians(0.5), num_elements)
d_0 = np.random.uniform(d_lims[:, 0], d_lims[:, 1], size = num_elements)
th_0 = np.random.uniform(theta_lims[:, 0], theta_lims[:, 1], size = num_elements)

initial_state = np.concatenate((d_0, th_0))
state_lims = np.concatenate((d_lims, theta_lims)).T
state_inc = np.concatenate((d_inc, th_inc))

def objective_function(state, optimal_state = False):

    def sigmoid_function(a, b, c, x):
        return a / (1 + np.exp(-b * x)) + c
    
    gamma_std = 10
    gamma_mean = 1
    gamma_balance = 5

    num_elements = len(state) // 2
    coords = np.zeros((num_elements, 2), dtype=np.float32)
    curr_angle = state[num_elements + 1]
    coords[1,:] = [state[0] * np.cos(curr_angle), state[0] * np.sin(curr_angle)]
    
    for i in range(2, num_elements):
        curr_angle += state[num_elements + i - 1]
        coords[i, 0] = coords[i-1, 0] + state[i-1] * np.cos(curr_angle)
        coords[i, 1] = coords[i-1, 1] + state[i-1] * np.sin(curr_angle)

    tree = cKDTree(coords)

    penalty_connected_elements = 0
    penalty_aperture_size = 0
    height_data = []
    width_data = []
    for freq in 3*np.logspace(2, 3, base = 10, num = 100):
        spacing = 343 / freq / 2
        max_aperture_size = spacing * num_elements

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
        penalty_connected_elements += (num_elements - N)/num_elements
    
        rect = cv2.minAreaRect(connected_points)
        aperture_width = rect[1][0]
        aperture_height = rect[1][1]
        penalty_aperture_size *= (max_aperture_size - aperture_width)/max_aperture_size
        penalty_aperture_size *= (max_aperture_size - aperture_height)/max_aperture_size

        height_val = aperture_height / spacing
        width_val = aperture_width / spacing

        # these values are 1/2 at num_elements and approach 0 as aperture size increases
        height_data.append(sigmoid_function(-2, np.log(3)/num_elements, 2, height_val))
        width_data.append(sigmoid_function(-2, np.log(3)/num_elements, 2, width_val))

    penalty_aperture_size += np.std(height_data) + np.std(width_data) * gamma_std
    penalty_aperture_size += (1 - (np.mean(height_data) + np.mean(width_data)) / 2) * gamma_mean
    penalty_aperture_size += np.max(np.abs(np.array(height_data) - np.array(width_data))) * gamma_balance

    if optimal_state:
        return [
            np.std(height_data) + np.std(width_data) * gamma_std,
            (1 - (np.mean(height_data) + np.mean(width_data)) / 2) * gamma_mean,
            np.max(np.array(height_data) - np.array(width_data)) * gamma_balance
        ]


    # penalty = penalty_connected_elements + penalty_aperture_size
    penalty = penalty_aperture_size

    return penalty

opt = SAOptimization(objective_function, state_0 = initial_state, state_lims = state_lims, state_inc = state_inc)
optimal_state, optimal_value = opt.optimize()

print("Penalty components: ", objective_function(optimal_state, optimal_state = True))

coords = np.zeros((num_elements, 2), dtype=np.float32)
curr_angle = optimal_state[num_elements + 1]
coords[1,:] = [optimal_state[0] * np.cos(curr_angle), optimal_state[0] * np.sin(curr_angle)]
    
for i in range(2, num_elements):
    curr_angle += optimal_state[num_elements + i - 1]
    coords[i, 0] = coords[i-1, 0] + optimal_state[i-1] * np.cos(curr_angle)
    coords[i, 1] = coords[i-1, 1] + optimal_state[i-1] * np.sin(curr_angle)
Y = coords[:, 0]
Z = coords[:, 1]
plt.scatter(Y, Z)

print("Optimal state (Y): ", [(np.round(float(y), 2)) for y in Y])
print("Optimal state (Z): ", [(np.round(float(z), 2)) for z in Z])

opt.plot_results()
plt.show()

