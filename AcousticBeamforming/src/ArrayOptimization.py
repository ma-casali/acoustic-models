import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.special import factorial
import networkx as nx
import sys
import os

from BeamformingArray import BeamformingArray, ElementDirectivity
from BeamformingModel import BeamformingModel
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../acoustic-models'))
from SimulatedAnnealing import SAOptimization

# This is a problem that is perfect for simulated annealing, since the objective function is discrete valued and non-differentiable

num_elements = 12
Y_lims = np.tile(np.array([-1, 1]), (num_elements, 1))
Z_lims = np.tile(np.array([-1, 1]), (num_elements, 1))
Y_0 = np.random.uniform(Y_lims[:, 0], Y_lims[:, 1], size = num_elements)
Z_0 = np.random.uniform(Z_lims[:, 0], Z_lims[:, 1], size = num_elements)

coords = np.column_stack((Y_0, Z_0))
sorted_indices = np.lexsort((coords[:,1], coords[:,0]))
Y_0 = coords[sorted_indices][:, 0]
Z_0 = coords[sorted_indices][:, 1]

initial_state = np.concatenate((Y_0, Z_0))
state_lims = np.concatenate((Y_lims, Z_lims)).T
state_inc = np.tile(0.01, num_elements * 2)

def objective_function(state):

    Y = state[:num_elements]
    Z = state[num_elements:]
    max_aperture_size = 2

    points = np.column_stack((Y, Z))
    tree = cKDTree(points)

    penalty = 0
    for freq in 3*np.logspace(2, 3, base = 10, num = 20):
        spacing = 343 / freq / 2
        pairs = tree.query_pairs(r = spacing)

        G = nx.Graph()
        G.add_nodes_from(range(num_elements))
        G.add_edges_from(pairs)

        components = list(nx.connected_components(G))

        max_ind = np.argmax([len(c) for c in components])

        connected_points = points[list(components[max_ind])]
        aperture_size = (np.max(connected_points[:, 0]) - np.min(connected_points[:, 0])) + (np.max(connected_points[:, 1]) - np.min(connected_points[:, 1]))
        N = len(connected_points)

        penalty += (num_elements - N)/num_elements
        penalty += np.abs((max_aperture_size - aperture_size)/max_aperture_size)

    # normalize penalty to be between 0 and 1
    penalty /= 2 * 20

    return penalty

opt = SAOptimization(objective_function, state_0 = initial_state, state_lims = state_lims, state_inc = state_inc)
optimal_state, optimal_value = opt.optimize()

Y = optimal_state[:num_elements]
Z = optimal_state[num_elements:]
plt.scatter(Y, Z)

print("Optimal state (Y): ", [float(np.round(y, 2)) for y in Y])
print("Optimal state (Z): ", [float(np.round(z, 2)) for z in Z])

opt.plot_results()
plt.show()

