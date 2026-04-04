import numpy as np
from enum import Enum
from BeamformingArray import BeamformingArray, ElementDirectivity
import matplotlib.pyplot as plt

class ArrayShading:

    def __init__(self, array: BeamformingArray):
        self.array = array

    def compute_raised_cosine_window(self, p, dims = [True, True, True]) -> np.ndarray:

        # create a raised cosine window in each dimension of the array and multiply them together to get the final shading vector
        def raised_cosine_function(N, p):
            c = p / N + (1 - p) / 2 * np.sin(np.pi / (2 * N)) # raised cosine (Van Trees)
            return c * (p + (1 - p) * np.cos( np.pi * (np.arange(N) - (N - 1) / 2) / N )) # shape (N,)
        
        w_x = self.get_dim_weights(raised_cosine_function, self.array.X, dims[0], {'p': p})
        w_y = self.get_dim_weights(raised_cosine_function, self.array.Y, dims[1], {'p': p})
        w_z = self.get_dim_weights(raised_cosine_function, self.array.Z, dims[2], {'p': p})

        raised_cosine_window = w_x * w_y * w_z
        raised_cosine_window /= np.max(raised_cosine_window) # normalize to max of 1

        return raised_cosine_window
    
    def compute_kaiser_window(self, beta, dims = [True, True, True]) -> np.ndarray:

        # create a kaiser window in each dimension of the array and multiply them together to get the final shading vector
        def kaiser_function(N, beta):
            n = np.arange(N) - (N - 1) / 2
            return np.i0(beta * np.sqrt(1 - (2 * n / N)**2))
        
        w_x = self.get_dim_weights(kaiser_function, self.array.X, dims[0], {'beta': beta})
        w_y = self.get_dim_weights(kaiser_function, self.array.Y, dims[1], {'beta': beta})
        w_z = self.get_dim_weights(kaiser_function, self.array.Z, dims[2], {'beta': beta})

        kaiser_window = w_x * w_y * w_z
        kaiser_window /= np.max(kaiser_window) # normalize to max of 1

        return kaiser_window
    
    def get_dim_weights(self, func: callable, coords: np.ndarray, do_apply: bool, kwargs) -> np.ndarray:
        if not do_apply:
            return np.ones(len(coords))
        
        unique_coords = np.unique(coords)
        weights = func(len(unique_coords), **kwargs)

        mapping = {val: w for val, w in zip(unique_coords, weights)}
        return np.array([mapping[c] for c in coords])
    
    def plot_weights_on_elements(self, fig, ax, weights):

        x_lims = (np.min(self.array.X), np.max(self.array.X))
        y_lims = (np.min(self.array.Y), np.max(self.array.Y))
        z_lims = (np.min(self.array.Z), np.max(self.array.Z))

        max_lims = [min(x_lims[0], y_lims[0], z_lims[0]), max(x_lims[1], y_lims[1], z_lims[1])]

        sc = ax[0].scatter(self.array.X, self.array.Y, self.array.Z, c=np.abs(weights), cmap='turbo', marker='o')
        fig.colorbar(sc, ax=ax[0], label='Element Weight')
        ax[0].set_xlabel('X (m)')
        ax[0].set_ylabel('Y (m)')
        ax[0].set_zlabel('Z (m)')
        ax[0].set_title('Magnitude Shading of Weights on Elements')
        ax[0].set_xlim(max_lims)
        ax[0].set_ylim(max_lims)
        ax[0].set_zlim(max_lims)

        sc = ax[1].scatter(self.array.X, self.array.Y, self.array.Z, c=np.angle(weights)*180/np.pi, cmap='bwr', marker='o')
        fig.colorbar(sc, ax=ax[1], label='Element Weight')
        ax[1].set_xlabel('X (m)')
        ax[1].set_ylabel('Y (m)')
        ax[1].set_zlabel('Z (m)')
        ax[1].set_title('Phase Shading of Weights on Elements')
        ax[1].set_xlim(max_lims)
        ax[1].set_ylim(max_lims)
        ax[1].set_zlim(max_lims)
    