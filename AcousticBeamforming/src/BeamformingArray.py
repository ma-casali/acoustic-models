import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from enum import Enum
import networkx as nx
import scipy
import cv2

class ElementDirectivity(Enum):
    OMNI = 'omni'
    DIPOLE = 'dipole'
    BAFFLED_DIPOLE = 'baffled_dipole'
    CUSTOM = 'custom'

class BeamformingArray:
    def __init__(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, eX: np.ndarray = None, eY: np.ndarray = None, eZ: np.ndarray = None, design_frequency: float = 2000, element_directivity: ElementDirectivity = ElementDirectivity.OMNI):

        self.design_frequency = design_frequency

        # element positions
        self.X = X
        self.Y = Y
        self.Z = Z

        # element directivity
        if not isinstance(element_directivity, ElementDirectivity):
            raise ValueError("element_directivity must be an instance of ElementDirectivity Enum")
        self.element_directivity = element_directivity

        # element axes
        if eX is None:
            self.eX = np.tile(np.array([1, 0, 0]), (len(self.X), 1))
            self.eY = np.tile(np.array([0, 1, 0]), (len(self.Y), 1))
            self.eZ = np.tile(np.array([0, 0, 1]), (len(self.Z), 1))
        else:
            self.eX = eX
            self.eY = eY
            self.eZ = eZ
        
        # get element de and az angles of rotation
        self.element_de = np.arctan2(self.eX[:, 2], self.eX[:, 0])
        self.element_az = np.arctan2(self.eX[:, 1], self.eX[:, 0]) 

        # confirm that element axes are orthogonal
        for i in range(len(self.X)):
            assert np.isclose(np.dot(self.eX[i], self.eY[i]), 0), "eX and eY are not orthogonal for element {}".format(i)
            assert np.isclose(np.dot(self.eX[i], self.eZ[i]), 0), "eX and eZ are not orthogonal for element {}".format(i)
            assert np.isclose(np.dot(self.eY[i], self.eZ[i]), 0), "eY and eZ are not orthogonal for element {}".format(i)

    def compute_element_directivity(self, AZ, DE, custom_directivity_function=None, element_mask: np.ndarray = None):
        # computes the directivity of the element at the angles specified by AZ and DE (in radians)
        # This is assumed to be in the far-field so that: 
        #   - R (distance from source to element) >> array size 
        #   - AZ and DE are ~= the angles of the incoming wave relative to the element's local coordinate system (eX, eY, eZ)
        # returns an array of shape (len(AZ), len(DE), num_elements) containing the directivity values for each angle

        if element_mask is None:
            element_mask = np.ones(len(self.X), dtype=bool)
        else:
            if len(element_mask) != len(self.X):
                raise ValueError("element_mask must have the same length as the number of elements in the array")
            element_mask = np.array(element_mask, dtype=bool)

        if len(AZ.shape) == 2:
            DE = DE[:, :, np.newaxis] # shape (len(AZ), len(DE), 1)
            AZ = AZ[:, :, np.newaxis] # shape (len(AZ), len(DE), 1)
            element_de = self.element_de[np.newaxis, np.newaxis, element_mask] # shape (1, 1, num_elements)
            element_az = self.element_az[np.newaxis, np.newaxis, element_mask] # shape (1, 1, num_elements)
        elif len(AZ.shape) == 1:
            DE = DE[:, np.newaxis] # shape (len(DE), 1)
            AZ = AZ[:, np.newaxis] # shape (len(AZ), 1)
            element_de = self.element_de[np.newaxis, element_mask] # shape (1, num_elements)
            element_az = self.element_az[np.newaxis, element_mask] # shape (1, num_elements)
        else:
            raise ValueError("AZ and DE must be either 1D or 2D arrays")
        
        if self.element_directivity == ElementDirectivity.OMNI:
            return np.ones((AZ.shape[0], AZ.shape[1], len(self.X))) 
        
        elif self.element_directivity == ElementDirectivity.DIPOLE:
            # dipole oriented along the x-axis of the element
            return np.abs(np.cos(DE - element_de))
        
        elif self.element_directivity == ElementDirectivity.BAFFLED_DIPOLE:
            # baffled dipole oriented along the x-axis of the element
            return np.abs(np.cos(DE - element_de)) * (DE - element_de >= 0)
        
        elif self.element_directivity == ElementDirectivity.CUSTOM:
            if custom_directivity_function is None:
                raise ValueError("custom_directivity_function must be provided for CUSTOM directivity")
            return custom_directivity_function(AZ, DE)
        else:
            raise ValueError("Invalid element directivity type")
        
    def plot_array_geometry(self):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.scatter(self.X, self.Y, self.Z, c='b', marker='o')

        # get maximum dimension of the array for scaling the vectors
        max_dim = np.max(np.sqrt(self.X**2 + self.Y**2 + self.Z**2))
        vector_length = max_dim * 0.1 # scale the vectors to be 10% of the maximum dimension of the array

        ax.quiver(self.X, self.Y, self.Z, self.eX[:, 0], self.eX[:, 1], self.eX[:, 2], length=vector_length, color='r', label='eX')
        ax.quiver(self.X, self.Y, self.Z, self.eY[:, 0], self.eY[:, 1], self.eY[:, 2], length=vector_length, color='g', label='eY')
        ax.quiver(self.X, self.Y, self.Z, self.eZ[:, 0], self.eZ[:, 1], self.eZ[:, 2], length=vector_length, color='b', label='eZ')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Array Geometry')

        ax.set_xlim(min(np.min(self.X), -max_dim), max(np.max(self.X), max_dim))
        ax.set_ylim(min(np.min(self.Y), -max_dim), max(np.max(self.Y), max_dim))
        ax.set_zlim(min(np.min(self.Z), -max_dim), max(np.max(self.Z), max_dim))

    def plot_array_connections(self, fig = None, ax = None, f = None, c = None):

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if f is None:
            f = 3 * np.logspace(2, 3, num = 10) # generic frequencies (used in optimization)
        if c is None:
            c = 343

        lam = c / f
        spacing = lam / 2 # half wavelength max. spacing for each frequency

        points = np.column_stack((self.Y, self.Z)) # shape (num_elements, 3)
        tree = scipy.spatial.cKDTree(points)

        cmap = plt.get_cmap('turbo')
        cmap_vals = cmap(np.linspace(0, 1, len(self.X)))
        for n, d in enumerate(spacing):
            pairs = tree.query_pairs(r = d)
            filtered_pairs = set()
            for (i, j) in pairs:
                dist = np.linalg.norm(points[i] - points[j])
                # only show connections that are at least 1/5 of the wavelength apart to avoid apertures that are too small compared to a wavelength
                if dist >= lam[n] / 5: 
                    filtered_pairs.add((i, j))

            G = nx.Graph()
            G.add_nodes_from(range(len(self.X)))
            G.add_edges_from(filtered_pairs)
            components = list(nx.connected_components(G))
            max_ind = np.argmax([len(c) for c in components])
            connected_points = points[list(components[max_ind])]
            connected_points = np.array(connected_points, dtype=np.float32)
            
            rect = cv2.minAreaRect(connected_points)
            print(f"Frequency: {f[n]:.0f} Hz, Connected Sensors: {components[max_ind]}, Aperture Width, Height / lambda: {rect[1][0]/lam[n]:.2f}, {rect[1][1]/lam[n]:.2f}")

            iter = 0
            for (i, j) in filtered_pairs:
                if iter == 0:
                    # no alpha used, so that only the shortest connection is shown
                    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color=cmap_vals[n, :], label = f'{f[n]:.0f} Hz spacing: {len(connected_points)}')
                else:
                    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color=cmap_vals[n, :])
                   
                iter += 1

        ax.scatter(self.Y, self.Z, c = 'k', marker = 'o')
        ax.set_aspect('equal')
        ax.legend()