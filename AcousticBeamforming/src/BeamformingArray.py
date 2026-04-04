import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from enum import Enum

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

    def compute_element_directivity(self, AZ, DE, custom_directivity_function=None):
        # computes the directivity of the element at the angles specified by AZ and DE (in radians)
        # This is assumed to be in the far-field so that: 
        #   - R (distance from source to element) >> array size 
        #   - AZ and DE are ~= the angles of the incoming wave relative to the element's local coordinate system (eX, eY, eZ)
        # returns an array of shape (len(AZ), len(DE), num_elements) containing the directivity values for each angle

        if len(AZ.shape) == 2:
            DE = DE[:, :, np.newaxis] # shape (len(AZ), len(DE), 1)
            AZ = AZ[:, :, np.newaxis] # shape (len(AZ), len(DE), 1)
            element_de = self.element_de[np.newaxis, np.newaxis, :] # shape (1, 1, num_elements)
            element_az = self.element_az[np.newaxis, np.newaxis, :] # shape (1, 1, num_elements)
        elif len(AZ.shape) == 1:
            DE = DE[:, np.newaxis] # shape (len(DE), 1)
            AZ = AZ[:, np.newaxis] # shape (len(AZ), 1)
            element_de = self.element_de[np.newaxis, :] # shape (1, num_elements)
            element_az = self.element_az[np.newaxis, :] # shape (1, num_elements)
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
