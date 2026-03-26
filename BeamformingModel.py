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
    def __init__(self, X, Y, Z, eX = None, eY = None, eZ = None, element_directivity: ElementDirectivity = ElementDirectivity.OMNI):

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
        print(self.element_de*180/np.pi)
        print(self.element_az*180/np.pi)

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

        DE = DE[:, :, np.newaxis] # shape (len(AZ), len(DE), 1)
        AZ = AZ[:, :, np.newaxis] # shape (len(AZ), len(DE), 1)
        element_de = self.element_de[np.newaxis, np.newaxis, :] # shape (1, 1, num_elements)
        element_az = self.element_az[np.newaxis, np.newaxis, :] # shape (1, 1, num_elements)

        if self.element_directivity == ElementDirectivity.OMNI:
            return np.ones((AZ.shape[0], AZ.shape[1], len(self.X))) 
        
        elif self.element_directivity == ElementDirectivity.DIPOLE:
            # dipole oriented along the x-axis of the element
            return np.abs(np.cos(DE - element_de) * np.cos(AZ - element_az))
        
        elif self.element_directivity == ElementDirectivity.BAFFLED_DIPOLE:
            # baffled dipole oriented along the x-axis of the element
            return np.abs(np.cos(DE - element_de) * np.cos(AZ - element_az)) * (DE - element_de >= 0)
        
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

class BeamformingModel:
    def __init__(self, array):
        self.array = array

    def compute_steering_vector(self, steer_az, steer_de, frequency):
        c = 343 # speed of sound in m/s
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        element_positions = np.stack((self.array.X, self.array.Y, self.array.Z), axis=-1)
        direction = np.array([np.cos(steer_de),
                              np.sin(steer_az) * np.sin(steer_de),
                              np.cos(steer_az) * np.sin(steer_de)])

        path_lengths = np.dot(element_positions, direction)
        steering_vector = np.exp(-1j * k * path_lengths)

        return steering_vector
    
    def compute_element_vector(self, plane_wave_direction, frequency):
        c = 343 # speed of sound in m/s
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        element_positions = np.concatenate((self.array.X[:, np.newaxis], self.array.Y[:, np.newaxis], self.array.Z[:, np.newaxis]), axis=1)

        element_vector = np.zeros(len(element_positions), dtype=complex)
        for i, mic in enumerate(element_positions):
            distance = np.dot(mic, plane_wave_direction)
            element_vector[i] = np.exp(-1j * k * distance)

        return element_vector
    
    def compute_shading_vector(self, plane_wave_directions, frequency):
        # this will contain a formulation of the shading vector use to control sidelobes and mainlobe width, for now we will use a simple rectangular window (no shading)
        return np.ones(len(self.array.X), dtype=complex)
    
    def compute_beampattern(self, frequency, delta_az = 0.25, delta_de = 0.25, steer_az = 0, steer_de = 0, source_distance=1000):
        c = 343 # speed of sound in m/s
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        az = np.radians(np.arange(-90, 90, delta_az)) # degrees
        de = np.radians(np.arange(-90, 90, delta_de)) # degrees
        AZ, DE = np.meshgrid(az, de, indexing = 'ij')

        # i = Az, j = De, k = XYZ
        directions = np.stack((np.cos(DE),
                               np.sin(AZ) * np.sin(DE), 
                               np.cos(AZ) * np.sin(DE)), axis=-1) # shape (len(az), len(de), 3)
        # m = element index, k = XYZ
        element_positions = np.stack([self.array.X, self.array.Y, self.array.Z], axis=1) # shape (num_elements, 3)
        
        phases = np.einsum('ijk,mk->ijm', directions, element_positions) # shape (num_elements, len(az)*len(de))
        element_directivity = self.array.compute_element_directivity(AZ, DE) # shape (len(az), len(de))
        manifold_vector = element_directivity * np.exp(-1j * k * phases) # shape ( len(az), len(de), num_elements)

        steering_vector = self.compute_steering_vector(steer_az, steer_de, frequency) # shape (num_elements,)
        shading_vector = self.compute_shading_vector(directions, frequency) # shape (num_elements,)

        beampattern = np.sum(manifold_vector * \
                             steering_vector.conj() * \
                                shading_vector, axis = 2)

        return az, de, beampattern
    
    def plot_beampattern(self, frequency, delta_az = 0.25, delta_de = 0.25, steer_az = 0, steer_de = 0, source_distance=1000):
        az, de, beampattern = self.compute_beampattern(frequency, delta_az, delta_de, steer_az, steer_de, source_distance)
        AZ, DE = np.meshgrid(az, de, indexing = 'ij')
        beampattern = beampattern / np.max(np.abs(beampattern)) # normalize the beampattern

        fig, ax = plt.subplots(1, 3, figsize=(15, 7))
        
        im = ax[0].imshow(10 * np.log10(np.abs(beampattern.T)), extent=(np.degrees(np.min(az)), np.degrees(np.max(az)), np.degrees(np.min(de)), np.degrees(np.max(de))), aspect='auto', origin='lower', vmin = -50, vmax = 0)
        fig.colorbar(im, ax=ax[0], label='Beamforming Gain (dB)')
        ax[0].contour(10 * np.log10(np.abs(beampattern.T)), levels=[-3], colors='red', extent=(np.degrees(np.min(az)), np.degrees(np.max(az)), np.degrees(np.min(de)), np.degrees(np.max(de))), linewidths=2)
        ax[0].set_xlabel('Azimuth (degrees)')
        ax[0].set_ylabel('Elevation (degrees)')
        ax[0].set_title(f'Beamforming Pattern at {frequency} Hz')
        ax[0].grid()

        ax[1].plot(np.degrees(az), 10 * np.log10(np.abs(beampattern[:, len(de) // 2])), label='Elevation = 0°')
        ax[1].set_ylim(-50, 0)
        ax[1].set_xlabel('Azimuth (degrees)')
        ax[1].set_ylabel('Beamforming Gain (dB)')
        ax[1].set_title(f'Beamforming Pattern at {frequency} Hz (Elevation = 0°)')
        ax[1].grid()

        # convert to cartesian coordinates for 3D plot
        R = 10 * np.log10(np.abs(beampattern)) + 50
        R[R < 0] = 0 # set negative values to 0 for better visualization
        X = R * np.cos(DE)
        Y = R * np.sin(AZ) * np.sin(DE)
        Z = R * np.cos(AZ) * np.sin(DE)

        my_cmap = cm.get_cmap('turbo') # Choose a vibrant colormap
        colors = my_cmap(R) # colors.shape is now (360, 360, 4) - RGBA        

        ax[2] = plt.subplot(133, projection='3d')
        ax[2].plot_surface(X, Y, Z, facecolors = colors)
        ax[2].set_title(f'3D Beamforming Pattern at {frequency} Hz')
        ax[2].set_xlabel('X')
        ax[2].set_ylabel('Y')
        ax[2].set_zlabel('Z')
        ax[2].set_xlim(-50, 50)
        ax[2].set_ylim(-50, 50)
        ax[2].set_zlim(-50, 50)
        ax[2].grid()


if __name__ == "__main__":
    # Example usage`
    d = 343/2e3/4 # element spacing of 1/4 wavelength at 2 kHz
    y, z = np.meshgrid(np.linspace(-2*d, 2*d, 4), np.linspace(-2*d, 2*d, 4))
    Y = np.ravel(y)
    Z = np.ravel(z)
    bf_array = BeamformingArray(X=np.zeros(16),
                                Y=Y,
                                Z=Z, 
                                element_directivity=ElementDirectivity.DIPOLE)
    bf_array.plot_array_geometry()

    model = BeamformingModel(bf_array)
    model.plot_beampattern(2e3, delta_az = 0.5, delta_de = 0.5) 

    plt.show()