import numpy as np
import matplotlib.pyplot as plt

class BeamformingArray:
    def __init__(self, height = 0, width = 0.1, length = 0.1, nX = 10, nY = 10, nZ = 1):
        
        # dimensions of the array
        self.Height = height # m
        self.Width = width # m
        self.Length = length # m

        # number of elements in each dimension
        self.nX = nX
        self.nY = nY
        self.nZ = nZ

        # element positions
        [self.X, self.Y, self.Z] = self.element_positions()

    def element_positions(self):
        x = np.linspace(-self.Width/2, self.Width/2, self.nX)
        y = np.linspace(-self.Length/2, self.Length/2, self.nY)
        z = np.linspace(-self.Height/2, self.Height/2, self.nZ) 
        X, Y, Z = np.meshgrid(x, y, z)
        return X.flatten(), Y.flatten(), Z.flatten()

class BeamformingModel:
    def __init__(self, array):
        self.array = array

    def compute_steering_vector(self, source_pos, frequency):
        c = 343 # speed of sound in m/s
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        element_positions = np.concatenate((self.array.X[:, np.newaxis], self.array.Y[:, np.newaxis], self.array.Z[:, np.newaxis]), axis=1)

        steering_vector = np.zeros(len(element_positions), dtype=complex)
        for i, mic in enumerate(element_positions):
            distance = np.linalg.norm(source_pos - mic)
            steering_vector[i] = np.exp(1j * k * distance)

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
    
    def compute_shading_vector(self, plane_wave_direction, frequency):
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
        directions = np.stack((np.cos(AZ) * np.cos(DE), np.sin(AZ) * np.cos(DE), np.sin(DE)), axis=-1) # shape (len(az), len(de), 3)
        # m = element index, k = XYZ
        element_positions = np.stack([self.array.X, self.array.Y, self.array.Z], axis=1) # shape (num_elements, 3)
        
        phases = np.einsum('ijk,mk->ijm', directions, element_positions) # shape (num_elements, len(az)*len(de))
        element_vectors = np.exp(-1j * k * phases) # shape ( len(az), len(de), num_elements)

        steering_direction = np.array([np.cos(np.radians(steer_az)) * np.cos(np.radians(steer_de)),
                                        np.sin(np.radians(steer_az)) * np.cos(np.radians(steer_de)),
                                        np.sin(np.radians(steer_de))]) # shape (3,)
        steering_vector = self.compute_steering_vector(steering_direction * source_distance, frequency) # shape (num_elements,)
        shading_vector = self.compute_shading_vector(steering_direction, frequency) # shape (num_elements,)

        beampattern = np.sum(element_vectors * steering_vector.conj() * shading_vector, axis = 2)

        return az, de, beampattern
    
    def plot_beampattern(self, frequency, delta_az = 0.25, delta_de = 0.25, steer_az = 0, steer_de = 0, source_distance=1000):
        az, de, beampattern = self.compute_beampattern(frequency, delta_az, delta_de, steer_az, steer_de, source_distance)
        beampattern = beampattern / np.max(np.abs(beampattern)) # normalize the beampattern

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        
        im = ax[0].imshow(10 * np.log10(np.abs(beampattern.T)), extent=(np.degrees(np.min(az)), np.degrees(np.max(az)), np.degrees(np.min(de)), np.degrees(np.max(de))), aspect='auto', origin='lower')
        fig.colorbar(im, ax=ax[0], label='Beamforming Gain (dB)')
        ax[0].contour(10 * np.log10(np.abs(beampattern.T)), levels=[-3], colors='red', extent=(np.degrees(np.min(az)), np.degrees(np.max(az)), np.degrees(np.min(de)), np.degrees(np.max(de))), linewidths=2)
        ax[0].set_xlabel('Azimuth (degrees)')
        ax[0].set_ylabel('Elevation (degrees)')
        ax[0].set_title(f'Beamforming Pattern at {frequency} Hz')
        ax[0].grid()

        ax[1].plot(np.degrees(az), 10 * np.log10(np.abs(beampattern[:, len(de) // 2])), label='Elevation = 0°')
        ax[1].set_xlabel('Azimuth (degrees)')
        ax[1].set_ylabel('Beamforming Gain (dB)')
        ax[1].set_title(f'Beamforming Pattern at {frequency} Hz (Elevation = 0°)')
        ax[1].grid()

        plt.show()

if __name__ == "__main__":
    bf_array = BeamformingArray(height=0, width=0.3, length=0.3, nX=4, nY=4, nZ=1)
    model = BeamformingModel(bf_array)
    model.plot_beampattern(2e3, delta_az = 0.5, delta_de = 0.5) 