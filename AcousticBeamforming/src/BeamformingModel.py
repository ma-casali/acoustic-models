from matplotlib import style
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import os
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path

from BeamformingArray import BeamformingArray, ElementDirectivity
from ArrayShading import ArrayShading

def time_varying_signal(array: BeamformingArray, frequency = 400, snr_db = 500, spline_points: int = 10, T: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixture to provide a time-varying signal to elements for testing."""
   
    # This source will move in a random path around the array, changing its azimuth and elevation over time.
    bf_model = BeamformingModel(array)

    # generate spline points for azimuth and elevation
    az_spline_points = np.random.uniform(-90, 90, size=(spline_points,))
    de_spline_points = np.random.uniform(0, 90, size=(spline_points,))
    print("Azimuth spline points (degrees):", az_spline_points)
    print("Elevation spline points (degrees):", de_spline_points)
    az_spline = scipy.interpolate.CubicSpline(np.linspace(0, T, spline_points), az_spline_points)
    de_spline = scipy.interpolate.CubicSpline(np.linspace(0, T, spline_points), de_spline_points)

    t = np.arange(0, T, 1/44100)
    arrival_az = np.radians(az_spline(t))
    arrival_de = np.radians(de_spline(t))

    manifold_vector_main = bf_model.compute_manifold_vector(arrival_az, arrival_de, frequency) # shape (T, num_elements)

    simple_tone = np.exp(1j * 2 * np.pi * frequency * t) # shape (T,)
    tone_array = np.real(manifold_vector_main * simple_tone[:, np.newaxis]) # shape (T, num_elements)  

    signal_power = np.mean(tone_array**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), tone_array.shape)
    tone_array += noise

    return arrival_az, arrival_de, tone_array

def generate_hex_array(num_elements = 16, d: float = 343/2000/4, max_angle: float = 15.0) -> BeamformingArray:
    """Helper function to generate a hexagonal array of elements for testing."""

    max_angle = np.radians(max_angle)
    
    d_horiz = d
    d_vert = d * np.sin(np.radians(60))

    coords = []
    L = num_elements
    for row in range(-L//2, L//2 + 1):
        for col in range(-L//2, L//2 + 1):
            x = 0
            y = col * d_horiz + (row % 2) * d_horiz / 2
            z = row * d_vert
            coords.append((x, y, z))

    coords = np.array(coords)
    dist_sq = coords[:,1]**2 + coords[:,2]**2
    sorted_inds = np.argsort(dist_sq) # sort by distance from center
    hex_coords = coords[sorted_inds][:num_elements] # shape (num_elements, 3)

    r2 = dist_sq[sorted_inds][:num_elements]
    norm_r = np.sqrt(r2) / np.max(np.sqrt(r2))

    num_circles = len(np.unique(np.round(r2, decimals=5)))
    radius_of_curvature = num_circles * d / max_angle
    hex_coords[:, 0] = radius_of_curvature * (1 - np.cos(max_angle * norm_r)) # curve the array in the x dimension based on distance from center, with a max angle of max_angle
    
    # create element axes where x-axis points to center
    shifted_hex_coords = hex_coords.copy()
    shifted_hex_coords[:, 0] -= radius_of_curvature
    eX = -shifted_hex_coords / np.linalg.norm(shifted_hex_coords, axis=1)[:, np.newaxis] # shape (num_elements, 3)
    eY = np.cross(eX, np.array([0, 0, 1])) # shape (num_elements, 3)
    eZ = np.cross(eX, eY) # shape (num_elements, 3)

    return hex_coords[:, 0], hex_coords[:, 1], hex_coords[:, 2], eX, eY, eZ

#region Beamforming Model
class BeamformingModel:
    def __init__(self, array: BeamformingArray, c: float = 343):
        self.array = array
        self.shading_model = ArrayShading(array)
        self.c = c # speed of sound in m/s

    def compute_steering_vector(self, steer_az: np.ndarray, steer_de: np.ndarray, frequency: float) -> np.ndarray:

        c = self.c # speed of sound in m/s
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        element_positions = np.stack([self.array.X, self.array.Y, self.array.Z], axis=1) # shape (num_elements, 3)

        directions = np.stack((np.cos(steer_de),
                              np.sin(steer_az) * np.sin(steer_de),
                              np.cos(steer_az) * np.sin(steer_de)), axis = -1) # shape (len(steer_az), len(steer_de), 3)

        if len(steer_az.shape) == 2:
            path_lengths = np.einsum('ijk,mk->ijm', directions, element_positions) # shape (len(steer_az), len(steer_de), num_elements)
        elif len(steer_az.shape) == 1:
            path_lengths = np.einsum('ik,mk->im', directions, element_positions) # shape (len(steer_az), num_elements)
        else:
            raise ValueError("steer_az and steer_de must be either 1D or 2D arrays")
        
        steering_vector = np.exp(-1j * k * path_lengths)

        return steering_vector
    
    def compute_manifold_vector(self, pw_az: np.ndarray, pw_de: np.ndarray, frequency: float) -> np.ndarray:
        
        c = self.c
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        directions = np.stack((np.cos(pw_de),
                        np.sin(pw_az) * np.sin(pw_de), 
                        np.cos(pw_az) * np.sin(pw_de)), axis=-1) # shape (len(az), len(de), 3)

        # m = element index, k = XYZ
        element_positions = np.stack([self.array.X, self.array.Y, self.array.Z], axis=1) # shape (num_elements, 3)

        if len(pw_az.shape) == 2:
            phases = np.einsum('ijk,mk->ijm', directions, element_positions) # shape (num_elements, len(az)*len(de))
        elif len(pw_az.shape) == 1:
            phases = np.einsum('ik,mk->im', directions, element_positions) # shape (num_elements, len(az))
        else:
            raise ValueError("pw_az and pw_de must be either 1D or 2D arrays")
        
        element_directivity = self.array.compute_element_directivity(pw_az, pw_de) # shape (len(az), len(de))
        manifold_vector = element_directivity * np.exp(-1j * k * phases) # shape ( len(az), len(de), num_elements)

        return manifold_vector
    
    def apply_spatial_filter(self, signals: np.ndarray, steer_az: np.ndarray, steer_de: np.ndarray, frequency: float, nperseg: int = 1024):
       
        steering_vector = self.compute_steering_vector(steer_az, steer_de, frequency) 
        shading_vector = self.shading_model.compute_raised_cosine_window(p = 0.5, dims = [False, True, True]) # shape (num_elements,)
        steering_vector = steering_vector * shading_vector[np.newaxis, np.newaxis, :] # shape (num_elements, Az*De) or (num_elements, Az) depending on the shape of steer_az

        # flatten steering vector angular dimensions from (Az, De, NumElements) to (Az*De, NumElements)
        if steering_vector.ndim == 3:
            steering_vector = np.reshape(steering_vector, (np.prod(steering_vector.shape[0:2]), steering_vector.shape[2])).T # shape (num_elements, Az*De)

        # split the signal into chunks to lower CPU overhead
        num_chunks = int(signals.shape[0] // nperseg)
        signal_chunks = signals[:num_chunks*nperseg, :].reshape(num_chunks, nperseg, signals.shape[1]) # shape (num_chunks, nperseg, num_elements)
        pad_length = nperseg - signals[num_chunks*nperseg:, :].shape[0]
        signal_leftover = np.pad(signals[num_chunks*nperseg:, :], ((0, pad_length), (0, 0)), mode='constant') # shape (leftover_samples, num_elements)
        signal_chunks = np.concatenate((signal_chunks, signal_leftover[np.newaxis, :, :]), axis=0) # shape (num_chunks+1, nperseg, num_elements)

        beam_time_series = np.zeros((num_chunks + 1, nperseg, steering_vector.shape[1])) # shape (num_chunks, nperseg, Az*De)
        for i in range(num_chunks + 1):
            beam_time_series[i, :, :] = np.real(signal_chunks[i] @ steering_vector) # shape (T, Az * De)

        print("Finished applying spatial filter to all signal chunks.")

        beam_time_series_chunks = beam_time_series.copy() # shape (num_chunks, nperseg, Az*De)
        beam_time_series = beam_time_series.reshape(signals.shape[0] + pad_length, steering_vector.shape[1]) # shape (T, Az*De)
        beam_time_series = beam_time_series[:signals.shape[0], :] # remove the padded samples, shape (T, Az*De)
        beam_time_series = beam_time_series.reshape(signals.shape[0], steer_az.shape[0], steer_de.shape[1]) # shape (T, Az, De)

        return beam_time_series, beam_time_series_chunks
    
    def compute_beampattern(self, frequency, c: float = None, 
                            delta_az = 0.25, delta_de = 0.25, steer_az = np.array([[0]]), steer_de = np.array([[0]]),
                            shading_method = 'uniform', shading_vector = None, use_primary_filter = False):
        
        c = c if c is not None else self.c
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        az = np.radians(np.arange(-180, 180, delta_az)) # degrees
        de = np.radians(np.arange(0, 90, delta_de)) # degrees
        AZ, DE = np.meshgrid(az, de, indexing = 'ij')

        if use_primary_filter:
            cutoff_frequencies = self.compute_cutoff_frequencies() # shape (num_elements,)
            element_mask = frequency < cutoff_frequencies
        else:
            element_mask = np.ones(len(self.array.X), dtype=bool)

        # compute vectors needed to compute the beampattern
        manifold_vector = self.compute_manifold_vector(AZ, DE, frequency)[:, :, element_mask] # shape (len(az), len(de), num_elements)
        steering_vector = self.compute_steering_vector(steer_az, steer_de, frequency)[:, :, element_mask] # shape (len(steer_az), len(steer_de), num_elements)
        if shading_method == 'uniform':
            shading_vector = np.ones(len(self.array.X))[element_mask] # shape (num_elements,)
        elif shading_method == 'raised_cosine':
            shading_vector = self.shading_model.compute_raised_cosine_window(p = 0.5, dims = [False, True, True])[element_mask] # shape (num_elements,)
        elif shading_method == 'kaiser':
            shading_vector = self.shading_model.compute_kaiser_window(beta = 3, dims = [False, True, True]) # shape (num_elements,)
        elif shading_method == 'custom':
            if (shading_vector == None).any():
                raise ValueError("Custom shading vector must be provided when shading_method is 'custom'")
            else:
                shading_vector = shading_vector[element_mask]
                steering_vector = np.ones_like(steering_vector)[element_mask] # ignore the steering vector when using custom shading, since the custom shading can already include steering by having complex values
        else:
            raise ValueError("Invalid shading method. Must be one of 'uniform', 'raised_cosine', 'kaiser', or 'custom'.")

        beampattern = np.sum(manifold_vector * \
                             steering_vector.conj() * \
                                shading_vector, axis = 2)

        return az, de, beampattern
    
    def compute_nearfield_beampattern(self):

        """Computation of the nearfield beampattern, by using inter-element time difference. 
        The near-field should be identified by measuring the time differences between elements, via x-correlation.
        When a certain threshold of the curvature of time difference is met, the source is identified as being in the near-field."""

        # 1. minimum time difference between elements that shows curvature
        #   a. dt = 1/fs -> a linear time difference will look like [-dt, 0, dt], but curved will be [dt, 0, dt]
        #   b. Measure across elements of the array: middle, and two radially extreme elements

        # Far-field approximation is broken when 

    def create_primary_filter(self, P: int = 5):

        """
        Primary filters applied to the array in the case of Frequency Invariant (FI) beamforming.
        P is the aperture per frequency in half-wavelengths, or the number of active elements per frequency
        """

        # use the multi-rate method from Ward et al. (1996)
        # x_n = P * lam_i / 2
        primary_filter = []
        for ele in range(len(self.array.X)):
            x_n = np.sqrt(self.array.X[ele]**2 + self.array.Y[ele]**2 + self.array.Z[ele]**2)
            cutoff_freq = P * self.c / (2 * x_n)
            sos = scipy.signal.butter(N = 4, Wn = cutoff_freq, btype = 'lowpass', output = 'sos')
            primary_filter.append(sos)

        return primary_filter

    def least_squares_beamforming_weights(self, az: np.ndarray, de: np.ndarray, B_p: np.ndarray, frequency: float = None) -> np.ndarray:
        """For a given beampattern, compute the least squares beamforming weights that would produce that beampattern."""

        if frequency is None:
            frequency = self.array.design_frequency

        assert az.ndim == 1 and de.ndim == 1, "az and de must be 1D arrays"
        assert B_p.shape == (len(az), len(de)), f"B_p must have shape {(len(az), len(de))}, but got {B_p.shape}"
        assert B_p.dtype == np.complex128 or B_p.dtype == np.complex64, "B_p must be a complex-valued array"

        AZ, DE = np.meshgrid(az, de, indexing='ij')

        manifold_vector = self.compute_manifold_vector(AZ, DE, frequency) # shape (len(az), len(de), num_elements)

        # flatten the angular dimensions of the manifold vector and target beam pattern 
        manifold_vector_flat = manifold_vector.reshape(-1, manifold_vector.shape[2]) # shape (Az*De, num_elements)
        B_p_flat = B_p.reshape(-1, 1) # shape (Az*De, 1)

        M, N = manifold_vector_flat.shape
        W_spatial = np.sqrt(np.cos(DE.flatten())[:, np.newaxis]) # shape (Az*De, 1)

        num_elements = manifold_vector_flat.shape[1]

        def residual_function(w_flat):
            
            # reconstruct the complex weights from the flattened real and imaginary parts
            w_real = w_flat[:num_elements]
            w_imag = w_flat[num_elements:]
            w = w_real + 1j * w_imag # shape (num_elements,)

            current_pattern = w.conj().T @ manifold_vector_flat.T # shape (Az*De, 1)

            diff = current_pattern - B_p_flat.flatten() # shape (Az*De, 1)

            return np.concatenate([np.real(diff), np.imag(diff)])
        
        def jacobian_func(x):
            """
            Returns jacobian matrix J, where J[i, j] = d(residual_function[i]) / d(x[j])
            Rows: 2*M (real then imaginary parts of the residuals)
            Columns: 2*N (real then imaginary parts of the weights) 
            """

            V_w = (manifold_vector_flat * W_spatial)

            J = np.block([
                [V_w.real, -V_w.imag],
                [V_w.imag, -V_w.real]
            ])

            return J

        # initial guess of uniform shading with no phase shifts
        initial_w_flat = np.concatenate([np.ones(num_elements), np.zeros(num_elements)])

        res = scipy.optimize.least_squares(residual_function, initial_w_flat, jac = jacobian_func, method='lm')

        w_opt = res.x[:num_elements] + 1j * res.x[num_elements:]

        return w_opt, res.cost, res.message
    
    def compute_cutoff_frequencies(self):

        """
        Computation of cutoff frequencies for each element for a low-pass filter, based off distance to closest neighbor
        """

        def get_aperture_size(nodes):
            if len(nodes) < 2: return 0
            subset_points = points[list(nodes)]
            return np.max(scipy.spatial.distance.pdist(subset_points, metric='cityblock'))
        
        n = len(self.array.X)
        points = np.column_stack((self.array.X, self.array.Y, self.array.Z))
        dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(points))
        unique_distances = np.sort(np.unique(dist_matrix))

        cutoff_frequencies = np.zeros(len(self.array.X))
        # start from smallest d so that points in multiple subarrays have lowest cutoff
        for d in np.flip(unique_distances): 
            if d == 0: d = 0.01 - 1e-6
            # points will be connected if their mutual distance <= d
            # a graph is created with edges between points where distance <= d
            G = nx.Graph()
            G.add_nodes_from(range(n))

            rows, cols = np.where((dist_matrix <= d))
            edges = zip(rows, cols)
            G.add_edges_from(edges)

            subarrays = list(nx.connected_components(G))
            largest_subarray = max(subarrays, key=get_aperture_size) # largest by points

            cutoff_frequencies[np.array(list(largest_subarray))] = 2 * self.c / d

        return cutoff_frequencies
    
    def get_beamforming_performance_measures(self, delta_az = 0.25, delta_de = 0.25, steer_az = np.array([[0]]), steer_de = np.array([[0]]), frequency: float = None, c: float = 343, use_primary_filter = False):

        """
        Computation of the following performance measusres for a given array:
        1. Directivity
        2. HPBW (min, mean, max across azimuth)
        """

        # 1. Directivity (B(0, 0) / 1/(4*pi) * integral of B(az, de) over the sphere)
        if frequency is None:
            frequency = self.array.design_frequency

        # rotate the entire array so that the steer_az, steer_de are along the x-axis
        r = scipy.spatial.transform.Rotation.from_euler('zy', [-steer_az[0][0], steer_de[0][0]])
        grid_points = np.stack([self.array.X, self.array.Y, self.array.Z], axis = -1)
        rotated_points = r.apply(grid_points)
        self.array.X = rotated_points[:, 0]
        self.array.Y = rotated_points[:, 1]
        self.array.Z = rotated_points[:, 2]

        az, de, bp = self.compute_beampattern(frequency = frequency, c=c, delta_az=delta_az, delta_de=delta_de, shading_method='uniform', use_primary_filter=use_primary_filter) # shape (Az, De)

        bp_norm = np.abs(bp) / np.max(np.abs(bp))
        sum_over_sphere = np.sum(np.sum(bp_norm * np.sin(de[np.newaxis, :]), axis = 1) * (de[1] - de[0]))*(az[1] - az[0]) # approximate integral over the sphere using the trapezoidal rule, with cos(de) weighting for the spherical coordinates
        directivity = bp_norm[az == 0, de == 0] / (sum_over_sphere / (2 * np.pi))
        DI = 10 * np.log10(directivity[0])

        # 2. Min, Mean, Max HPBW
        bp_norm_db = 10 * np.log10(bp_norm)
        hpbw_de = np.zeros(len(az))
        for i, th in enumerate(az):
            # find de where beampattern first drops below -3 dB for this azimuth
            de_slice = bp_norm_db[i, :]
            if np.any(de_slice < -3):
                hpbw_de[i] = de[np.where(de_slice < -3)[0][0]] # find the first elevation where the beampattern drops below -3 dB
            else:
                hpbw_de[i] = np.radians(90) # if the beampattern never drops below -3 dB, set the HPBW to 90 degrees

        min_hpbw = np.min(hpbw_de)
        mean_hpbw = np.mean(hpbw_de)
        max_hpbw = np.max(hpbw_de)

        # 3. Maximum Side Lobe Level
        msll = np.zeros(len(az))
        for i in range(len(az)):
            de_slice = bp_norm_db[i, :]
            null = np.where(np.sign(np.diff(de_slice)) > -1)[0]
            if len(null) > 1:
                msll[i] = np.max(de_slice[null[0]:])
            else:
                msll[i] = 0

        msll = np.max(msll)

        return DI, (min_hpbw, mean_hpbw, max_hpbw), msll

#region Plotting Functions
class BeamformingPlot:
    def __init__(self, bf_model: BeamformingModel):
        self.bf_model = bf_model

    def plot_spatially_filtered_result(self, filtered_power: np.array, steer_az: np.array, steer_de: np.array, frame_idx: int = 0):

        # find max power index for one time index
        max_inds = np.unravel_index(np.argmax(filtered_power[:,:,frame_idx]), filtered_power[:,:,frame_idx].shape)

        az = np.arange(-90, 90, 5)
        de = np.arange(-90, 90, 5)

        fig, ax = plt.subplots()
        im = ax.imshow(10 * np.log10(filtered_power[:, :, frame_idx]/np.max(filtered_power[:, :, frame_idx])),
                       extent=(np.degrees(np.min(steer_az)), np.degrees(np.max(steer_az)), np.degrees(np.min(steer_de)), np.degrees(np.max(steer_de))),
                    aspect='auto', origin='lower', vmin = -20, vmax = 0)
        ax.scatter(az[max_inds[1]], de[max_inds[0]], c='red', marker='x', label='Estimated Source Location')
        fig.colorbar(im, ax=ax, label='Filtered Signal RMS Power (dB)')
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_title(f'Filtered Signal RMS Power')

    def setup_animation_plot(self, beam_power, steer_az, steer_de, arrival_az, arrival_de):
        fig, ax = plt.subplots()
        
        # Initialize the image with the first frame
        initial_data = 10 * np.log10(beam_power[:, :, 0] / np.max(beam_power[:, :, 0]))
        im = ax.imshow(initial_data.T,
                    extent=(np.degrees(np.min(steer_az)), np.degrees(np.max(steer_az)), 
                            np.degrees(np.min(steer_de)), np.degrees(np.max(steer_de))),
                    aspect='auto', origin='lower', vmin=-20, vmax=0, cmap='turbo')
        
        path_line, = ax.plot([], [], 'w-', label='Actual Source Path')
        
        # Initialize the scatter point for max power
        max_point, = ax.plot([], [], 'rx', label='Estimated Source Location')
        
        fig.colorbar(im, ax=ax, label='Filtered Signal RMS Power (dB)')
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        
        return fig, ax, im, max_point, path_line
    
    def example_beampattern_animation(self):

        # Example source detection test
        arrival_az, arrival_de, tone_array = time_varying_signal(self.bf_model.array, 
                                                                frequency = frequency, 
                                                                snr_db = 400, 
                                                                spline_points = 5, 
                                                                T = 15)

        # Use a range of angles for the spatial filter to create a "map"
        az = np.radians(np.arange(-90, 90, 2.5))
        de = np.radians(np.arange(0, 90, 2.5))
        steer_az, steer_de = np.meshgrid(az, de, indexing='ij')

        _, bts_chunks = self.bf_model.apply_spatial_filter(tone_array, steer_az, steer_de, frequency=frequency, nperseg=4096)

        # Compute the power of the filtered signal for each steering angle for windowed time steps
        beam_power = np.mean(np.abs(bts_chunks)**2, axis=1) # shape (num_chunks, nperseg, Az * De)
        beam_power = beam_power.transpose(1, 0) # shape (Az * De, num_chunks, 1)
        beam_power = beam_power.reshape(steer_az.shape[0], steer_de.shape[1], -1) # shape (Az, De, num_chunks)

        print("Finished computing beam power for all time steps and steering angles.")

        # Setup the figure
        fig, ax, im, max_point, path_line = self.setup_animation_plot(beam_power, steer_az, steer_de, arrival_az, arrival_de)

        def update(frame):
            # Calculate normalized power for this frame
            frame_data = beam_power[:, :, frame]
            norm_power = 10 * np.log10(frame_data / np.max(beam_power[:, :, frame]))

            # Update the image
            im.set_array(norm_power.T)

            # Update the "Max Power" marker
            max_inds = np.unravel_index(np.argmax(frame_data), frame_data.shape)
            # Map indices back to degrees
            curr_az = np.degrees(az[max_inds[0]])
            curr_de = np.degrees(de[max_inds[1]])
            max_point.set_data([curr_az], [curr_de])
            arrival_index = frame * 4096 % len(arrival_az) # loop through the arrival angles if we run out of time steps
            path_line.set_data(np.degrees(arrival_az[:frame*4096]), np.degrees(arrival_de[:frame*4096]))
            
            return im, max_point, path_line

        # Create animation: interval is in ms (200ms = 5 frames per second)
        ani = FuncAnimation(fig, update, frames=beam_power.shape[2], interval=int(4096/44100*1000), blit=True)
        ani.save('AcousticBeamforming/Figures/beamforming_animation.gif', writer='pillow', fps = int(44100/4096), dpi = 100)

        plt.show()

    def plot_beampattern_image(self, frequency, az, de, beampattern, method = 'rectangular'):

        assert method in ['rectangular', 'polar'], "Method must be either 'rectangular' or 'polar'"

        if np.max(beampattern) != 0:
            beampattern = np.abs(beampattern) / np.max(np.abs(beampattern)) # normalize the beampattern for better visualization
            beampattern = 10 * np.log10(beampattern) # convert to dB scale
        
        if method == 'rectangular':
        
            fig, ax = plt.subplots()
            
            im = ax.imshow(beampattern.T,
                            extent=(np.degrees(np.min(az)), np.degrees(np.max(az)), np.degrees(np.min(de)), np.degrees(np.max(de))),
                                aspect='auto', origin='lower', vmin = -50, vmax = 0, cmap = 'turbo')
            fig.colorbar(im, ax=ax, label='Beamforming Gain (dB)')
            ax.contour(beampattern.T,
                        levels=[-3], colors='white',
                            extent=(np.degrees(np.min(az)), np.degrees(np.max(az)), np.degrees(np.min(de)), np.degrees(np.max(de))), linewidths=2)
            ax.set_xlabel('Azimuth (degrees)')
            ax.set_ylabel('Elevation (degrees)')

        if method == 'polar':
            # 1. Define the layout
            # We create a 3x3 (or similar) grid logic. 
            # 'A' is the Array geometry, 'P' is the Polar beampattern.
            # The list-of-lists represents the visual arrangement.
            layout = [
                ['array', '.',     '.'],
                ['.',     'polar', 'polar'],
                ['.',     'polar', 'polar'],
                ['.',     'polar', 'polar']
            ]

            fig, ax_dict = plt.subplot_mosaic(
                layout,
                per_subplot_kw={
                    'polar': {'projection': 'polar'} # Only the beampattern is polar
                },
                figsize=(10, 8)
            )

            # Alias our axes for clarity
            ax = ax_dict['polar']
            ax_geom = ax_dict['array']

            # --- BEAMPATTERN PLOT (Your existing logic) ---
            R_COORD = de
            AZ_MESH, R_MESH = np.meshgrid(az, R_COORD, indexing='ij')

            cmap = mpl.cm.turbo
            bounds = np.arange(-21, 3, 3)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='min')
            im = ax.pcolormesh(AZ_MESH, R_MESH, beampattern, shading='auto', cmap=cmap, norm=norm)
            
            fig.colorbar(im, ax=ax, label='Normalized Gain (dB)')
            
            # Ensure contour uses the same transformed mesh
            ax.contour(AZ_MESH, R_MESH, beampattern, levels=[-3], colors='red', linewidths=2)

            # 3. Fix the labels so the user still sees "Elevation" values
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_rlim(0, np.radians(90))
            
            ticks = np.radians([0, 30, 60, 90])
            ax.set_rticks(ticks)
            ax.set_yticklabels(['0°', '30°', '60°', '90°'], fontweight='bold', fontsize=8)
            ax.set_xlabel('Azimuth (degrees)')

            # --- ARRAY GEOMETRY PLOT (The new addition) ---
            # Plotting X and Y coordinates (assuming Z is uniform or ignored for the 2D footprint)
            element_mask = frequency < self.bf_model.compute_cutoff_frequencies()
            self.bf_model.array.plot_array_geometry(ax = ax_geom, projection = '2d', element_mask=element_mask)

            plt.tight_layout()

    def plot_beampattern_slice(self, ax, az, de, beampattern, slice_az = None, slice_de = None, label_text: str = '', style: str = 'polar'):
        
        """
        The slice is on a line with slope equivalent to the tangent of the slice angle in azimuth.
        """
        assert (slice_az is not None) or (slice_de is not None), "Either slice_az or slice_de must be provided"
        assert (style in ['polar', 'cartesian']), "Style must be either 'polar' or 'cartesian'"

        if np.max(beampattern) != 0:
            beampattern = np.abs(beampattern) / np.max(np.abs(beampattern)) # normalize the beampattern for better visualization
            beampattern_dB = 10 * np.log10(beampattern) # convert to dB scale\

        if slice_az is not None:
            complimentary_az = np.radians(180) - (slice_az + np.radians(180)) % np.radians(360)
            beampattern_plot = np.concatenate((beampattern_dB[az == slice_az, :].T, np.flip(beampattern_dB[az == complimentary_az, 1:].T, axis=0)), axis = 0).flatten()
            de_plot = np.concatenate((de, -np.flip(de[1:])), axis = 0)
            if style == 'polar':
                ax.plot(de_plot, beampattern_plot, label=label_text)
                ax.set_rmax(0)
                ax.set_rmin(-25)
                ax.set_rticks([])
                ax.grid(False)
            else:   
                ax.plot(np.degrees(de_plot), beampattern_plot, label=label_text)
                ax.set_xlabel('Elevation (degrees)')
                ax.set_ylim(-25, 0)
                ax.set_ylabel('Beamforming Gain (dB)')
                ax.grid()
        elif slice_de is not None:
            if style == 'polar':
                ax.plot(az, beampattern_dB[:, de == slice_de], label=label_text)
                ax.set_xlabel('Azimuth (degrees)')
                ax.set_rmax(0)
                ax.set_rmin(-25)
                ax.set_rticks([])
            else:
                ax.plot(np.degrees(az), beampattern_dB[:, de == slice_de], label=label_text)
                ax.set_xlabel('Azimuth (degrees)')
                ax.set_ylim(-25, 0)
                ax.set_ylabel('Beamforming Gain (dB)')

    def plot_beampattern_3d(self, az, de, beampattern):

        if np.max(beampattern) != 0:
            beampattern = beampattern / np.max(beampattern) # normalize the beampattern for better visualization
            beampattern = 10 * np.log10(np.abs(beampattern)) # convert to dB scale

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        # convert to cartesian coordinates for 3D plot
        delta_az = np.degrees(np.diff(az)[0])
        delta_de = np.degrees(np.diff(de)[0])
        AZ, DE = np.meshgrid(np.radians(np.arange(-180, 180, delta_az)), np.radians(np.arange(-90, 90, delta_de)), indexing='ij')

        R = beampattern + 50
        R[R < 0] = 0 # set negative values to 0 for better visualization]
        R = R[:, de >= 0] # only plot the upper hemisphere
        R = np.concatenate((R, R), axis=0) # duplicate the beampattern to cover the full 360 degrees in azimuth

        X = R * np.cos(DE)
        Y = R * np.sin(AZ) * np.sin(DE)
        Z = R * np.cos(AZ) * np.sin(DE)

        my_cmap = cm.get_cmap('turbo') # Choose a vibrant colormap
        colors = my_cmap(np.abs(R) / np.max(R)) # colors.shape is now (360, 360, 4) - RGBA        

        ax.plot_surface(X, Y, Z, facecolors = colors, shade = False, cstride = 2, rstride = 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        ax.grid()
        
#region Example runs
if __name__ == "__main__":

    # # random point search
    # X = np.zeros(16)
    # Y = np.array([np.float64(0.0), np.float64(0.13), np.float64(-0.14), np.float64(-0.17), np.float64(-0.72), np.float64(-0.92), np.float64(-1.39), np.float64(-1.77), np.float64(-2.13), np.float64(-2.1), np.float64(-2.41), np.float64(-2.35), np.float64(-2.44), np.float64(-2.47), np.float64(-2.48), np.float64(-2.06)])
    # Z = np.array([np.float64(0.0), np.float64(-0.42), np.float64(-0.6), np.float64(-0.68), np.float64(-0.65), np.float64(-0.68), np.float64(-0.95), np.float64(-1.01), np.float64(-1.46), np.float64(-1.91), np.float64(-1.88), np.float64(-2.39), np.float64(-2.52), np.float64(-3.03), np.float64(-3.35), np.float64(-3.5)])

    opt_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_20260427-150406.npz'))
    coords = opt_data['arr_0']
    accepted_states = opt_data['arr_1']
    accepted_energies = opt_data['arr_2']
    num_elements = np.shape(coords)[0]
    # spiral search
    X = np.zeros(num_elements)
    Y = coords[:,0]
    Z = coords[:,1]

    running_min = np.minimum.accumulate(accepted_energies)
    mask = np.concatenate(([True], np.diff(running_min) < 0))
    decreasing_states = accepted_states[mask, :]
    decreasing_energies = accepted_energies[mask]
    accepted_coords = np.zeros((decreasing_states.shape[0], coords.shape[0], 2))

    curr_angle = np.zeros(accepted_coords.shape[0])
    for i in range(2, num_elements):
        curr_angle += decreasing_states[:, (num_elements - 1) + (i - 2)]
        accepted_coords[:, i, 0] = accepted_coords[:, i-1, 0] + decreasing_states[:,i-1] * np.cos(curr_angle)
        accepted_coords[:, i, 1] = accepted_coords[:, i-1, 1] + decreasing_states[:,i-1] * np.sin(curr_angle)
    
    last_points = accepted_coords[:, -1, :]
    angles = np.arctan2(last_points[:, 1], last_points[:, 0])

    rotated_coords = accepted_coords.copy()
    
    # rotate accepted coords
    rotated_coords[:, :, 0] = accepted_coords[:, :, 0] * np.cos(-angles)[:, np.newaxis] - accepted_coords[:, :, 1] * np.sin(-angles)[:, np.newaxis]
    rotated_coords[:, :, 1] = accepted_coords[:, :, 0] * np.sin(-angles)[:, np.newaxis] + accepted_coords[:, :, 1] * np.cos(-angles)[:, np.newaxis]

    accepted_coords = rotated_coords.copy()

    if np.any(accepted_coords[:, -1, 1] > 1e-6):
        print(accepted_coords[:, -1, 1])
        raise ValueError("Last points are not on x-axis")

    fig, ax = plt.subplots(figsize = (8, 8))
    limit = np.max(np.abs(accepted_coords))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, linestyle = '--', alpha = 0.6)

    my_cmap = cm.get_cmap('turbo') # Choose a vibrant colormap
    colors = my_cmap(np.linspace(0, 1, num_elements)) # colors.shape is now (360, 360, 4) - RGBA      
     
    scat = ax.scatter(accepted_coords[0, :, 0].flatten(), accepted_coords[0, :, 1].flatten(), s = 50, c = colors, marker = 'o', edgecolors='k', zorder=10)
    text_energy = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontweight='bold')

    def init():
        scat.set_offsets(np.empty((0,2)))
        text_energy.set_text('')
        return scat, text_energy
    
    def update(frame):

        current_points = accepted_coords[frame]
        scat.set_offsets(current_points)
        text_energy.set_text(f"Energy: {decreasing_energies[frame]:.2e}")
        
        return scat, text_energy
    
    ani = FuncAnimation(fig, update, frames = len(accepted_coords), init_func = init, blit = True, interval = 1000)

    array = BeamformingArray(X, Y, Z, design_frequency=3e3, element_directivity=ElementDirectivity.DIPOLE)
    bf_model = BeamformingModel(array)
    plotter = BeamformingPlot(bf_model)

    f = np.logspace(np.log10(20), np.log10(500), num = 10) # generic frequencies (used in optimization)
    hpbw = np.zeros((len(f), 3))
    di = np.zeros(len(f))
    msll = np.zeros(len(f))
    for i, freq in enumerate(f):
        di[i], (hpbw[i,0], hpbw[i,1], hpbw[i,2]), msll[i] = bf_model.get_beamforming_performance_measures(frequency=freq)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(f, di)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Directivity Index (dB)')
    ax[0].set_title('Directivity Index vs Frequency')
    ax[0].grid()

    ax[1].plot(f, np.degrees(hpbw[:, 0]), 'b-', label='Min HPBW')
    ax[1].plot(f, np.degrees(hpbw[:, 1]), 'k--', label='Mean HPBW')
    ax[1].plot(f, np.degrees(hpbw[:, 2]), 'r-', label='Max HPBW')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('HPBW (degrees)')
    ax[1].set_title('HPBW vs Frequency')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(f, msll)
    ax[2].set_xscale('log')
    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('Max Side Lobe Level (dB)')
    ax[2].set_title('MSLL vs. Frequency')
    ax[2].grid()

    fig, ax = plt.subplots()
    
    array.plot_array_connections(fig, ax, f=f, c = 1460)

    plt.show()