from matplotlib import style
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
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
    def __init__(self, array: BeamformingArray):
        self.array = array
        self.shading_model = ArrayShading(array)

    def compute_steering_vector(self, steer_az: np.ndarray, steer_de: np.ndarray, frequency: float) -> np.ndarray:

        c = 343 # speed of sound in m/s
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        element_positions = np.stack((self.array.X, self.array.Y, self.array.Z), axis=-1) # shape (num_elements, 3)
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
        
        c = 343 # speed of sound in m/s
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
    
    def compute_beampattern(self, frequency, 
                            delta_az = 0.25, delta_de = 0.25, steer_az = np.array([[0]]), steer_de = np.array([[0]]),
                            shading_method = 'uniform', shading_vector = None):
        
        c = 343 # speed of sound in m/s
        wavelength = c / frequency
        k = 2 * np.pi / wavelength # wavenumber

        az = np.radians(np.arange(-180, 180, delta_az)) # degrees
        de = np.radians(np.arange(0, 90, delta_de)) # degrees
        AZ, DE = np.meshgrid(az, de, indexing = 'ij')

        # compute vectors needed to compute the beampattern
        manifold_vector = self.compute_manifold_vector(AZ, DE, frequency)
        steering_vector = self.compute_steering_vector(steer_az, steer_de, frequency) # shape (num_elements,)
        if shading_method == 'uniform':
            shading_vector = np.ones(len(self.array.X)) # shape (num_elements,)
        elif shading_method == 'raised_cosine':
            shading_vector = self.shading_model.compute_raised_cosine_window(p = 0.5, dims = [False, True, True]) # shape (num_elements,)
        elif shading_method == 'kaiser':
            shading_vector = self.shading_model.compute_kaiser_window(beta = 3, dims = [False, True, True]) # shape (num_elements,)
        elif shading_method == 'custom':
            if (shading_vector == None).any():
                raise ValueError("Custom shading vector must be provided when shading_method is 'custom'")
            else:
                shading_vector = shading_vector
                steering_vector = np.ones_like(steering_vector) # ignore the steering vector when using custom shading, since the custom shading can already include steering by having complex values
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
    
    def get_beamforming_performance_measures(self, frequency: float = None) -> tuple[float, float, float]:

        """
        Computation of the following performance measusres for a given array:
        1. Directivity
        2. Array gain vs. spatially white noise
        3. Sensitivity and the tolerance factor
        """

        # 1. Directivity (B(0, 0) / 1/(4*pi) * integral of B(az, de) over the sphere)
        if frequency is None:
            frequency = self.array.design_frequency
        az, de, bp = self.compute_beampattern(frequency = frequency, shading_method='uniform')

        bp_norm = np.abs(bp) / np.max(np.abs(bp))
        sum_over_sphere = np.sum(np.sum(bp_norm * np.cos(de[np.newaxis, :]), axis = 1) * (de[1] - de[0]))*(az[1] - az[0]) # approximate integral over the sphere using the trapezoidal rule, with cos(de) weighting for the spherical coordinates
        directivity = bp_norm[az == 0, de == 0] / (sum_over_sphere / (2 * np.pi))
        DI = 10 * np.log10(directivity[0])

        # 2. Array gain vs. spatially white noise
        w = self.compute_steering_vector(np.array([[0]]), np.array([[0]]), self.array.design_frequency).flatten() # shape (num_elements,)
        array_gain = 1 / np.linalg.norm(np.abs(w))**2
        array_gain_db = 10 * np.log10(array_gain)

        return DI, array_gain_db

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

    def plot_beampattern_image(self, az, de, beampattern, method = 'rectangular'):

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
                        levels=[-3], colors='red',
                            extent=(np.degrees(np.min(az)), np.degrees(np.max(az)), np.degrees(np.min(de)), np.degrees(np.max(de))), linewidths=2)
            ax.set_xlabel('Azimuth (degrees)')
            ax.set_ylabel('Elevation (degrees)')

        if method == 'polar':
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            
            R_COORD = de
            AZ_MESH, R_MESH = np.meshgrid(az, R_COORD, indexing='ij')

            # 2. Plot using the new radial mesh
            im = ax.pcolormesh(AZ_MESH, R_MESH, beampattern, shading='auto', vmin=-50, vmax=0, cmap='turbo')
            
            fig.colorbar(im, ax=ax, label='Beamforming Gain (dB)')
            
            # Ensure contour uses the same transformed mesh
            ax.contour(AZ_MESH, R_MESH, beampattern, levels=[-3], colors='red', linewidths=2)

            # 3. Fix the labels so the user still sees "Elevation" values
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            
            # Radial limits: 0 at center, 90 at edge
            ax.set_rlim(0, np.radians(90))
            
            # Map radial positions [0, 30, 60, 90] to Elevation labels [0, 30, 60, 90]
            # Note: center (radius 0) is now 0° elevation
            ticks = np.radians([0, 30, 60, 90])
            ax.set_rticks(ticks)
            ax.set_yticklabels(['0°', '30°', '60°', '90°'])
            
            ax.set_xlabel('Azimuth (degrees)')
            # Position the label so it doesn't overlap
            ax.yaxis.set_label_coords(0.5, 1.1) 
            ax.set_ylabel('Elevation (degrees)')

    def plot_beampattern_slice(self, ax, az, de, beampattern, slice_az = None, slice_de = None, label_text: str = '', style: str = 'polar'):
        
        assert (slice_az is not None) or (slice_de is not None), "Either slice_az or slice_de must be provided"
        assert (style in ['polar', 'cartesian']), "Style must be either 'polar' or 'cartesian'"

        if np.max(beampattern) != 0:
            beampattern = np.abs(beampattern) / np.max(np.abs(beampattern)) # normalize the beampattern for better visualization
            beampattern_dB = 10 * np.log10(beampattern) # convert to dB scale\

        if slice_az is not None:
            if style == 'polar':
                ax.plot(de, beampattern_dB[az == slice_az, :].T, label=label_text)
                ax.set_rmax(0)
                ax.set_rmin(-25)
                ax.set_rticks([])
                ax.grid(False)
            else:   
                ax.plot(np.degrees(de), beampattern_dB[az == slice_az, :].T, label=label_text)
                ax.set_xlabel('Azimuth (degrees)')
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

    # def chord_distance(x, a, b, c, theta_1, d):
    #     r1 = a + b * theta_1**(1/c)
    #     r2 = a + b * x**(1/c)
    #     return d - np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(x - theta_1))
    
    f = 3*np.logspace(2, 3, num = 12)
    # f = np.flip(f)
    # d_target = 343 / (f * 2) # half wavelength distance for each frequency

    # a = 0
    # c = 2
    # b = d_target[0] / ((np.pi/4) ** (1/c))
    # theta_1 = 0

    # X = np.zeros(12)
    # Y = np.zeros(12)
    # Z = np.zeros(12)
    # for i in range(1,12):
    #     if i == 1:
    #         r1 = a
    #         initial_guess = np.pi / 2
    #     else:
    #         r1 = a + b * theta_1**2
    #         initial_guess = theta_1 + d_target[i-1] / r1

    #     theta_2_solution = scipy.optimize.fsolve(chord_distance, initial_guess, args=(a, b, c, theta_1, d_target[i]))
    #     theta_2 = theta_2_solution[0]
    #     theta_1 = theta_2

    #     Y[i] = (a + b * theta_2**(1/c)) * np.cos(theta_2)
    #     Z[i] = (a + b * theta_2**(1/c)) * np.sin(theta_2)

    # random point search
    X = np.zeros(12)
    Y = np.array([ 0.35, 0.47, 0.12, 0.2, 0.28, 0.16, 0.17, 0.17, 0.43, -0.01, 0.21, 0.05])
    Z = np.array([0.30, 0.60, -.16, -.30, 0.00,  .71,  -.19, -.10, .87, -.17, .15, -.16])

    # # spiral search
    X = np.zeros(12)
    Y = np.array([-0.04, 0.58, 0.39, 0.83, 0.37, 0.12, 0.37, 0.28, 0.18, 0.11, -0.1, 0.25])
    Z = np.array([-0.78, 0.06, -0.11, 0.35, -0.19, -0.77, -0.37, -0.61, -0.69, -0.75, -0.87, -0.41])

    array = BeamformingArray(X, Y, Z, design_frequency=3e3, element_directivity=ElementDirectivity.DIPOLE)
    bf_model = BeamformingModel(array)
    plotter = BeamformingPlot(bf_model)

    fig, ax = plt.subplots()

    for freq in f:
        DI, array_gain = bf_model.get_beamforming_performance_measures(frequency=freq)
        print(f"DI, AG @ {freq:.0f} Hz: {DI:.2f}, {array_gain:.2f}")
        az, de, bp = bf_model.compute_beampattern(frequency=freq, shading_method='uniform')
        plotter.plot_beampattern_slice(ax, az, de, bp, slice_az = 0, label_text = f'{freq:.0f} Hz', style = 'cartesian')
    ax.legend()
    
    array.plot_array_geometry()
    # theta_fine = np.arange(0, theta_2, 0.01)
    # r_fine = a + b * theta_fine**(1/c)

    # line_y = r_fine * np.cos(theta_fine)
    # line_z = r_fine * np.sin(theta_fine)
    # line_x = np.zeros_like(line_y) # Matches your sensor X array

    # plt.plot(line_x, line_y, line_z, color='gray', linestyle='--', alpha=0.6, label='Spiral Path')
    plt.show()