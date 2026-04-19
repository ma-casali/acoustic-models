import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.interpolate

class PropagationEnvironment:
    """
    Propagation environment with automatic computation of missing parameters.
    """

    def __init__(self, 
                 f, c_0, z_0=None, max_range=None, z_lims=None, 
                 roughness=None, topography=None, topography_lims=None,
                 ssp=None, rho=None, alpha=None,
                 measurement_points=None):
        """
        INPUTS
        f: frequency of study [Hz]
        c_0: reference sound speed [m/s]
        
        z_0: starting depth for terrain [m]
        max_range: the maximum range of the environment [m]
        z_lims: two elements giving ideal bounds for z [m]
        
        roughness: roughly equivalent to std in depth [m]
        topography: pre-defined topography array (optional)
        topography_lims: limits for the depth (or elevation) of the topography (optional)
        
        ssp: sound speed profile (optional, will be computed if None)
        rho: density profile (optional, will be computed if None)
        alpha: attenuation profile (optional, will be computed if None)
        measurement_points: (N, 2) array of (r, z) measurement locations

        """
        
        # Store basic parameters
        self.f = f
        self.c_0 = c_0
        self.h = ( c_0 / f ) / 4 # Grid spacing: 1/10 wavelength
        self.k = self.h * 5
        self.topography_lims = topography_lims
        self.max_range = max_range
        
        # Handle topography parameters
        if topography is None:
            # Need parameters to generate topography
            if any(item is None for item in [max_range, z_0, z_lims, roughness]):
                raise ValueError(
                    "'ranges', 'z_0', 'z_lims', and 'roughness' must be defined if no topography is given."
                )
            self.ranges = np.arange(0, max_range, self.k)
            self.z_0 = z_0
            self.z_lims = np.array(z_lims)
            self.roughness = roughness
            self.topography_input = self._compute_topography()
        else:
            # Use provided topography
            self.ranges = np.arange(0, np.max(measurement_points[:,0]), self.k)
            self.topography_input = np.interp(self.ranges, np.unique(measurement_points[:,0]), topography)
            self.z_0 = topography[0] if z_0 is None else z_0
            self.z_lims = [np.min(measurement_points[:,1]), np.max(measurement_points[:,1])]
            self.roughness = roughness
        
        # Generate mesh from topography
        self._generate_mesh()
        
        # Store measurement points
        self.measurement_points = measurement_points
        
        # Interpolate/compute physical properties
        self.ssp = self._process_ssp(ssp)
        self.rho = self._process_rho(rho)
        self.alpha = self._process_alpha(alpha)

        # Add absorbing layers
        self._add_absorbing_layer()
        
        print(f"Environment initialized: {len(self.r_mesh)} × {len(self.z_mesh)} grid")

    def _compute_topography(self):
        """
        Randomly generate topography with skewed normal distribution.
        """
        n_points = len(self.ranges)
        topography = np.zeros(n_points)
        topography[0] = self.z_0

        # Use topography_lims if provided, otherwise use z_lims
        if self.topography_lims is None:
            topo_lims = self.z_lims
        else:
            topo_lims = np.array(self.topography_lims)

        for i in range(1, n_points):
            # Skew distribution towards center of topography limits
            center_deviation = (topography[i-1] - np.mean(topo_lims)) / np.diff(topo_lims)[0]
            skew_factor = -center_deviation / self.roughness
            skew_factor = np.clip(skew_factor, -5, 5)  # Prevent extreme values
            
            # Generate next point
            topography[i] = scipy.stats.skewnorm.rvs(
                skew_factor, 
                loc=topography[i-1], 
                scale=self.roughness
            )
            
            # Keep within bounds
            topography[i] = np.clip(topography[i], topo_lims[0], topo_lims[1])
        
        return topography

    def _generate_mesh(self):
        """
        Create mesh from topography, handling slope constraints.
        """
        # Discretize topography to grid spacing
        discretized_topo = self.h * np.round(self.topography_input / self.h)
        eps = 1e-9 * self.h
        
        # Create depth mesh with artificial absorption layer (aal)
        # PML must be several wavelengths
        self.z_mesh = np.arange(self.z_lims[0], np.max(discretized_topo) + 2*self.h , self.h)
        
        # Initialize interpolated topography and range mesh
        new_topography = [discretized_topo[0]]
        r_mesh = [0.0]
        i_orig = 1  # Index in original topography
        i_new = 1   # Index in interpolated arrays
        
        # Interpolate points where slope is too steep
        while i_orig < len(discretized_topo):
            
            depth_change = discretized_topo[i_orig] - new_topography[i_new-1]
            
            if np.abs(depth_change) >= self.h - eps:
                dr_full = self.ranges[i_orig] - self.ranges[i_orig - 1]
                n_steps = int(np.floor(np.abs(depth_change) / self.h))
                if n_steps > 0:
                    dr_step = dr_full / (n_steps + 1)
                    for _ in range(n_steps):
                        r_mesh.append(r_mesh[-1] + dr_step)
                        new_topography.append(new_topography[-1] + self.h * np.sign(depth_change))
                        i_new += 1

                # Add final point
                r_mesh.append(self.ranges[i_orig])
                if r_mesh[-1] <= r_mesh[-2]:
                    raise ValueError(f"r_mesh is stepping back from {r_mesh[-2]:.2f} to {r_mesh[-1]:.2f}")
                new_topography.append(discretized_topo[i_orig])
                
            elif np.abs(depth_change) <= eps:
                # Flat section
                r_mesh.append(self.ranges[i_orig])
                new_topography.append(discretized_topo[i_orig])
                
            else:
                raise ValueError(
                    f"Unexpected depth change: {np.abs(depth_change):.9f} < h = {self.h:.9f}. "
                    "Topography discretized incorrectly!"
                )
            
            i_orig += 1
            i_new += 1
        
        # Store final mesh
        self.r_mesh = np.array(r_mesh)
        self.topography = np.array(new_topography)
        
        print(f"Mesh created: {len(self.r_mesh)} range × {len(self.z_mesh)} depth points")

    def _process_ssp(self, ssp_input):
        """
        Process sound speed profile.
        If provided: interpolate to mesh
        If None: create default
        """
        if ssp_input is not None:
            # User provided SSP - interpolate to mesh
            if self.measurement_points is None:
                raise ValueError("measurement_points required when providing ssp data")

            # Create mesh grid for interpolation
            r_grid, z_grid = np.meshgrid(self.r_mesh, self.z_mesh)
            mesh_points = np.column_stack([r_grid.ravel(), z_grid.ravel()])

            mesh_ssp = scipy.interpolate.griddata(
                self.measurement_points, 
                ssp_input,
                mesh_points
            )

            if (np.isnan(mesh_ssp)).any():
                print(mesh_points[np.isnan(mesh_ssp),:])
                raise ValueError("Nans in the sound speed profile!")
                
            mesh_ssp = mesh_ssp.reshape(len(self.r_mesh), len(self.z_mesh))

        else:
            # create default
            mesh_ssp = self._create_default_ssp()

        return mesh_ssp
        
    def _create_default_ssp(self):
        raise NotImplementedError("Child class must implement create_default_ssp()")

    def _process_rho(self, rho_input):
        """Process density profile."""
        nr = len(self.r_mesh)
        nz = len(self.z_mesh)
        
        if rho_input is not None:
            # User provided density - interpolate
            if self.measurement_points is None:
                raise ValueError("measurement_points required when providing rho data")

            # Create mesh grid for interpolation
            r_grid, z_grid = np.meshgrid(self.r_mesh, self.z_mesh)
            mesh_points = np.column_stack([r_grid.ravel(), z_grid.ravel()])
            
            rho = scipy.interpolate.griddata(
                self.measurement_points, 
                rho_input,
                mesh_points
            )
            
            rho = rho.reshape(len(self.r_mesh), len(self.z_mesh))
        
        else:
            # Create default density profile
            rho = self._create_default_rho()
            
        # Add extra column for density ratio calculations in IFD
        rho = np.column_stack([rho, rho[:, -1]])
        
        return rho

    def _create_default_rho(self):
        raise NotImplementedError("Child class must implement create_default_rho()")

    def _process_alpha(self, alpha_input):
        """Process attenuation profile."""
        nr = len(self.r_mesh)
        nz = len(self.z_mesh)
        
        if alpha_input is not None:
            # User provided attenuation - interpolate
            if self.measurement_points is None:
                raise ValueError("measurement_points required when providing alpha data")

            # Create mesh grid for interpolation
            r_grid, z_grid = np.meshgrid(self.r_mesh, self.z_mesh)
            mesh_points = np.column_stack([r_grid.ravel(), z_grid.ravel()])
            
            alpha = scipy.interpolate.griddata(
                self.measurement_points, 
                alpha_input,
                mesh_points
            )
            
            alpha = alpha.reshape(len(self.r_mesh), len(self.z_mesh))
        
        else:
            alpha = self._create_default_alpha()

        self.refractive_index2 = (self.c_0 / self.ssp)**2 * (1  + 1j * alpha / 27.29)
            
        return alpha

    def _create_default_alpha(self):
        raise NotImplementedError("Child class must implement _create_default_alpha()")

    def _create_absorbing_layer(self, alpha_ref):
        """Create an absorbing layer to approximate radiation conditions"""
    
        H = np.ptp(self.z_mesh)
        n_points = int(np.ceil((H / 3) / self.h)) # absorbing layer thickness = domain thickness / 3
        nr = len(self.r_mesh)
        
        z_mesh_absorbing_layer = np.arange(0, self.h * (n_points + 1), self.h)
        z_max = self.h * n_points
        D = (z_max - H)/3
    
        alpha_absorbing_layer = np.zeros((nr, len(z_mesh_absorbing_layer)))
        for i_range in range(nr):
            # alpha_absorbing_layer[i_range, :] = 1 + np.exp(-((z_mesh_absorbing_layer - z_max)/D)**2)
            alpha_absorbing_layer[i_range, :] = alpha_ref[i_range]*np.exp(np.log(1e5/alpha_ref[i_range])*z_mesh_absorbing_layer/z_max)
        
        return [alpha_absorbing_layer, z_mesh_absorbing_layer]

    def _add_absorbing_layer(self):
        raise NotImplementedError("Child class must implement _add_absorbing_layer()")
        
    def plot_environment(self, v_min = 1400, v_max = 1600):
        """Visualize the environment."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        biggest_range_ind = np.argmax(self.topography)
        biggest_range_ssp = self.ssp[biggest_range_ind, self.z_mesh < self.topography[biggest_range_ind]]
        rho_temp = self.rho[:,:-1]
        biggest_range_rho = rho_temp[biggest_range_ind, self.z_mesh < self.topography[biggest_range_ind]]
        biggest_range_alpha = self.alpha[biggest_range_ind, self.z_mesh < self.topography[biggest_range_ind]]

        ax = axes[0]
        im = ax.pcolormesh(self.r_mesh, self.z_mesh, self.ssp.T, shading='auto', vmin = np.min(biggest_range_ssp[1:-1]), vmax = np.max(biggest_range_ssp[1:-1]))
        ax.plot(self.r_mesh, self.topography, 'k-', lw=2)
        ax.set_title('Sound Speed [m/s]')
        plt.colorbar(im, ax=ax)
        
        # Density
        ax = axes[1]
        im = ax.pcolormesh(self.r_mesh, self.z_mesh, self.rho[:, :-1].T, shading='auto', vmin = np.min(biggest_range_rho[1:-1]), vmax = np.max(biggest_range_rho[1:-1]))
        ax.plot(self.r_mesh, self.topography, 'k-', lw=2)
        ax.set_title('Density [kg/m³]')
        plt.colorbar(im, ax=ax)
        
        # Attenuation
        ax = axes[2]
        im = ax.pcolormesh(self.r_mesh, self.z_mesh, self.alpha.T, shading='auto', vmin = np.min(biggest_range_alpha), vmax = np.max(biggest_range_alpha))
        ax.plot(self.r_mesh, self.topography, 'k-', lw=2)
        ax.set_title('Attenuation [dB/λ]')
        plt.colorbar(im, ax=ax)

        for i_axis in range(len(axes)):
            axes[i_axis].set_xlabel('Range [m]')
            axes[i_axis].invert_yaxis()
            axes[i_axis].set_ylabel('Depth [m]')        
        plt.tight_layout()
        plt.show()

class Air(PropagationEnvironment):

    """ A starter environment resembling an air environment using U.S. Standard 1962 """

    def __init__(self, 
                 f, z_0=None, max_range=None, z_lims=None, 
                 roughness=None, topography=None, topography_lims=None,
                 ssp=None, rho=None, alpha=None,
                 measurement_points=None):

        c_0 = 343
        self.rho_0 = 1.21
        self.us_standard_z = np.concatenate(
            (np.arange(-5000, 500, 500),
             np.array([50]),
             np.arange(500, 11500, 500),
             np.arange(12000, 33000, 1000),
             np.arange(34000, 52000, 2000),
             np.arange(55000, 90000, 5000),
             np.array([85500])
            ), axis = 0
        )
        self.name = 'air'
        
        super().__init__(f, c_0, z_0,
                         max_range, z_lims, 
                         roughness, topography, topography_lims,
                         ssp, rho, alpha,
                         measurement_points)

    def _create_default_ssp(self):
        """Use 1962 U.S. Standard Atmosphere"""
        c_measured = np.array([
            359, 357, 355, 353, 352, 349, 348, 346, 344, 342,
            340, 340, 338, 336, 334, 333, 330, 328, 327, 325,
            323, 321, 319, 316, 314, 312, 310, 308, 306, 304,
            302, 300, 297, 295, 295, 295, 295, 295, 295, 295,
            295, 295, 295, 296, 296, 297, 298, 298, 299, 300,
            300, 301, 302, 302, 303, 306, 310, 314, 318, 321,
            325, 328, 330, 330, 323, 315, 306, 297, 289, 283,
            276, 275
        ]) 
        
        x = np.polyfit(self.us_standard_z, c_measured, 4)
        ssp = x[0]*self.z_mesh**4 + x[1]*self.z_mesh**3 + x[2]*self.z_mesh**2 + x[3]*self.z_mesh + x[4]
        ssp = np.tile(ssp, [len(self.r_mesh), 1])

        for i_range in range(len(self.r_mesh)):
            ssp[i_range, self.z_mesh > self.topography[i_range]] = 1700 # some solid
        
        return ssp

    def _create_default_rho(self):

        rho_measured = np.array([
            1.911, 1.8491, 1.7697, 1.6930, 1.6189, 1.5473, 1.4782, 1.4114,
            1.3470, 1.2849, 1.2250, 1.2191, 1.1673, 1.1117, 1.0581, 1.0066, 0.95695,
            0.90925, 0.86340, 0.81935, 0.77704, 0.73643, 0.69747, 0.66011, 0.62431, 
            0.59002, 0.55719, 0.52579, 0.49576, 0.46706, 0.43966, 0.41351, 0.38857, 
            0.36392, 0.31083, 0.26548, 0.22675, 0.19367, 0.16542, 0.14129, 0.12068, 
            0.10400, 0.088910, 0.075715, 0.064510, 0.055006, 0.046938, 0.040084, 0.034257,
            0.029298, 0.025076, 0.021478, 0.018410, 0.015792, 0.013555, 0.0098874, 0.0072579, 
            0.0053666, 0.0039957, 0.0029948, 0.0022589, 0.0017142, 0.0013167, 0.0010269, 0.00056810, 
            0.00030968, 0.00016321, 0.000082829, 0.000039921, 0.000018458, 0.0000082196, 0.0000075641
        ])

        x = np.polyfit(self.us_standard_z, np.log(rho_measured), 1)
        rho = np.exp(x[0]*self.z_mesh + x[1])
        rho = np.tile(rho, [len(self.r_mesh), 1])
        
        for i_range in range(len(self.r_mesh)):
            rho[i_range, self.z_mesh > self.topography[i_range]] = 1700 # some solid 
        
        return rho

    def _create_default_alpha(self):
        
        # get dynamic viscosity
        mu_measured = 1.7894e-5*np.array([
            1.0854, 1.0770, 1.0686, 1.0602, 1.0517, 1.0432, 1.0346, 1.0260,
            1.0174, 1.0087, 1.0000, 0.9991, 0.9912, 0.9823, 0.9734, 0.9645, 0.9555,
            0.9465, 0.9374, 0.9283, 0.9191, 0.9099, 0.9006, 0.8913, 0.8819, 
            0.8724, 0.8629, 0.8534, 0.8438, 0.8341, 0.8244, 0.8146, 0.8047,
            0.7948, 0.7944, 0.7944, 0.7944, 0.7944, 0.7944, 0.7944, 0.7944,
            0.7944, 0.7944, 0.7973, 0.8003, 0.8034, 0.8064, 0.8094, 0.8124, 
            0.8154, 0.8184, 0.8214, 0.8244, 0.8274, 0.8304, 0.8461, 0.8624,
            0.8786, 0.8946, 0.9105, 0.9262, 0.9417, 0.9521, 0.9521, 0.9244,
            0.8850, 0.8447, 0.8034, 0.7689, 0.7381, 0.7067, 0.7036
        ])

        x = np.polyfit(self.us_standard_z, mu_measured, 4)
        mu = x[0]*self.z_mesh**4 + x[1]*self.z_mesh**3 + x[2]*self.z_mesh**2 + x[3]*self.z_mesh + x[4]
        mu = np.tile(mu, [len(self.r_mesh), 1])

        # Stoke's law of sound attenuation
        alpha = 2 * mu * (2*np.pi*self.f)**2 / (3 * self.rho[:,:-1] * self.ssp**3)

        alpha *= (self.ssp/self.f)*20*np.log10(np.exp(1))
        
        for i_range in range(len(self.r_mesh)):
            alpha[i_range, self.z_mesh > self.topography[i_range]] = 1e3  # Sediment
            alpha[i_range, -1] = 1e3 # infinitely attenuating atmosphere

        return alpha

    def _add_absorbing_layer(self):

        # bottom absorbing layer
        alpha_ref = self.alpha[:,-1]
        [alpha_absorbing_layer, z_mesh_absorbing_layer] = self._create_absorbing_layer(alpha_ref)
        ssp_ref = self.ssp[:,-1]
        rho_ref = self.rho[:,-1]
        self.alpha = np.concatenate((self.alpha, alpha_absorbing_layer), axis = 1)
        self.z_mesh = np.concatenate((self.z_mesh, self.z_mesh[-1] + z_mesh_absorbing_layer))
        self.ssp = np.concatenate((self.ssp, ssp_ref[:, np.newaxis]*np.ones_like(alpha_absorbing_layer)), axis = 1)
        self.rho = np.concatenate((self.rho, rho_ref[:, np.newaxis]*np.ones_like(alpha_absorbing_layer)), axis = 1)
        self.refractive_index2 = np.concatenate((
            self.refractive_index2,
            (self.c_0/ssp_ref[:, np.newaxis])**2 + 1j*alpha_absorbing_layer
        ), axis = 1)
        
        # top absorbing layer
        alpha_ref = self.alpha[:,0]
        [alpha_absorbing_layer, z_mesh_absorbing_layer] = self._create_absorbing_layer(alpha_ref)
        ssp_ref = self.ssp[:,0]
        rho_ref = self.rho[:,0]
        self.alpha = np.concatenate((np.flip(alpha_absorbing_layer, axis = 1), self.alpha), axis = 1)
        self.z_mesh = np.concatenate((self.z_mesh[0] - np.flip(z_mesh_absorbing_layer), self.z_mesh))
        self.ssp = np.concatenate((ssp_ref[:, np.newaxis]*np.ones_like(alpha_absorbing_layer), self.ssp), axis = 1)
        self.rho = np.concatenate((rho_ref[:, np.newaxis]*np.ones_like(alpha_absorbing_layer), self.rho), axis = 1)
        self.refractive_index2 = np.concatenate((           
            np.flip((self.c_0/ssp_ref[:, np.newaxis])**2 + 1j*alpha_absorbing_layer, axis = 1),
            self.refractive_index2,
        ), axis = 1)

class Water(PropagationEnvironment):

    def __init__(self, 
                 f, z_0=None, max_range=None, z_lims=None, 
                 roughness=None, topography=None, topography_lims=None,
                 ssp=None, rho=None, alpha=None,
                 measurement_points=None):
        c_0 = 1500

        self.name = 'water'
        self.standard_ocean_depths = np.array([
            5, 150, 300, 450, 550, 600, 700, 900,
            1000, 1250, 1600, 2000, 2500, 2997, 3500,
            4000, 4500, 5000, 5500, 6000, 6500, 7000
            ])

        super().__init__(f, c_0, z_0, max_range, z_lims, 
                 roughness, topography, topography_lims,
                 ssp, rho, alpha,
                 measurement_points)

    def _create_default_ssp(self):
        """Create default Munk profile."""
        nr = len(self.r_mesh)
        nz = len(self.z_mesh)
        mesh_ssp = np.zeros((nr, nz))

        T = np.array([
            24.04, 23.65, 23.01, 21.75, 19.77, 16.69,
            12.3, 7.96, 6.97, 5.59, 4.84, 4.48, 4.23,
            4.1, 3.93, 3.62, 3.29, 2.95, 2.51, 1.96, 1.52, 0.86
        ])
        S = np.array([
            35.28, 35.06, 34.52, 34.13, 34.07, 34.09,
            34.11, 34.31, 34.38, 34.51, 34.56, 34.61,
            34.65, 34.67, 34.62, 34.44, 34.08, 33.45,
            32.52, 31.2, 29.45, 27.19
        ])
        c_model = 1449.05 + 45.7*(T/10) - 5.21*(T/10)**2 + 0.23*(T/10)**3 + \
                (1.333 - 0.126*(T/10) + 0.009*(T/10)**2)*(S-35) + \
                16.3*(self.standard_ocean_depths/1000) + 0.18*(self.standard_ocean_depths/1000)**2
        c_func = scipy.interpolate.CubicSpline(self.standard_ocean_depths, c_model)
        c_z = c_func(self.z_mesh)
        
        for i_range in range(nr):
            mesh_ssp[i_range, :] = c_z
            mesh_ssp[i_range, self.z_mesh > self.topography[i_range]] = 1624 # Sediment
            mesh_ssp[i_range, 0] = 343 # Air
        
        return mesh_ssp

    def _create_default_rho(self):

        z_measured = np.array([0, 50, 95, 340, 350, 388, 481, 637, 863, 1121, 1322, 1500, 1683, 1859, 2027, 2200, 2445])

        rho_measured = np.array([
            1021.6466004861300, 1021.681470207400, 1021.8285023057200, 1027.338202221670, 
            1027.9607460075900, 1028.2855909679500, 1028.6281547443300, 1028.968390086550,
            1029.2752322755000, 1029.40891847073, 1029.46162059017, 1029.5129597237700,
            1029.5946252924700, 1029.6046205220200, 1029.6810045205700, 1029.7336498489400, 1029.8426887167500
        ])

        rho = np.interp(self.z_mesh, z_measured, rho_measured)
        rho = np.tile(rho, [len(self.r_mesh), 1])
        for i_range in range(len(self.r_mesh)):
                rho[i_range, self.z_mesh > self.topography[i_range]] = 1700  # Sediment
                rho[i_range, 0] = 1.21 # Air

        return rho

    def _create_default_alpha(self, pH = 8.1, S = 35, T = 20):

        # proportionality factors [dB/(km*kHz)]
        A_1 = 8.86/self.ssp * 10**(0.78 * pH - 5)
        A_2 = 21.44*S/self.ssp * (1 + 0.025 * T)
        if T <= 20:
            A_3 = 4.937e-4 - 2.59e-5*T + 9.11e-7*T**2 - 1.50e-8*T**3
        else:
            A_3 = 3.964e-4 - 1.146e-5*T + 1.45e-7*T**2 - 6.5e-10*T**3

        # dimensionless depth functions [1] 
        P_1 = 1
        P_2 = 1 - 1.37e-4*self.z_mesh + 6.2e-9*self.z_mesh**2
        P_3 = 1 - 3.83e-5*self.z_mesh + 4.9e-10*self.z_mesh**2

        # relaxation frequencies [kHz]
        f_1 = 2.8 * (S/35)**0.5 * 10**(4 - 1245*(T + 273))
        f_2 = 8.17 * 10**(8-1990/(T + 273)) / (1 + 0.0018*(S - 35))

        # putting alpha together
        boric_acid_term = A_1*P_1*f_1*self.f**2 / (self.f**2 + f_1**2)
        magnesium_sulfate_term = A_2*P_2*f_2*self.f**2 / (self.f**2 + f_2**2)
        pure_water_term = A_3*P_3*self.f**2

        alpha = boric_acid_term + magnesium_sulfate_term + pure_water_term # dB/km
        alpha *= 1e-3 / (self.c_0 / self.f)
        max_alpha = np.max(alpha)

        for i_range in range(len(self.r_mesh)):
            alpha[i_range, self.z_mesh > self.topography[i_range]] = 100 / (self.ssp[i_range, self.z_mesh > self.topography[i_range]]/self.f)
        
        return alpha

    def _add_absorbing_layer(self):

        # bottom absorbing layer
        alpha_ref = self.alpha[:,-1]
        [alpha_absorbing_layer, z_mesh_absorbing_layer] = self._create_absorbing_layer(alpha_ref)
        ssp_ref = self.ssp[:,-1]
        rho_ref = self.rho[:,-1]
        self.alpha = np.concatenate((self.alpha, alpha_absorbing_layer), axis = 1)
        self.z_mesh = np.concatenate((self.z_mesh, self.z_mesh[-1] + z_mesh_absorbing_layer))
        self.ssp = np.concatenate((self.ssp, ssp_ref[:, np.newaxis]*np.ones_like(alpha_absorbing_layer)), axis = 1)
        self.rho = np.concatenate((self.rho, rho_ref[:, np.newaxis]*np.ones_like(alpha_absorbing_layer)), axis = 1)
        self.refractive_index2 = np.concatenate((
            self.refractive_index2,
            (self.c_0/ssp_ref[:, np.newaxis])**2 + 1j*alpha_absorbing_layer
        ), axis = 1)
        