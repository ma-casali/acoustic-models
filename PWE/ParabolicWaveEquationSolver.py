import numpy as np
from numba import jit
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os
import cmocean
from mpl_toolkits import basemap
import warnings

sys.path.append(os.path.abspath('../acoustic-models'))

from PWE.GenerateRadialTopography import TopographyRadials
from PWE.PropagationEnvironment import PropagationEnvironment, Water, Air

def u0_gaussian_generalized(z, z_s, k_0, theta_half_bw = np.radians(45), theta_tilt = np.radians(0)):
    """
    Gaussian initial field with configurable beamwidth and steering.
    
    Parameters:
    -----------
    z : array-like
        Depth array [m]
    z_s : float
        Source depth [m]
    k_0 : float
        Reference wavenumber [1/m]
    
    Returns:
    --------
    u_0 : array-like
        Initial field amplitude
    """

    if theta_half_bw <= np.radians(45):
        # generalized gaussian source
        exp_term1 = -1 * k_0**2 / 2 * (z - z_s)**2 * np.tan(theta_half_bw)**2
        exp_term2 = 1j * k_0 * (z - z_s) * np.sin(theta_tilt)
        return np.sqrt(k_0) * np.tan(theta_half_bw) * np.exp(exp_term1) * np.exp(exp_term2)
    
    else:
        # greene's source with angular tilt
        exp_term1 = -k_0**2 * (z - z_s)**2 / 3.0512
        exp_term2 = 1j * k_0 * (z - z_s) * np.sin(theta_tilt)
        return np.sqrt(k_0) * (1.4467 - 0.4201 * k_0**2 * (z - z_s)**2) * np.exp(exp_term1) * np.exp(exp_term2)



class ParabolicWaveEquationSolver:
    """
    Optimized IFD solver with LU factorization caching and vectorized operations.
    """
    
    def __init__(self, environment: PropagationEnvironment, type_of_solve = 'standard'):
        
        """
        A class representing a parabolic wave equation solver via the IFD method.
        """

        self.env = environment
        self.type_of_solve = type_of_solve
        self.f = environment.f
        self.r_mesh = environment.r_mesh
        self.z_mesh = environment.z_mesh
        self.h = environment.h
        self.k = environment.k
        self.ssp = environment.ssp
        self.rho = environment.rho
        self.alpha = environment.alpha
        self.topography = environment.topography
        self.refractive_index2 = environment.refractive_index2
        
        # Reference parameters
        self.c_0 = environment.c_0
        self.rho_0 = self.rho[0, 0]
        self.k_0 = 2 * np.pi * self.f / self.c_0
                
        # Pre-allocate result array
        self.u = np.zeros((len(self.r_mesh), len(self.z_mesh)), dtype=complex)

    @staticmethod
    @jit(nopython=True, cache=True)
    def thomas_algorithm(lower, main, upper, rhs):
        """
        Thomas algorithm (tridiagonal matrix solver).
        Solves Ax = b where A is tridiagonal.
        Note: This modifies the input arrays, so pass copies if needed.
        """
        n = len(main)
        x = np.zeros(n, dtype=np.complex128)
        
        # Forward elimination
        for i in range(1, n):
            factor = lower[i-1] / main[i-1]
            main[i] = main[i] - factor * upper[i-1]
            rhs[i] = rhs[i] - factor * rhs[i-1]
        
        # Back substitution
        x[n-1] = rhs[n-1] / main[n-1]
        for i in range(n-2, -1, -1):
            x[i] = (rhs[i] - upper[i] * x[i+1]) / main[i]
        
        return x
    
    def solve_tridiagonal_matrix_eq(self, A, B, u_curr):
    
        lower = A.diagonal(-1).copy()
        main = A.diagonal(0).copy()
        upper = A.diagonal(1).copy()
    
        rhs = (B @ u_curr).copy()
    
        return self.thomas_algorithm(lower, main, upper, rhs)
    
    def build_system_matrices(self, nn, k, k_0_sq_h_sq, sqrt_approx_vec):

        # Using equations outlined in Computational Ocean Acoustics (Jensen)
        [a_0, a_1, b_0, b_1] = sqrt_approx_vec # Eq. 6.177
        
        refractive_index2 = self.refractive_index2
        M = len(self.z_mesh)

        w_1 = b_0 + (1j*self.k_0*k/2)*(a_0 - b_0) # Eq. 6.180
        w_2 = b_1 + (1j*self.k_0*k/2)*(a_1 - b_1) # Eq. 6.182

        # Eq. 6.187
        rho_1_reduced = self.rho[:,:-1]
        rho_2_reduced = self.rho[:,1:]
        rho_ratio = rho_1_reduced/rho_2_reduced

        # Eq. 6.186
        X = (rho_ratio + 1)*(k_0_sq_h_sq/2 * (np.conj(w_1)/np.conj(w_2)) - 1) + \
        k_0_sq_h_sq * ((refractive_index2 - 1)*(rho_ratio + 1))

        # Eq. 6.188
        Y = (rho_ratio + 1)*(k_0_sq_h_sq/2 * (w_1/w_2) - 1) + \
        k_0_sq_h_sq * ((refractive_index2 - 1)*(rho_ratio + 1))

        # the following matrices are defined in Eq. 6.189
        upper_diag_A = rho_ratio[nn+1,:]
        main_diag_A = X[nn+1,:]
        lower_diag_A = np.ones_like(rho_ratio[nn+1,:])

        upper_diag_B = rho_ratio[nn,:]*(w_2/np.conj(w_2))
        main_diag_B = Y[nn,:]*(w_2/np.conj(w_2))
        lower_diag_B = np.ones_like(rho_ratio[nn,:])*(w_2/np.conj(w_2))

        A = sp.diags(
            [lower_diag_A, main_diag_A, upper_diag_A],
            offsets=[-1, 0, 1],
            shape=(M, M),
            format='csr'
        )
        B = sp.diags(
            [lower_diag_B, main_diag_B, upper_diag_B],
            offsets=[-1, 0, 1],
            shape=(M, M),
            format='csr'
        )
        
        return A, B
    
    def build_q_operator(self, nn = 0):
        # constructs the Q operator matrix for a given range index nn
        # Q = (1/k_0^2) * d^2/dz^2 + (refractive_index^2 - 1)
        
        nz = len(self.z_mesh)
        h_sq = self.h**2
        k0_sq = self.k_0**2

        # second derivative coefficients (1/h^2 * [-1, 2, 1])
        diag_val = -2.0 / (h_sq * k0_sq)
        off_diag_val = 1.0 / (h_sq * k0_sq)

        n2_minus_1 = self.refractive_index2[nn, :] - 1.0

        main_diag = diag_val + n2_minus_1
        upper_diag = np.full(nz - 1, off_diag_val)
        lower_diag = np.full(nz - 1, off_diag_val)

        Q = sp.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format = 'csr')
        return Q
    
    def generate_collins_starter(self, z_s):

        self_starter_coeffs = {
            'a': [0.000155620, 0.001995837, 0.007606368, 0.019183490, 0.038165502, 0.063855792, 0.093125232, 0.117203893],
            'b': [0.000523258, 0.004124967, 0.013444457, 0.030801314, 0.058379268, 0.098485210, 0.154143223, 0.229415410]
        }

        nz = len(self.z_mesh)
        u_0 = np.zeros(nz, dtype = complex)

        # create delta function at z_s
        src_idx = np.argmin(np.abs(self.z_mesh - z_s))
        delta_src = np.zeros(nz, dtype = complex)
        delta_src[src_idx] = 1.0 / self.h

        # build Q operator matrix
        Q = self.build_q_operator() # nn=0 for initial range

        for a, b in zip(self_starter_coeffs['a'], self_starter_coeffs['b']):
            LHS = sp.eye(nz) + b * Q
            RHS = a * delta_src

            u_0 += spsolve(LHS, RHS)

        return u_0 * np.exp(1j * self.k_0 * 0) # phase alignment
    
    def solve(self, u0_func, z_s, progress_callback = None, theta_half_bw = np.radians(45), theta_tilt = np.radians(0), use_self_starter=False):
        """
        Solve the parabolic equation with optimized range marching.
        
        INPUTS
        u0_func: function to generate initial field u0(z, z_s, k_0, k)
        z_s: depth of source
        
        RETURNS
        u: complex field array (range x depth)
        """
        nr = len(self.r_mesh)
        nz = len(self.z_mesh)
        self.u = np.zeros((nr, nz), dtype = complex)
        
        # Initialize field
        if use_self_starter:
            self.u[0, :] = self.generate_collins_starter(z_s)
        else:
            if theta_half_bw > np.radians(45):
                warnings.warn("You have chosen to use a wide angle without using the Collin's starter, which is not recommended.")
            if (z_s < self.c_0/self.f) and (self.env.name == 'water'):
                self.u[0, :] = u0_func(self.z_mesh, z_s, self.k_0, theta_half_bw, theta_tilt) - u0_func(self.z_mesh, -1*z_s, self.k_0, theta_half_bw, -theta_tilt)
                print('Source is within a wavelength of the surface, incorporating mirror source')
            else:
                self.u[0, :] = u0_func(self.z_mesh, z_s, self.k_0, theta_half_bw, theta_tilt)

        if self.topography[0] - z_s < self.c_0/self.f:
            raise ValueError(f"Source is within a wavelength of the sea floor and initial value is ill-poised for this solution type.\n \
                             Source Depth: {z_s:.3f} m,\n \
                             Bottom Depth: {self.topography[0]:.3f} m\n \
                             Wavelength: {self.c_0/self.f:.3f}")
            
        # Pre-compute items for the solver

        # sqrt approximation coefficients
        if self.type_of_solve == 'standard': # Tappert
            sqrt_approx_vec = [1, 0.5, 1, 0]
        elif self.type_of_solve == 'wide-angle': # Claerbout
            sqrt_approx_vec = [1, 0.75, 1, 0.25]
        elif self.type_of_solve == 'high-angle': # Greene
            sqrt_approx_vec = [0.99987, 0.79624, 1, 0.30102]
        else:
            raise ValueError(f"Unknown solve type: {self.type_of_solve}")

        # Pre-compute all k values 
        k_steps = np.diff(self.r_mesh)

        # Pre-compute constants
        k_0_sq_h_sq = self.k_0**2 * self.h**2

        for nn in range(len(self.r_mesh) - 1):

            # From Eq. 6.189 (Jensen, Computational Ocean Acoustics)
            A, B = self.build_system_matrices(nn, k_steps[nn], k_0_sq_h_sq, sqrt_approx_vec)
            self.u[nn+1, :] = self.solve_tridiagonal_matrix_eq(A, B, self.u[nn, :])

            # set boundaries to zero pressure
            self.u[nn+1, 0] = 0 # pressure release
            self.u[nn+1, -1] = 0 # absorbing layer
            
            if progress_callback is not None:
                progress_callback(nn, len(self.r_mesh) - 1)
        
        return self.u

    def plot(self, fig = None, ax = None, min_TL = -60):
        z_max = self.env.z_lims[1]
        def reverse_label(z, pos):
            return f"{int(z_max - z)}"

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize = (15, 5))

        plot_u_dB = 20*np.log10(np.abs(self.u[:,1:-1]/np.max(self.u[0,1:-1])))
        for iR in range(len(self.r_mesh)):
            plot_u_dB[iR, self.z_mesh[1:-1] > self.topography[iR]] = min_TL - 10

        im = ax.pcolormesh(
            self.r_mesh,
            self.z_mesh[1:-1],
            np.transpose(plot_u_dB),
            vmin = min_TL,
            shading='gouraud',
            cmap = 'turbo'
        )
        ax.plot(self.r_mesh, self.topography, 'w')
        if self.r_mesh[-1] > 5e3:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
            ax.set_xlabel('Range [km]')
        else:
            ax.set_xlabel('Range [m]')
        ax.set_ylim([0,self.env.z_lims[1]])

        # create and store the colorbar so it can be removed on next plot
        cbar = fig.colorbar(im, ax=ax, label='TL [dB]')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(np.abs(x)):d}'))

        ax.invert_yaxis()
        if self.env.name == 'water':
            ax.set_ylabel('Depth [m]')
        elif self.env.name == 'air':
            ax.set_ylabel('Altitude [m]')
            ax.yaxis.set_major_formatter(FuncFormatter(reverse_label))
            ax.get_xticklabels
            
        plt.savefig('Figures/sample_figure.png')

if __name__ == "__main__":

    topo_file = 'PWE/Data/Topography/MontereyBay.tiff'
    mars_coords = np.array([36 + 42.7481/60, -122 - 11.2139/60])
    topo_radials = TopographyRadials(center_coords=mars_coords, topo_file=topo_file)
    frequency = 44 # Hz, third harmonic of blue whale b-call

    bearings = np.arange(-180, -15, 15) # degrees
    for i, th in enumerate(bearings):

        layout_format = [['.', '.', 'B', 'B', 'B', 'B', 'B', 'B'],
                         ['A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'],
                         ['A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'],
                         ['.', '.', 'B', 'B', 'B', 'B', 'B', 'B'],
                         ['.', '.', 'C', 'C', 'C', 'C', 'C', '.'],
                         ['.', '.', 'C', 'C', 'C', 'C', 'C', '.'],
                         ['.', '.', 'C', 'C', 'C', 'C', 'C', '.']]
        fig, ax = plt.subplot_mosaic(layout_format, figsize = (15, 7))

        m = basemap.Basemap(llcrnrlon=topo_radials.box_bounds.left,
            llcrnrlat=topo_radials.box_bounds.bottom,
            urcrnrlon=topo_radials.box_bounds.right,
            urcrnrlat=topo_radials.box_bounds.top,
            projection = 'lcc', resolution = 'i',
            lat_1 = topo_radials.box_bounds.bottom,
            lat_0 = mars_coords[0], lon_0 = mars_coords[1], ax = ax['A'])
        
        m.drawcoastlines()
        m.fillcontinents(color='green')

        r_max = 200e3
        dr = 1e3
        r_line = np.arange(0, r_max, dr)
        lat_line, lon_line = topo_radials.get_destination_point(r_line, np.array([th]))

        topo = topo_radials.topo_data
        topo[topo < 0] = np.nan
        topo[topo > 3e3] = np.nan
        cmap = cmocean.cm.deep

        m.pcolormesh(topo_radials.LON, topo_radials.LAT, topo, shading = 'nearest', cmap = cmap, latlon = True)
        m.plot(lon_line, lat_line, 'r', linewidth = 2, latlon = True)
        m.scatter(mars_coords[1], mars_coords[0], s = 50, marker = 'o', color = 'r', latlon = True)
        m.scatter(lon_line[-1], lat_line[-1], s = 40, marker = '^', color = 'r', latlon=True)
        ax['A'].set_title("Radial from MARS")

        environment = Water(f=frequency, topography_method='radial', max_range = r_max, z_lims=[0, 3e3], bearing = np.array([th]))
        pwe_solver = ParabolicWaveEquationSolver(environment=environment)
        u = pwe_solver.solve(u0_gaussian_generalized, z_s = environment.topography[0] - environment.c_0/frequency, theta_half_bw=np.radians(78), theta_tilt = np.radians(90), use_self_starter=True)

        pwe_solver.plot(fig = fig, ax = ax['B'], min_TL = -120)
        ax['B'].scatter(0, environment.z_lims[-1], s = 50, marker = 'o', color = 'r')
        ax['B'].scatter(r_max, environment.z_lims[-1], s = 50, marker = '^', color = 'r')
        ax['B'].set_title(f"TL [dB] along radial of bearing {th}° N")

        whale_point = 14.8 # m (Oestreich et al.)
        u_dB = 20 * np.log10(np.abs(u) / np.max(np.abs(u)))
        d_point = np.where(environment.z_mesh - whale_point > 0)[0][0]

        ax['C'].plot(environment.r_mesh, u_dB[:, d_point].flatten())
        ax['C'].grid()
        ax['C'].set_ylim([-120, 0])
        ax['C'].set_xlim([0, r_max])
        ax['C'].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{np.abs(x):.0f}'))
        ax['C'].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
        ax['C'].set_xlabel('Range [km]')
        ax['C'].set_ylabel('TL [dB] @ depth = 14.8 m')
        plt.tight_layout()

        plt.savefig(f"PWE/Data/Figures/pwe_mars_f{frequency}Hz_b{th:d}N.png", dpi = 300, bbox_inches = 'tight')

    plt.show()