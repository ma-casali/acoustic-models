import numpy as np
from numba import jit
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu
from tqdm import tqdm
import matplotlib.pyplot as plt
from PropagationEnvironment import PropagationEnvironment

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
    
    def solve(self, u0_func, z_s, progress_callback = None):
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
        if (z_s < self.c_0/self.f) and (self.env.name == 'water'):
            self.u[0, :] = u0_func(self.z_mesh, z_s, self.k_0) - u0_func(self.z_mesh, -1*z_s, self.k_0)
            print('Source is within a wavelength of the surface, incorporating mirror source')
        else:
            self.u[0, :] = u0_func(self.z_mesh, z_s, self.k_0)

        if self.topography[0] - z_s < self.c_0/self.f:
            raise ValueError('Source is within a wavelength of the sea floor and initial value is ill-poised for this solution type.')
            
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
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize = (15, 5))

        im = ax.pcolormesh(
            self.r_mesh,
            self.z_mesh[1:-1],
            np.transpose(20*np.log10(np.abs(self.u[:,1:-1]/np.max(self.u[0,1:-1])))),
            vmin = min_TL,
            shading='gouraud',
            cmap = 'turbo'
        )
        ax.plot(self.r_mesh, self.topography, 'w')
        ax.set_xlabel('Range [m]')
        ax.set_ylim([0,self.env.z_lims[1]])

        # create and store the colorbar so it can be removed on next plot
        cbar = fig.colorbar(im, ax=ax, label='TL [dB]')

        ax.invert_yaxis()
        ax.set_ylabel('Depth [m]')

        plt.savefig('Figures/sample_figure.png')
