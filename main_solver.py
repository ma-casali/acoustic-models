import numpy as np
from PropagationEnvironment import Air, Water
from ParabolicWaveEquationSolver import ParabolicWaveEquationSolver


def u0_gaussian_generalized(z, z_s, k_0):
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
    theta_half_bw = 45 * np.pi / 180  # beamwidth [rad]
    theta_tilt = 0 * np.pi / 180      # beam-steering [rad]
    return np.sqrt(k_0) * np.tan(theta_half_bw) * np.exp(
        -1 * k_0**2 / 2 * (z - z_s)**2 * np.tan(theta_half_bw)**2
    ) * np.exp(-1j * k_0 * (z - z_s) * np.sin(theta_tilt))

def run_solver_from_gui(input_dict, fig, ax, progress_callback = None):
    """
    Main function to create and run the solver.
    """
    roughness = 5     # m - roughness parameter for topography generation
    
    if input_dict['Environment'] == 'Air':
        # Create Air environment
        print("Creating Air environment...")
        env = Air(
            f=int(input_dict['f']),
            z_0=int(input_dict['z_0']),
            max_range=int(input_dict['r_max']),
            z_lims=[0, int(input_dict['z_max'])],
            topography_lims = [int(input_dict['topo_min']), int(input_dict['topo_max'])],
            roughness=roughness
        )
    else:
        # Create Water environment
        print("Creating Air environment...")
        env = Water(
            f=int(input_dict['f']),
            z_0=int(input_dict['z_0']),
            max_range=int(input_dict['r_max']),
            z_lims=[0, int(input_dict['z_max'])],
            topography_lims = [int(input_dict['topo_min']), int(input_dict['topo_max'])],
            roughness=roughness
        )
        
    # Create solver with the air environment
    print("\nCreating ParabolicWaveEquationSolver...")
    solver = ParabolicWaveEquationSolver(
        environment=env,
        type_of_solve='standard'  # Options: 'standard', 'wide-angle', 'high-angle'
    )
    
    # Solve the parabolic wave equation
    print("\nSolving parabolic wave equation...")
    u = solver.solve(u0_gaussian_generalized, z_s=int(input_dict['z_s']), progress_callback = progress_callback)
    
    # Plot the results
    print("\nPlotting results...")
    solver.plot(fig = fig, ax = ax, min_TL=-60)
