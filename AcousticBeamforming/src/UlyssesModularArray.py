import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys

from BeamformingModel import BeamformingModel, BeamformingPlot
from BeamformingArray import BeamformingArray, ElementDirectivity
from ArrayShading import ArrayShading

sys.path.append(os.path.abspath('../acoustic-models'))

if __name__ == '__main__':

    save_path = os.path.dirname(os.path.abspath(__file__))

    # modular array is an array attached to the hull of the AUV
    # diameter of mako is ~ 8 in. ~= 22 cm

    R = 0.11 # m
    width = 0.25 # m (modular width)

    c_0 = 1460 # m/s
    f_hi = 1e4 # Hz
    f_lo = 1e2 # Hz

    aperture_angle = np.radians(90)

    horizontal_spacing = c_0 / f_hi / 2
    circumferential_spacing = c_0 / f_hi / 2
    angular_spacing = circumferential_spacing / R # radians

    theta = np.arange(-aperture_angle/2, aperture_angle, angular_spacing)
    theta -= np.mean(theta) # center the elements on the horizontal axis

    n_z = len(theta)
    n_y = int(width // horizontal_spacing + 1) * 2

    X = R * np.cos(theta)
    X = np.tile(X, (n_y,))

    Y = np.linspace(0, n_y * horizontal_spacing, n_y)
    Y -= np.mean(Y)
    Y = np.repeat(Y, repeats=n_z)

    Z = R * np.sin(theta)
    Z = np.tile(Z, (n_y,))

    array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.BAFFLED_DIPOLE)
    bf_model = BeamformingModel(array=array, c=1460)
    plotter = BeamformingPlot(bf_model)

    az, de, bp = bf_model.compute_beampattern(frequency=f_hi, c=1460)

    array.plot_array_geometry()
    ax = plt.gca()
    ax.view_init(elev=15, azim=-30, roll=0)
    fig = plt.gcf()
    fig.suptitle(f"Modular Array for {f_hi/1e3:.0f} kHz Design Frequency")
    ax.set_title(f"Modules needed for minimum low-frequency resolution @ {f_lo:.0f} Hz: {np.ceil(width/(2 * c_0 / f_lo)):.0f}")
    plt.savefig(os.path.join(save_path, f"Data/Figures/ModularArrayGeometry_{f_lo:.0f}Hz_{f_hi/1e3:.0f}kHz.png"), dpi = 300)

    plotter.plot_beampattern_image(beampattern=bp, az=az, de=de, method='polar')
    plt.savefig(os.path.join(save_path, f"Data/Figures/ModularbeamPattern_{f_hi/1e3:.0f}kHz.png"), dpi = 300)

    plt.show()


