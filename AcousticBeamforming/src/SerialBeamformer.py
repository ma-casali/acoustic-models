from matplotlib import style
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from BeamformingArray import BeamformingArray, ElementDirectivity
from ArrayShading import ArrayShading
from SerialAnalyzer import SerialAnalyzer
from BeamformingModel import BeamformingModel
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys

class SerialBeamformer:

    def __init__(self, serial_analyzer: SerialAnalyzer, beamforming_model: BeamformingModel, steer_az: np.ndarray, steer_de: np.ndarray):
        
        self.serial_analyzer = serial_analyzer

        self.beamforming_model = beamforming_model
        self.array_shading = ArrayShading(beamforming_model.array) 

        self.shading_vector = self.array_shading.compute_raised_cosine_window(p = 0.5)
        self.steering_vector = self.beamforming_model.compute_steering_vector(steer_az, steer_de, self.beamforming_model.array.design_frequency)
        self.steering_vector *= self.shading_vector[np.newaxis, :] # apply shading to steering vector

        # flatten steering vector angular dimensions from (Az, De, NumElements) to (Az*De, NumElements)
        if self.steering_vector.ndim == 3:
            self.steering_vector = np.reshape(self.steering_vector, (np.prod(self.steering_vector.shape[0:2]), self.steering_vector.shape[2])).T # shape (num_elements, Az*De)

    def apply_spatial_filter_rt(self, signal: np.ndarray, steering_vector: np.ndarray) -> np.ndarray:

        """
        Docstring for apply_spatial_filter_rt
        
        :param self: instance of SerialBeamformer
        :param signal: time series data from the microphone array, shape (num_samples, num_elements)
        :param steering_vector: precomputed steering vector and shading vector, shape (num_elements, num_angles)
        :return: beamformed output signal, shape (num_samples, num_angles)
        """

        # apply spatial filter to each time sample

        beam_time_series = signal @ steering_vector # shape (num_samples, Az*De)
        beam_power = np.sum(np.abs(beam_time_series)**2, axis=0) # shape (Az*De,)

        return beam_time_series, beam_power

class BeamformingVisualizer(QtWidgets.QWidget):

    def __init__(self, bf_array: BeamformingArray):
        super().__init__()

        # Start the serial analyzer and beamformer
        self.serial_analyzer = SerialAnalyzer(port = '/dev/cu.usbmodem196242501', window_size=4096, num_elements=12)
        self.beamforming_model = BeamformingModel(bf_array)
        
        # Start the serial beamformer
        az = np.radians(np.arange(-180, 180, 5)) # steer from -90 to 90 degrees in azimuth
        de = np.radians(np.arange(0, 90, 5)) # steer from -90 to 90 degrees in elevation
        self.steer_az, self.steer_de = np.meshgrid(az, de, indexing = 'ij') # create a grid of steering angles
        self.serial_beamformer = SerialBeamformer(self.serial_analyzer, self.beamforming_model, self.steer_az, self.steer_de)

        # Create the CORNERS (Vertices) - these are N+1
        az_corners = np.radians(np.arange(-180, 185, 5)) # -180 to 180 inclusive
        de_corners = np.radians(np.arange(0, 95, 5))    # 0 to 90 inclusive

        # Create the mesh for corners
        self.mesh_x = de_corners * np.cos(az_corners[:, np.newaxis])
        self.mesh_y = de_corners * np.sin(az_corners[:, np.newaxis])

        self.layout = QtWidgets.QVBoxLayout(self)
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win)

        self.p1 = self.win.addPlot(title="Polar Acoustic Camera")
        self.p1.setAspectLocked(True) # Keep the circles circular!  
        
        # make the plot as big as possible
        self.p1.hideAxis('bottom')
        self.p1.hideAxis('left')
        self.p1.layout.setContentsMargins(0, 0, 0, 0)
        self.p1.setMenuEnabled(False)
        self.p1.getViewBox().setAspectLocked(True)
        self.p1.getViewBox().setDefaultPadding(0)
        max_r = np.radians(90)
        self.p1.setRange(xRange=[-max_r, max_r], yRange=[-max_r, max_r], padding=0)
        
        # 3. Create the PColorMeshItem
        # We pass the X and Y coordinate grids here
        self.mesh = pg.PColorMeshItem(x=self.mesh_x, y=self.mesh_y, edgeColor=None)
        self.mesh.setColorMap(pg.colormap.get('turbo'))
        self.p1.addItem(self.mesh)

        self.add_polar_grid()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30) # ~33 FPS

    def update(self):
        # Read Serial Data
        self.serial_analyzer.update_buffer()

        # Use a generator expression and avoid list() if possible
        signal_block = np.array(self.serial_analyzer.data_buffer).T # Shape (4096, 12)

        _, heatmap = self.serial_beamformer.apply_spatial_filter_rt(signal_block, self.serial_beamformer.steering_vector)
        heatmap = heatmap.reshape(self.steer_de.shape) # reshape back to (Az, De)

        # Update the ImageItem
        heatmap = heatmap / np.max(heatmap) # normalize for visualization
        heatmap_db = 10 * np.log10(heatmap + 1e-6) # convert to dB scale, add small value to avoid log(0)

        self.mesh.setData(self.mesh_x, self.mesh_y, heatmap_db)

    def add_polar_grid(self):
        # Draw concentric circles for Elevation
        for deg in [30, 60, 90]:
            r = np.radians(deg)
            circle = QtWidgets.QGraphicsEllipseItem(-r, -r, r*2, r*2)
            circle.setPen(pg.mkPen(255, 255, 255, 50, style=QtCore.Qt.DashLine))
            self.p1.addItem(circle)
            
        # Draw Azimuth spokes
        for deg in [-90, -60, -30, 0, 30, 60, 90]:
            rad = np.radians(deg)
            r_max = np.radians(90)
            line = pg.PlotCurveItem([0, r_max * np.sin(rad)], 
                                    [0, r_max * np.cos(rad)], 
                                    pen=pg.mkPen(255, 255, 255, 50))
            self.p1.addItem(line)

if __name__ == "__main__":

    dZ = (1 + 3/16) * 2.54 * 1e-2 # cm
    dY = (2 + 1/16) * 2.54 * 1e-2 # cm
    d = (3 + 7/8) * 2.54 * 1e-2 # cm
    frequency = 343 / (3 * dY)

    print("Element Veritcal spacing (m):", dZ)
    print("Element Horizontal spacing (m):", dY)
    print("Design frequency (Hz):", frequency)

    # Array Element Numbering (based on teensy prototype):
    # 8  7  T  5  6
    # 10 9  T  3  4
    # 12 11 T  1  2

    X = np.zeros(12)
    Y = np.array([d/2, d/2 + dY, d/2, d/2 + dY, d/2, d/2 + dY, -d/2, -d/2 - dY, -d/2, -d/2 - dY, -d/2, -d/2 - dY])
    Z = np.array([-dZ, -dZ, 0, 0, dZ, dZ, dZ, dZ, 0, 0, -dZ, -dZ])

    bf_array = BeamformingArray(X=X,
                                Y=Y,
                                Z=Z,
                                design_frequency=frequency,
                                element_directivity=ElementDirectivity.DIPOLE)

    app = QtWidgets.QApplication(sys.argv)
    viz = BeamformingVisualizer(bf_array)
    viz.show()
    sys.exit(app.exec_())
    


