import sys
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QTextEdit)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from BeamformingModel import BeamformingModel, BeamformingPlot
from BeamformingArray import BeamformingArray, ElementDirectivity

class ParetoWeightGUI(QMainWindow):
    def __init__(self, pareto_front_values, pareto_front_states, labels):
        super().__init__()
        self.setWindowTitle("Pareto Front Weight Assignment")
        
        # Data Setup
        self.pareto_front_values = pareto_front_values
        self.pareto_front_states = pareto_front_states
        self.num_objectives = self.pareto_front_values.shape[1]
        
        self.ideal = np.min(self.pareto_front_values, axis=0)
        self.nadir = np.max(self.pareto_front_values, axis=0)
        # Prevent division by zero if all points have the same value for an objective
        diff = self.nadir - self.ideal
        diff[diff == 0] = 1.0
        self.norm_front = (self.pareto_front_values - self.ideal) / diff
        
        # Initial Weights
        self.weights = np.ones(self.num_objectives) / self.num_objectives

        self.labels = labels
        self.cmap = plt.get_cmap('tab10')
        
        self.init_ui()
        self.update_analysis()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Side: Controls
        left_layout = QVBoxLayout()
        self.sliders = []
        self.weight_labels = []

        for i in range(self.num_objectives):
            label = QLabel(f"{labels[i]} weight: {self.weights[i]:.2f}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(self.weights[i] * 100))
            slider.valueChanged.connect(self.on_slider_change)
            
            left_layout.addWidget(label)
            left_layout.addWidget(slider)
            self.sliders.append(slider)
            self.weight_labels.append(label)

        left_layout.addStretch()
        
        # Result Display: Show a histogram of the values as you change the weights
        self.histogram_canvas = FigureCanvas(Figure(figsize=(3,5)))
        self.ax_hist = self.histogram_canvas.figure.add_subplot(111)
        left_layout.addWidget(self.histogram_canvas, 1)

        right_layout = QVBoxLayout()

        self.array_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax_array = self.array_canvas.figure.add_subplot(111)
        self.ax_array.set_aspect('equal')
        right_layout.addWidget(self.array_canvas)

        # frequency slider for different subarrays
        self.f_select = 0 
        self.freq_label = QLabel(f"Frequency Selector for Subarrays: {self.f_select:.2f} Hz")
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 100)
        self.freq_slider.setValue(0)
        self.freq_slider.valueChanged.connect(self.on_freq_slider_change)
        right_layout.addWidget(self.freq_label)
        right_layout.addWidget(self.freq_slider)

        self.band_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.ax_bands = self.band_canvas.figure.add_subplot(111)
        right_layout.addWidget(self.band_canvas)
        
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

    def on_slider_change(self):
        # Update weights from sliders
        self.f_select = 0.0
        self.freq_label.setText(f"Frequency Selector for Subarrays: {self.f_select:.2f} Hz")
        raw_weights = np.array([s.value() for s in self.sliders])
        total = np.sum(raw_weights)
        
        if total > 0:
            self.weights = raw_weights / total  # Normalizing so sum(w) = 1.0 
        
        for i, label in enumerate(self.weight_labels):
            label.setText(f"{labels[i]} weight: {self.weights[i]:.2f}")
            
        self.update_analysis()
    
    def on_freq_slider_change(self):
        
        self.f_select = self.f_lo + self.freq_slider.value()/100 * (self.f_hi - self.f_lo)
        self.freq_label.setText(f"Frequency Selector for Subarrays: {self.f_select:.2f} Hz")
        self.update_analysis()

    def update_analysis(self):

        # identify minimum value with weights
        weighted_sums = np.dot(self.norm_front, self.weights)
        best_idx = np.argmin(weighted_sums)
        best_point_raw = self.pareto_front_values[best_idx]
        best_state = self.pareto_front_states[best_idx]

        # update histogram of pareto front values
        self.ax_hist.clear()
        self.ax_hist.hist(weighted_sums, bins=15)
        self.ax_hist.set_title("Unified Value Distribution")
        self.ax_hist.set_xlim([0, 1])
        self.histogram_canvas.draw()

        # update visualization of array
        self.ax_array.clear()
        self.ax_bands.clear()

        num_elements = len(best_state) // 2 + 2
        coords = np.zeros((num_elements, 2), dtype=np.float32)
        coords[1, :] = [0, best_state[0]] # place the second element on the y-axis to break symmetry and reduce the search space
        curr_angle = 0
        for i in range(2, num_elements):
            curr_angle += best_state[(num_elements - 1)+ (i - 2)]
            coords[i, 0] = coords[i-1, 0] + best_state[i-1] * np.cos(curr_angle)
            coords[i, 1] = coords[i-1, 1] + best_state[i-1] * np.sin(curr_angle)

        self.ax_array.scatter(coords[:,0], coords[:,1], s = 15, c = 'k', marker = 'o')
        self.ax_array.set_xlim([np.min(coords[:, 0]) - 1, np.max(coords[:, 0]) + 1])
        self.ax_array.set_ylim([np.min(coords[:, 1]) - 1, np.max(coords[:, 1]) + 1])

        Y = coords[:, 0]
        Z = coords[:, 1]
        X = np.zeros_like(Y)

        bf_array = BeamformingArray(X, Y, Z, element_directivity=ElementDirectivity.DIPOLE)
        bf_model = BeamformingModel(bf_array, c = 1460)
        self.f_lo = np.min(bf_model.f_cutoff)
        self.f_hi = np.max(bf_model.f_cutoff)
        bands = bf_model.active_elements
        unique_subarrays, first_occurrence, mapping = np.unique(bands, axis=1, return_index=True, return_inverse=True)
        subarray_mask = np.sum(unique_subarrays, axis = 0) > 2
        valid_inds = np.where(subarray_mask)[0]
        for i in valid_inds: 
                color = self.cmap(i % 10)
                mask = unique_subarrays[:, i].astype(bool)
                subarray_points = coords[mask]

                f_hi = bf_model.f_cutoff[first_occurrence[i]]   
                f_lo = bf_model.f_cutoff[np.where(mapping == i)[0][-1]]
                num_elements = np.sum(mask)
                if f_hi - f_lo > 0:
                    self.ax_bands.barh(y=num_elements, width=(f_hi - f_lo), left=f_lo, height = 0.8, alpha=0.6, color=color)
                else:
                    self.ax_bands.scatter(f_lo, num_elements, s = 20, marker = 'o', color = color)

        self.f_select = self.f_select if self.f_select != 0 else (self.f_hi - self.f_lo)/2 + self.f_lo
        band_id = np.where((self.f_select <= bf_model.f_cutoff[:-1]) & (self.f_select >= bf_model.f_cutoff[1:]))[0][0]
        element_mask = bf_model.active_elements[:, band_id].flatten()
        subarray_points = coords[element_mask]

        if len(subarray_points) >= 3:
            tri = scipy.spatial.Delaunay(subarray_points)
            self.ax_array.triplot(subarray_points[:,0], subarray_points[:,1], tri.simplices, color = color, alpha = 0.5)
            self.ax_array.scatter(subarray_points[:,0], subarray_points[:,1], marker = 'o', color = 'k')

        self.ax_bands.set_xlabel("Frequency (Hz)")
        self.ax_bands.set_ylabel("Elements in Subarray")
        self.ax_bands.set_title("Subarray Frequency Coverage")
        self.ax_bands.grid(True, linestyle='--', alpha=0.5)
        self.band_canvas.draw()

        self.array_canvas.draw()

# Example usage with dummy data
if __name__ == "__main__":
    # initialize model
    opt_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'ArrayOpt_20260506-215528.npz'))
    accepted_states = opt_data['arr_1']
    accepted_energies = opt_data['arr_2']
    min_energy = opt_data['arr_3']
    pareto_states = opt_data['arr_4']
    pareto_values = np.array(opt_data['arr_5'])

    labels = ['total_size', 'count', 'aperture_min', 'f_dist']
    
    app = QApplication(sys.argv)
    gui = ParetoWeightGUI(pareto_front_values=pareto_values, pareto_front_states=pareto_states, labels = labels)
    gui.show()
    sys.exit(app.exec())

    # fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    # ax.scatter(pareto_values[:,1], pareto_values[:,2], pareto_values[:,3], s=np.int32(pareto_values[:,0]*100))

    plt.show()