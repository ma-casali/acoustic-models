"""
Basic PyQt6 GUI Template
A starting point for creating a GUI application from scratch.
"""

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QLineEdit, QGroupBox,
                             QComboBox, QSizePolicy, QProgressBar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
import numpy as np
import sys
from main_solver import run_solver_from_gui


class MainWindow(QWidget):
    """
    Main application window.
    """
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Window properties
        self.setWindowTitle("Parabolic Wave Equation Solver")
        self.setGeometry(100, 100, 1500, 1000)

        # Formatting
        self.normal_font = "font-size: 12 px; font-weight: normal; padding: 10px;"
        self.style_colors = {
            "background-hover": "#87CEFA",
            "background-pressed": "#4682B4",
            "background-normal": "lightblue",
            "text-normal": "black",
            "text-pressed": "white"
        }
        
        # --- Main layout ---
        self.main_layout = QVBoxLayout()

        # =============================
        # Environment Section
        # =============================
        self.environment_layout = QHBoxLayout()

        self.environment_choice_label = QLabel("Choose a type of environment:")
        self.environment_choice_label.setStyleSheet(self.normal_font)
        self.environment_choice_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.environment_layout.addWidget(self.environment_choice_label)
        
        self.environment_choice = QComboBox()
        self.environment_choice.addItems(["--", "Air", "Water"])
        self.environment_layout.addWidget(self.environment_choice)

        self.environment_box = QGroupBox("Environment Settings")
        self.environment_box.setLayout(self.environment_layout)
        self.main_layout.addWidget(self.environment_box)

        # =============================
        # Frequency Section
        # =============================
        self.frequency_layout = QHBoxLayout()

        self.frequency_label = QLabel("Frequency of study [Hz]:")
        self.frequency_label.setStyleSheet(self.normal_font)
        self.frequency_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.frequency_layout.addWidget(self.frequency_label)

        self.frequency_choice = QLineEdit()
        self.frequency_choice.setStyleSheet(self.normal_font)
        self.frequency_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        self.frequency_choice.setPlaceholderText("50")
        self.frequency_layout.addWidget(self.frequency_choice)

        self.frequency_box = QGroupBox("Frequency Settings")
        self.frequency_box.setLayout(self.frequency_layout)
        self.main_layout.addWidget(self.frequency_box)

        # =============================
        # Simulation Limits (Range & Depth)
        # =============================
        self.simulation_limits = QHBoxLayout()

        self.range_label = QLabel("Range for simulation [m]:")
        self.range_label.setStyleSheet(self.normal_font)
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.simulation_limits.addWidget(self.range_label)

        self.range_choice = QLineEdit()
        self.range_choice.setStyleSheet(self.normal_font)
        self.range_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        self.range_choice.setPlaceholderText("2000")
        self.simulation_limits.addWidget(self.range_choice)

        self.depth_label = QLabel("Depth/altitude for simulation [m]:")
        self.depth_label.setStyleSheet(self.normal_font)
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.simulation_limits.addWidget(self.depth_label)

        self.depth_choice = QLineEdit()
        self.depth_choice.setStyleSheet(self.normal_font)
        self.depth_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        self.depth_choice.setPlaceholderText("7000")
        self.simulation_limits.addWidget(self.depth_choice)

        self.limits_box = QGroupBox("Simulation Limits")
        self.limits_box.setLayout(self.simulation_limits)
        self.main_layout.addWidget(self.limits_box)

        # =============================
        # Depth Parameters
        # =============================
        self.depth_parameters = QHBoxLayout()

        # Source depth
        self.source_depth_layout = QHBoxLayout()

        self.source_depth_slider = QSlider(Qt.Orientation.Vertical)
        self.source_depth_slider.setInvertedAppearance(True)
        self.source_depth_slider.setFixedHeight(100)
        self.source_depth_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.source_depth_slider.setMinimum(0)
        self.source_depth_slider.setMaximum(2000)
        self.source_depth_slider.setValue(1500)
        self.source_depth_slider.setTickInterval(100)
        self.source_depth_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.source_depth_slider.valueChanged.connect(self.enforce_slider_max)
        self.source_depth_layout.addWidget(self.source_depth_slider)

        self.source_depth_box = QGroupBox("Source Depth [m]")
        self.source_depth_box.setLayout(self.source_depth_layout)
        self.depth_parameters.addWidget(self.source_depth_box)

        # Topography limits

        self.topo_lim_layout = QHBoxLayout()

        self.topo_min_slider = QSlider(Qt.Orientation.Vertical)
        self.topo_min_slider.setInvertedAppearance(True)
        self.topo_min_slider.setFixedHeight(100)
        self.topo_min_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.topo_min_slider.setMinimum(0)
        self.topo_min_slider.setMaximum(2000)
        self.topo_min_slider.setValue(1500)
        self.topo_min_slider.setTickInterval(100)
        self.topo_min_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.topo_min_slider.valueChanged.connect(self.enforce_slider_max)
        self.topo_lim_layout.addWidget(self.topo_min_slider)

        self.topo_max_slider = QSlider(Qt.Orientation.Vertical)
        self.topo_max_slider.setInvertedAppearance(True)
        self.topo_max_slider.setFixedHeight(100)
        self.topo_max_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.topo_max_slider.setMinimum(0)
        self.topo_max_slider.setMaximum(2000)
        self.topo_max_slider.setValue(1500)
        self.topo_max_slider.setTickInterval(100)
        self.topo_max_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.topo_max_slider.max_limit_source = self.depth_choice
        self.topo_max_slider.min_limit_source = self.topo_min_slider
        self.topo_max_slider.valueChanged.connect(self.enforce_slider_max)
        self.topo_lim_layout.addWidget(self.topo_max_slider)

        self.topo_min_slider.max_limit_source = self.topo_max_slider

        self.topo_limits_box = QGroupBox("Topography limits (min/max) [m]:")
        self.topo_limits_box.setLayout(self.topo_lim_layout)
        self.depth_parameters.addWidget(self.topo_limits_box)

        # z0 depth
        self.z0_layout = QHBoxLayout()

        self.z0_slider = QSlider(Qt.Orientation.Vertical)
        self.z0_slider.setInvertedAppearance(True)
        self.z0_slider.setFixedHeight(100)
        self.z0_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.z0_slider.setMinimum(0)
        self.z0_slider.setMaximum(2000)
        self.z0_slider.setValue(1500)
        self.z0_slider.setTickInterval(100)
        self.z0_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.z0_slider.max_limit_source = self.topo_max_slider
        self.z0_slider.min_limit_source = self.topo_min_slider
        self.z0_slider.valueChanged.connect(self.enforce_slider_max)
        self.z0_slider.valueChanged.connect(self.enforce_slider_min)
        self.z0_layout.addWidget(self.z0_slider)

        self.source_depth_slider.max_limit_source = self.z0_slider

        self.z0_depth_box = QGroupBox("Starting topography depth [m]:")
        self.z0_depth_box.setLayout(self.z0_layout)
        self.depth_parameters.addWidget(self.z0_depth_box)

        # dynamically update maximums of all slider bars
        self.depth_choice.textEdited.connect(lambda text, s = self.source_depth_slider: self.update_slider_max(s))
        self.depth_choice.textEdited.connect(lambda text, s = self.topo_min_slider: self.update_slider_max(s))
        self.depth_choice.textEdited.connect(lambda text, s = self.topo_max_slider: self.update_slider_max(s))
        self.depth_choice.textEdited.connect(lambda text, s = self.z0_slider: self.update_slider_max(s))

        self.main_layout.addLayout(self.depth_parameters)

        # =============================
        # Dictionary of inputs
        # =============================
        self.inputs = {
            'Environment': self.environment_choice,
            'f': self.frequency_choice,
            'r_max': self.range_choice,
            'z_max': self.depth_choice,
            'z_s': self.source_depth_slider,
            'z_0': self.z0_slider,
            'topo_min': self.topo_min_slider,
            'topo_max': self.topo_max_slider
        }

        for key, widget in self.inputs.items():
            if key == "Environment":
                widget.currentTextChanged.connect(lambda text, k=key: self.update_dict(k, text))
            elif key == "z_s" or key == "topo_min" or key == "topo_max" or key == "z_0":
                widget.valueChanged.connect(lambda val, k = key: self.update_dict(k, val))
                widget.sliderReleased.connect(lambda k = key, w = widget: self.update_dict(k, w.value()))
            else:
                widget.textEdited.connect(lambda text, k=key: self.update_dict(k, text))

        self.values = {key: "...waiting..." for key in self.inputs}

        # =============================
        # Display area for input values
        # =============================
        self.display = QLabel("Waiting for input...")
        self.display.setStyleSheet(self.normal_font)
        self.main_layout.addWidget(self.display)

        # =============================
        # Run Simulation Button
        # =============================
        self.simulation_button = QPushButton("Run Simulation")
        self.simulation_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.style_colors['background-normal']};
                color: {self.style_colors['text-normal']};
                border: 2px solid #555;
                border-radius: 8px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {self.style_colors['background-hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.style_colors['background-pressed']};
                color: {self.style_colors['text-pressed']};
            }}
        """)
        self.simulation_button.clicked.connect(self.button_clicked)
        self.main_layout.addWidget(self.simulation_button)

        # Add stretch to push everything up
        self.main_layout.addStretch()

        # =============================
        # Plot Canvas
        # =============================

        self.fig = Figure(figsize = (15,5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title("Parameter Visualization")
        self.ax.set_xlabel("Range [m]")
        self.ax.set_ylabel("Depth [m]")
        self.ax.set_ylim([0, 2000])
        self.ax.set_xlim([0, 2000])
        self.ax.invert_yaxis()

        self.main_layout.addWidget(self.canvas)

        # =============================
        # Final layout setup
        # =============================
        self.setLayout(self.main_layout)

        self.update_plot()
        
    # ---------------------------------------------------------
    # Event methods
    # ---------------------------------------------------------
    def button_clicked(self):
        self.simulation_button.setText("Running Simulation...")
        self.ax.clear()
        print("Running Simulation...")

        # Create a progress bar in the GUI
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v percent finished")
        self.main_layout.addWidget(self.progress_bar)

        self.show_plot()

    def update_dict(self, key, text):
        """Called whenever a QLineEdit or QComboBox value changes."""
        self.values[key] = text
        display_text = (
            f"Environment: {self.values['Environment']}\n"
            f"Frequency: {self.values['f']} [Hz]\n"
            f"Range: {self.values['r_max']} [m]\n"
            f"Depth: {self.values['z_max']} [m]\n"
            f"Source Depth: {self.values['z_s']} [m]\n"
            f"Topography Limits: [{self.values['topo_min']}, {self.values['topo_max']}] [m]\n"
            f"Topography Starting Depth: {self.values['z_0']}\n"
        )
        self.display.setText(display_text)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Parameter Visualization")
        self.ax.set_xlabel("Range [m]")
        self.ax.set_ylabel("Depth [m]")
        self.ax.invert_yaxis()

        # Helper function to safely convert values
        def safe_float(val, default=0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        topo_min = safe_float(self.values.get('topo_min', 0))
        topo_max = safe_float(self.values.get('topo_max', 0))
        z_0 = safe_float(self.values.get('z_0', 0))
        z_s = safe_float(self.values.get('z_s', 0))
        z_max = safe_float(self.values.get('z_max', 2000))
        r_max = safe_float(self.values.get('r_max', 2000))

        # Topography limits
        self.ax.axhline(topo_min, color='k', linestyle='-.')
        self.ax.text(0.5, topo_min, f"min. topography: {topo_min}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='center', fontsize=9, color='k')

        self.ax.axhline(topo_max, color='k', linestyle='-.')
        self.ax.text(0.5, topo_max, f"max. topography: {topo_max}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='center', fontsize=9, color='k')

        # Starting topography
        self.ax.axhline(z_0, color='k', linestyle='--')
        self.ax.text(0.5, z_0, f"starting topography: {z_0}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='right', fontsize=9, color='k')

        # Source depth
        self.ax.axhline(z_s, color='r', linestyle='-')
        self.ax.text(0.5, z_s, f"source depth: {z_s}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='right', fontsize=9, color='r')

        # Set axis limits
        self.ax.set_xlim([0, r_max])
        self.ax.set_ylim([0, z_max])

        self.ax.invert_yaxis()

        # Redraw canvas
        self.canvas.draw_idle()

    def show_plot(self):
        """Clears and re-runs the solver plot."""
        self.ax.clear()
        run_solver_from_gui(self.values, self.canvas.figure, self.ax, progress_callback=self.update_progress)
        self.progress_bar.deleteLater()
        self.progress_bar = None
        self.simulation_button.setText("Run Simulation")
        self.canvas.draw()
        self.canvas.show()

    def update_progress(self, i, N):
        percent = int((i+1) / N * 100)
        self.progress_bar.setValue(percent)
        QApplication.processEvents()

    def update_slider_max(self, slider):

        try:
            depth_val = float(self.depth_choice.text())
            if depth_val > 0:
                slider.setMaximum(int(depth_val))
        except ValueError:
            pass

    def enforce_slider_max(self, value):

        slider = self.sender()
        limit_source = getattr(slider, "max_limit_source", None)

        try:
            if getattr(limit_source, "text", None) == None:
                max_depth = float(limit_source.value())
            else:
                max_depth = float(limit_source.text())
        except ValueError:
            max_depth = 7000
        
        if value > max_depth:
            slider.blockSignals(True)
            slider.setValue(int(max_depth))
            slider.blockSignals(False)

    def enforce_slider_min(self, value):

        slider = self.sender()
        limit_source = getattr(slider, "min_limit_source", None)

        try:
            min_depth = float(limit_source.value())
        except ValueError:
            min_depth = 0

        if value < min_depth:
            slider.blockSignals(True)
            slider.setValue(int(min_depth))
            slider.blockSignals(False)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
