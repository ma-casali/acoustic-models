"""
Basic PyQt6 GUI Template
A starting point for creating a GUI application from scratch.
"""

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QLineEdit, QGroupBox,
                             QComboBox)
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
        # Set window properties
        self.setWindowTitle("Parabolic Wave Equation Solver")
        self.setGeometry(100, 100, 1500, 1000)  # x, y, width, height

        # formatting variables
        self.normal_font = "font-size: 12 px; font-weight: normal; padding: 10px;"
        self.style_colors = {
            "background-hover": "#87CEFA",
            "background-pressed": "#4682B4",
            "background-normal": "lightblue",
            "text-normal": "black",
            "text-pressed": "white"
        }
        
        # Create main layout
        self.main_layout = QVBoxLayout()
        
        # Choosing environment
        environment_layout = QHBoxLayout()

        environment_choice = QLabel("Choose a type of environment:")
        environment_choice.setStyleSheet(self.normal_font)
        environment_choice.setAlignment(Qt.AlignmentFlag.AlignLeft)
        environment_layout.addWidget(environment_choice)
        
        # Creating the environment choice options
        environment_choice_options = QComboBox()
        environment_choice_options.addItems(["--", "Air", "Water"])
        environment_layout.addWidget(environment_choice_options)

        self.main_layout.addLayout(environment_layout)

        # Determining frequency of study
        frequency_layout = QHBoxLayout()

        frequency_prompt = QLabel("Frequency of study [Hz]:")
        frequency_prompt.setStyleSheet(self.normal_font)
        frequency_prompt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        frequency_layout.addWidget(frequency_prompt)

        frequency_choice = QLineEdit()
        frequency_choice.setStyleSheet(self.normal_font)
        frequency_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        frequency_choice.setPlaceholderText("50")
        frequency_layout.addWidget(frequency_choice)

        self.main_layout.addLayout(frequency_layout)

        # Determing range and depth 
        simulation_limits = QHBoxLayout()

        range_prompt = QLabel("Range for simulation [m]:")
        range_prompt.setStyleSheet(self.normal_font)
        range_prompt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        simulation_limits.addWidget(range_prompt)

        range_choice = QLineEdit()
        range_choice.setStyleSheet(self.normal_font)
        range_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        range_choice.setPlaceholderText("2000")
        simulation_limits.addWidget(range_choice)

        depth_prompt = QLabel("Depth/altitude for simulation [m]:")
        depth_prompt.setStyleSheet(self.normal_font)
        depth_prompt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        simulation_limits.addWidget(depth_prompt)

        depth_choice = QLineEdit()
        depth_choice.setStyleSheet(self.normal_font)
        depth_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        depth_choice.setPlaceholderText("2000")
        simulation_limits.addWidget(depth_choice)

        self.main_layout.addLayout(simulation_limits)

        # Depth parameters
        depth_parameters = QHBoxLayout()

        # source depth
        source_depth_prompt = QLabel("Source depth [m]:")
        source_depth_prompt.setStyleSheet(self.normal_font)
        source_depth_prompt.setAlignment(Qt.AlignmentFlag.AlignRight)
        depth_parameters.addWidget(source_depth_prompt)

        source_depth_choice = QLineEdit()
        source_depth_choice.setStyleSheet(self.normal_font)
        source_depth_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        source_depth_choice.setPlaceholderText("1500")
        depth_parameters.addWidget(source_depth_choice)

        # z0 depth 
        z0_depth_prompt = QLabel("Starting topography depth [m]:")
        z0_depth_prompt.setStyleSheet(self.normal_font)
        z0_depth_prompt.setAlignment(Qt.AlignmentFlag.AlignRight)
        depth_parameters.addWidget(z0_depth_prompt)

        z0_depth_choice = QLineEdit()
        z0_depth_choice.setStyleSheet(self.normal_font)
        z0_depth_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        z0_depth_choice.setPlaceholderText("1900")
        depth_parameters.addWidget(z0_depth_choice)

        # topography_limits
        topo_limits_prompt = QLabel("Topography limits (min/max) [m]:")
        topo_limits_prompt.setStyleSheet(self.normal_font)
        topo_limits_prompt.setAlignment(Qt.AlignmentFlag.AlignRight)
        depth_parameters.addWidget(topo_limits_prompt)

        topo_min_choice = QLineEdit()
        topo_min_choice.setStyleSheet(self.normal_font)
        topo_min_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        topo_min_choice.setPlaceholderText("1000 (min. value)")
        depth_parameters.addWidget(topo_min_choice)

        topo_max_choice = QLineEdit()
        topo_max_choice.setStyleSheet(self.normal_font)
        topo_max_choice.setAlignment(Qt.AlignmentFlag.AlignTrailing)
        topo_max_choice.setPlaceholderText("2000 (max. value)")
        depth_parameters.addWidget(topo_max_choice)

        self.main_layout.addLayout(depth_parameters)

        self.inputs = {
            'Environment': environment_choice_options,
            'f': frequency_choice,
            'r_max': range_choice,
            'z_max': depth_choice,
            'z_s': source_depth_choice,
            'z_0': z0_depth_choice,
            'topo_min': topo_min_choice,
            'topo_max': topo_max_choice
        }

        for key, widget in self.inputs.items():
            if key == "Environment":
                widget.currentTextChanged.connect(lambda text, k = key: self.update_dict(k, text))
            else:
                widget.textEdited.connect(lambda text, k = key: self.update_dict(k, text))

        self.values = {key: "...waiting..." for key in self.inputs}

        # Display area for input values
        self.display = QLabel("Waiting for input...")
        self.display.setStyleSheet(self.normal_font)
        self.main_layout.addWidget(self.display)

        # Run Simulation Button
        self.simulation_button = QPushButton()
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
        self.simulation_button.setText("Run Simulation!")
        self.simulation_button.clicked.connect(self.button_clicked)
        self.main_layout.addWidget(self.simulation_button)

         # Add stretch to push everything to top
        self.main_layout.addStretch()

        # Plot canvas
        self.canvas = FigureCanvas(Figure(figsize = (15,5)))
        self.main_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.hide()
        
        # Set the main layout
        self.setLayout(self.main_layout)
        
    def button_clicked(self):
        self.simulation_button.setText("Running Simulation...")
        print("Running Simulation...")
        self.show_plot()

    def update_dict(self, key, text):
        """ Called whenever a QLineEdit text changes."""
        self.values[key] = text
        display_text = {
            f"Environment: {self.values['Environment']} \n"
            f"Frequency: {self.values['f']} [Hz] \n"
            f"Range: {self.values['r_max']} [m] \n"
            f"Depth: {self.values['z_max']} [m] \n"
            f"Source Depth: {self.values['z_s']} [m] \n"
            f"Topography Limits: [{self.values['topo_min']}, {self.values['topo_max']}] [m] \n"
            f"Topography Starting Depth: {self.values['z_0']} \n"
        }
        self.display.setText("".join(display_text))

    def show_plot(self):

        # Clear the previous plot
        self.ax.clear()
        run_solver_from_gui(self.values, self.canvas.figure, self.ax)

        # Redraw canvas
        self.canvas.draw()
        self.canvas.show()


def main():
    """
    Main function to run the application.
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
