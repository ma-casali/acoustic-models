"""
Basic PyQt6 GUI Template
A starting point for creating a GUI application from scratch.
"""

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QLineEdit, QGroupBox,
                             QComboBox, QSizePolicy, QProgressBar, QStackedWidget,)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtCore import Qt
import numpy as np
import sys
from main_solver import run_solver_from_gui

class ConfigManager(QObject):
    configChanged = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self._data = {
            'Environment': 'Water',
            'f': 50,
            'r_max': 1000,
            'z_max': 500,
            'z_s': 100,
            'z_0': 1500,
            'topo_min': 0,
            'topo_max': 1500
        }

        self._schema = {
            'Environment': str,
            'f': float,
            'r_max': float,
            'z_max': float,
            'z_s': float,
            'z_0': float,
            'topo_min': float,
            'topo_max': float
        }

    def set(self, key, value):
        if key in self._schema:
            try: 
                typed = self._schema[key](value)
            except Exception:
                typed = value
        else:
            typed = value

        old = self._data.get(key, None)
        if old != typed:
            self._data[key] = typed
            self.configChanged.emit(key, typed)

    def get(self, key):
        return self._data[key]
    
    def as_dict(self):
        return dict(self._data)
    

class SharedControls(QWidget):
    pass

class WaterPage(QWidget):
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config

        # Style Parameters
        self.normal_font = "font-size: 12 px; font-weight: normal; padding: 10px;"

        main_layout = QVBoxLayout()
        
        # ============================
        # Simulation Limits (Range & Depth)
        # ============================
        simulation_limits_layout = QHBoxLayout()

        # Range
        self.range_label = QLabel("Range for simulation [m]:")
        self.range_label.setStyleSheet(self.normal_font)
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        simulation_limits_layout.addWidget(self.range_label)

        self.range_choice = QLineEdit()
        self.range_choice.setStyleSheet(self.normal_font)
        self.range_choice.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.range_choice.setPlaceholderText("2000")
        self.range_choice.setFixedWidth(100)
        self.range_choice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.range_choice.editingFinished.connect(lambda: self.config.set('r_max', self.range_choice.text()))
        simulation_limits_layout.addWidget(self.range_choice)

        # Depth
        self.depth_label = QLabel("Depth for simulation [m]:")
        self.depth_label.setStyleSheet(self.normal_font)
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        simulation_limits_layout.addWidget(self.depth_label)

        self.depth_choice = QLineEdit()
        self.depth_choice.setStyleSheet(self.normal_font)
        self.depth_choice.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.depth_choice.setPlaceholderText("7000")
        self.depth_choice.setFixedWidth(100)
        self.depth_choice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.depth_choice.editingFinished.connect(lambda: self.config.set('z_max', self.depth_choice.text()))
        simulation_limits_layout.addWidget(self.depth_choice)

        self.limits_box = QGroupBox("Simulation Limits")
        self.limits_box.setLayout(simulation_limits_layout)
        main_layout.addWidget(self.limits_box)

        # =============================
        # Topography and depth parameters
        # =============================
        depth_parameters = QHBoxLayout()
        
        # Topography limits

        topography_layout = QHBoxLayout()
        self.topo_min_slider = QSlider(Qt.Orientation.Vertical)
        self.topo_min_slider.config_key = 'topo_min'
        self.topo_min_slider.setInvertedAppearance(True)
        self.topo_min_slider.setFixedHeight(100)
        self.topo_min_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.topo_min_slider.setMinimum(0)
        self.topo_min_slider.setMaximum(2000)
        self.topo_min_slider.setValue(1500)
        self.topo_min_slider.setTickInterval(100)
        self.topo_min_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.topo_min_slider.sliderReleased.connect(self.enforce_slider_max)
        self.topo_min_slider.valueChanged.connect(lambda val: self.config.set('topo_min', val))
        topography_layout.addWidget(self.topo_min_slider)

        self.topo_max_slider = QSlider(Qt.Orientation.Vertical)
        self.topo_max_slider.config_key = 'topo_max'
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
        self.topo_max_slider.sliderReleased.connect(self.enforce_slider_max)
        self.topo_max_slider.valueChanged.connect(lambda val: self.config.set('topo_max', val))
        topography_layout.addWidget(self.topo_max_slider)

        self.topo_min_slider.max_limit_source = self.topo_max_slider

        self.topo_limits_box = QGroupBox("Topography limits (min/max) [m]:")
        self.topo_limits_box.setLayout(topography_layout)
        
        depth_parameters.addWidget(self.topo_limits_box)

        # z0 depth
        z0_layout = QHBoxLayout()

        self.z0_slider = QSlider(Qt.Orientation.Vertical)
        self.z0_slider.config_key = 'z_0'
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
        self.z0_slider.sliderReleased.connect(self.enforce_slider_max)
        self.z0_slider.sliderReleased.connect(self.enforce_slider_min)
        self.z0_slider.valueChanged.connect(lambda val: self.config.set('z_0', val))
        z0_layout.addWidget(self.z0_slider)

        self.z0_depth_box = QGroupBox("Starting topography depth [m]:")
        self.z0_depth_box.setLayout(z0_layout)
        depth_parameters.addWidget(self.z0_depth_box)

        # Source depth
        source_depth_layout = QHBoxLayout()

        self.source_depth_slider = QSlider(Qt.Orientation.Vertical)
        self.source_depth_slider.config_key = 'z_s'
        self.source_depth_slider.setInvertedAppearance(True)
        self.source_depth_slider.setFixedHeight(100)
        self.source_depth_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.source_depth_slider.setMinimum(0)
        self.source_depth_slider.setMaximum(2000)
        self.source_depth_slider.setValue(1500)
        self.source_depth_slider.setTickInterval(100)
        self.source_depth_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.source_depth_slider.sliderReleased.connect(self.enforce_slider_max)
        self.source_depth_slider.valueChanged.connect(lambda val: self.config.set('z_s', val))
        source_depth_layout.addWidget(self.source_depth_slider)

        self.source_depth_box = QGroupBox("Source Depth [m]")
        self.source_depth_box.setLayout(source_depth_layout)
        depth_parameters.addWidget(self.source_depth_box)

        self.source_depth_slider.max_limit_source = self.z0_slider

        # dynamically update maximums of all slider bars
        self.depth_choice.editingFinished.connect(lambda: self.update_slider_max(self.source_depth_slider))
        self.depth_choice.editingFinished.connect(lambda: self.update_slider_max(self.topo_min_slider))
        self.depth_choice.editingFinished.connect(lambda: self.update_slider_max(self.topo_max_slider))
        self.depth_choice.editingFinished.connect(lambda: self.update_slider_max(self.z0_slider))

        main_layout.addLayout(depth_parameters)
        self.setLayout(main_layout)

        # also listen to config changes coming from other classes
        self.config.configChanged.connect(self._on_config_changed)

    def _on_config_changed(self, key, value):
        """Keep page widgets in sync when config changes externally."""
        if key == "r_max" and self.range_choice.text() != str(value):
            # avoid resetting if user is typing
            self.range_choice.setText(str(value))
        elif key == "z_max" and self.depth_choice.text() != str(value):
            self.depth_choice.setText(str(value))
        elif key == "topo_min" and self.topo_min_slider.value() != int(value):
            self.topo_min_slider.blockSignals(True)
            self.topo_min_slider.setValue(int(value))
            self.topo_min_slider.blockSignals(False)
        elif key == "topo_max" and self.topo_max_slider.value() != int(value):
            self.topo_max_slider.blockSignals(True)
            self.topo_max_slider.setValue(int(value))
            self.topo_max_slider.blockSignals(False)
        elif key == "z_0" and self.z0_slider.value() != int(value):
            self.z0_slider.blockSignals(True)
            self.z0_slider.setValue(int(value))
            self.z0_slider.blockSignals(False)
        elif key == "z_s" and self.source_depth_slider.value() != int(value):
            self.source_depth_slider.blockSignals(True)
            self.source_depth_slider.setValue(int(value))
            self.source_depth_slider.blockSignals(False)

    def update_slider_max(self, slider):

        try:
            depth_val = float(self.depth_choice.text())
            if depth_val > 0:
                slider.setMaximum(int(depth_val))
        except ValueError:
            pass

    def enforce_slider_max(self):
        slider = self.sender()
        limit_source = getattr(slider, "max_limit_source", None)
        config_key = getattr(slider, "config_key", None)

        max_val = self.read_numeric_value(limit_source, default=7000)

        value = slider.value()  
        if value > max_val:
            slider.blockSignals(True)
            slider.setValue(int(max_val))
            slider.blockSignals(False)

            self.config.set(config_key, int(max_val))

    def enforce_slider_min(self):
        slider = self.sender()
        limit_source = getattr(slider, "min_limit_source", None)
        config_key = getattr(slider, "config_key", None)

        min_val = self.read_numeric_value(limit_source, default=0)

        value = slider.value()
        if value < min_val:
            slider.blockSignals(True)
            slider.setValue(int(min_val))
            slider.blockSignals(False)

            self.config.set(config_key, int(min_val))

    def read_numeric_value(self, widget, default=None):
        try:
            if isinstance(widget, QLineEdit):
                return float(widget.text())

            if isinstance(widget, QSlider):
                return float(widget.value())
        except ValueError:
            pass
        return default
    
class AirPage(QWidget):

    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        # Style Parameters
        self.normal_font = "font-size: 12 px; font-weight: normal; padding: 10px;"

        main_layout = QVBoxLayout()

        # ============================
        # Simulation Limits (Range & Altitude)
        # ============================
        simulation_limits_layout = QHBoxLayout()

        # Range
        self.range_label = QLabel("Range for simulation [m]:")
        self.range_label.setStyleSheet(self.normal_font)
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        simulation_limits_layout.addWidget(self.range_label)

        self.range_choice = QLineEdit()
        self.range_choice.setStyleSheet(self.normal_font)
        self.range_choice.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.range_choice.setPlaceholderText("2000")
        self.range_choice.setFixedWidth(100)
        self.range_choice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.range_choice.editingFinished.connect(lambda: self.config.set("r_max", self.range_choice.text()))
        simulation_limits_layout.addWidget(self.range_choice)

        # Altitude
        self.altitude_label = QLabel("Altitude for simulation [m]:")
        self.altitude_label.setStyleSheet(self.normal_font)
        self.altitude_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        simulation_limits_layout.addWidget(self.altitude_label)

        self.altitude_choice = QLineEdit()
        self.altitude_choice.setStyleSheet(self.normal_font)
        self.altitude_choice.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.altitude_choice.setPlaceholderText("7000")
        self.altitude_choice.setFixedWidth(100)
        self.altitude_choice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.altitude_choice.editingFinished.connect(lambda: self.config.set("z_max", self.altitude_choice.text()))
        simulation_limits_layout.addWidget(self.altitude_choice)

        self.limits_box = QGroupBox("Simulation Limits")
        self.limits_box.setLayout(simulation_limits_layout)
        main_layout.addWidget(self.limits_box)

        # =============================
        # Topography + altitude parameters
        # =============================
        altitude_parameters = QHBoxLayout()

        # --------------------------------
        # Topography limits (identical structure to WaterPage)
        # --------------------------------
        topography_layout = QHBoxLayout()

        self.topo_min_slider = QSlider(Qt.Orientation.Vertical)
        self.topo_min_slider.config_key = 'topo_min'
        self.topo_min_slider.setInvertedAppearance(True)
        self.topo_min_slider.setFixedHeight(100)
        self.topo_min_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.topo_min_slider.setMinimum(0)
        self.topo_min_slider.setMaximum(2000)
        self.topo_min_slider.setValue(1500)
        self.topo_min_slider.setTickInterval(100)
        self.topo_min_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.topo_min_slider.sliderReleased.connect(self.enforce_slider_max)
        self.topo_min_slider.valueChanged.connect(lambda val: self.config.set("topo_min", val))
        topography_layout.addWidget(self.topo_min_slider)

        self.topo_max_slider = QSlider(Qt.Orientation.Vertical)
        self.topo_max_slider.config_key = 'topo_max'
        self.topo_max_slider.setInvertedAppearance(True)
        self.topo_max_slider.setFixedHeight(100)
        self.topo_max_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.topo_max_slider.setMinimum(0)
        self.topo_max_slider.setMaximum(2000)
        self.topo_max_slider.setValue(1500)
        self.topo_max_slider.setTickInterval(100)
        self.topo_max_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.topo_max_slider.max_limit_source = self.altitude_choice
        self.topo_max_slider.min_limit_source = self.topo_min_slider
        self.topo_max_slider.sliderReleased.connect(self.enforce_slider_max)
        self.topo_max_slider.valueChanged.connect(lambda val: self.config.set("topo_max", val))
        topography_layout.addWidget(self.topo_max_slider)

        self.topo_min_slider.max_limit_source = self.topo_max_slider

        self.topo_limits_box = QGroupBox("Topography limits (min/max) [m]:")
        self.topo_limits_box.setLayout(topography_layout)
        altitude_parameters.addWidget(self.topo_limits_box)

        # --------------------------------
        # z0 altitude slider
        # --------------------------------
        z0_layout = QHBoxLayout()

        self.z0_slider = QSlider(Qt.Orientation.Vertical)
        self.z0_slider.config_key = 'z_0'
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
        self.z0_slider.sliderReleased.connect(self.enforce_slider_max)
        self.z0_slider.sliderReleased.connect(self.enforce_slider_min)
        self.z0_slider.valueChanged.connect(lambda: self.config.set("z_0", self.z0_slider.value()))
        z0_layout.addWidget(self.z0_slider)

        self.z0_altitude_box = QGroupBox("Starting topography altitude [m]:")
        self.z0_altitude_box.setLayout(z0_layout)
        altitude_parameters.addWidget(self.z0_altitude_box)

        # --------------------------------
        # Source altitude slider
        # --------------------------------
        source_altitude_layout = QHBoxLayout()

        self.source_altitude_slider = QSlider(Qt.Orientation.Vertical)
        self.source_altitude_slider.config_key = 'z_s'
        self.source_altitude_slider.setInvertedAppearance(True)
        self.source_altitude_slider.setFixedHeight(100)
        self.source_altitude_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.source_altitude_slider.setMinimum(0)
        self.source_altitude_slider.setMaximum(2000)
        self.source_altitude_slider.setValue(1500)
        self.source_altitude_slider.setTickInterval(100)
        self.source_altitude_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.source_altitude_slider.max_limit_source = self.z0_slider
        self.source_altitude_slider.sliderReleased.connect(self.enforce_slider_max)
        self.source_altitude_slider.valueChanged.connect(lambda val: self.config.set("z_s", val))
        source_altitude_layout.addWidget(self.source_altitude_slider)

        self.source_altitude_box = QGroupBox("Source Altitude [m]")
        self.source_altitude_box.setLayout(source_altitude_layout)
        altitude_parameters.addWidget(self.source_altitude_box)

        # --------------------------------
        # Update all slider maxima when altitude changes
        # --------------------------------
        self.altitude_choice.editingFinished.connect(lambda: self.update_slider_max(self.source_altitude_slider))
        self.altitude_choice.editingFinished.connect(lambda: self.update_slider_max(self.topo_min_slider))
        self.altitude_choice.editingFinished.connect(lambda: self.update_slider_max(self.topo_max_slider))
        self.altitude_choice.editingFinished.connect(lambda: self.update_slider_max(self.z0_slider))

        main_layout.addLayout(altitude_parameters)
        self.setLayout(main_layout)

        # listen to config changes
        self.config.configChanged.connect(self._on_config_changed)

    def _on_config_changed(self, key, value):
        if key == "r_max" and self.range_choice.text() != str(value):
            self.range_choice.setText(str(value))

        elif key == "z_max" and self.altitude_choice.text() != str(value):
            self.altitude_choice.setText(str(value))

        elif key == "topo_min" and self.topo_min_slider.value() != int(value):
            self.topo_min_slider.blockSignals(True)
            self.topo_min_slider.setValue(int(value))
            self.topo_min_slider.blockSignals(False)

        elif key == "topo_max" and self.topo_max_slider.value() != int(value):
            self.topo_max_slider.blockSignals(True)
            self.topo_max_slider.setValue(int(value))
            self.topo_max_slider.blockSignals(False)

        elif key == "z_0" and self.z0_slider.value() != int(value):
            self.z0_slider.blockSignals(True)
            self.z0_slider.setValue(int(value))
            self.z0_slider.blockSignals(False)

        elif key == "z_s" and self.source_altitude_slider.value() != int(value):
            self.source_altitude_slider.blockSignals(True)
            self.source_altitude_slider.setValue(int(value))
            self.source_altitude_slider.blockSignals(False)

    def update_slider_max(self, slider):
        try:
            altitude_val = float(self.altitude_choice.text())
            if altitude_val > 0:
                slider.setMaximum(int(altitude_val))
        except ValueError:
            pass

    def enforce_slider_max(self):
        slider = self.sender()
        limit_source = getattr(slider, "max_limit_source", None)
        config_key = getattr(slider, "config_key", None)

        max_val = self.read_numeric_value(limit_source, default=7000)
        
        value = slider.value()
        if value > max_val:
            slider.blockSignals(True)
            slider.setValue(int(max_val))
            slider.blockSignals(False)

            self.config.set(config_key, int(max_val))

    def enforce_slider_min(self):
        slider = self.sender()
        limit_source = getattr(slider, "min_limit_source", None)
        config_key = getattr(slider, "config_key", None)
        
        if limit_source is None:
            return
        
        min_val = self.read_numeric_value(limit_source, default=0)
        
        value = slider.value()
        if value < min_val:
            slider.blockSignals(True)
            slider.setValue(int(min_val))
            slider.blockSignals(False)

            self.config.set(config_key, int(min_val))

    def read_numeric_value(self, widget, default=None):
        try:
            if isinstance(widget, QLineEdit):
                return float(widget.text())

            if isinstance(widget, QSlider):
                return float(widget.value())
        except ValueError:
            pass
        return default
    
class MainWindow(QWidget):
    """
    Main application window.
    """
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.init_ui()
    
    def init_ui(self):
        """
        Initialize the user interface.
        """

        self.main_layout = QVBoxLayout()

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
        
        # --- Define sections of layouts ---
        
        # non-dependent layouts
        self.environment_layout = QHBoxLayout()
        self.frequency_layout = QHBoxLayout()

        # =============================
        # Environment Section
        # =============================

        self.environment_choice_label = QLabel("Choose a type of environment:")
        self.environment_choice_label.setStyleSheet(self.normal_font)
        self.environment_choice_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.environment_layout.addWidget(self.environment_choice_label)
        
        self.environment_choice = QComboBox()
        self.environment_choice.addItems(["--", "Air", "Water"])
        self.environment_layout.addWidget(self.environment_choice)
        self.environment_choice.currentIndexChanged.connect(self.on_environment_changed)

        self.environment_box = QGroupBox("Environment Settings")
        self.environment_box.setLayout(self.environment_layout)
        
        self.main_layout.addWidget(self.environment_box)

        # =============================
        # Frequency Section
        # =============================

        self.frequency_label = QLabel("Frequency of study [Hz]:")
        self.frequency_label.setStyleSheet(self.normal_font)
        self.frequency_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.frequency_layout.addWidget(self.frequency_label)

        self.frequency_choice = QLineEdit()
        self.frequency_choice.setStyleSheet(self.normal_font)
        self.frequency_choice.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.frequency_choice.setPlaceholderText("50")
        self.frequency_choice.setFixedWidth(300)
        self.frequency_choice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.frequency_layout.addWidget(self.frequency_choice)

        self.frequency_box = QGroupBox("Frequency Settings")
        self.frequency_box.setLayout(self.frequency_layout)

        self.main_layout.addWidget(self.frequency_box)

        # =============================
        # Stacked Layouts for Environment-Specific Controls
        # =============================

        self.environment_stack = QStackedWidget()
        self.water_page = WaterPage(self.config)
        self.air_page = AirPage(self.config)

        self.environment_stack.addWidget(self.air_page)
        self.environment_stack.addWidget(self.water_page)
        self.main_layout.addWidget(self.environment_stack)

        self.environment_choice.currentIndexChanged.connect(self.on_environment_changed)

        self.config.configChanged.connect(self._on_config_changed)

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
        self.simulation_button.clicked.connect(self.simulation_button_clicked)
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

        self.update_parameter_plot()
        
    # ---------------------------------------------------------
    # Event methods
    # ---------------------------------------------------------
    def on_environment_changed(self, index):
        """Handle changes to the environment selection."""
        if index == 0:
            # Reset to default values or hide controls
            self.environment_stack.setCurrentIndex(0)
        elif index == 1:
            # Air selected
            self.environment_stack.setCurrentIndex(0)
        elif index == 2:
            # Water selected
            self.environment_stack.setCurrentIndex(1)

        # Update the parameter plot to reflect any changes
        self.update_parameter_plot() 

    def _on_config_changed(self, key, value):
        self.update_parameter_plot()
        
    def update_parameter_plot(self):

        cfg = self.config.as_dict()

        # textual display of current parameters
        display_text = (
            f"Environment: {cfg['Environment']}\n"
            f"Frequency: {cfg['f']} [Hz]\n"
            f"Range: {cfg['r_max']} [m]\n"
            f"Depth: {cfg['z_max']} [m]\n"
            f"Source Depth: {cfg['z_s']} [m]\n"
            f"Topography Limits: [{cfg['topo_min']}, {cfg['topo_max']}] [m]\n"
            f"Topography Starting Depth: {cfg['z_0']}\n"
        )
        self.display.setText(display_text)

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

        topo_min = safe_float(cfg.get('topo_min', 0))
        topo_max = safe_float(cfg.get('topo_max', 0))
        z_0 = safe_float(cfg.get('z_0', 0))
        z_s = safe_float(cfg.get('z_s', 0))
        z_max = safe_float(cfg.get('z_max', 2000))
        r_max = safe_float(cfg.get('r_max', 2000))

        # Topography limits
        self.ax.axhline(topo_min, color='k', linestyle='-.')
        self.ax.text(0.5, topo_min, f"min. topography: {topo_min}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='center', fontsize=9, color='k',
                    backgroundcolor='w')

        self.ax.axhline(topo_max, color='k', linestyle='-.')
        self.ax.text(0.5, topo_max, f"max. topography: {topo_max}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='center', fontsize=9, color='k',
                    backgroundcolor='w')

        # Starting topography
        self.ax.axhline(z_0, color='k', linestyle='--')
        self.ax.text(0.5, z_0, f"starting topography: {z_0}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='right', fontsize=9, color='k',
                    backgroundcolor='w')

        # Source depth
        self.ax.axhline(z_s, color='r', linestyle='-')
        self.ax.text(0.5, z_s, f"source depth: {z_s}",
                    transform=self.ax.get_yaxis_transform(),
                    va='center', ha='right', fontsize=9, color='r',
                    backgroundcolor='w')

        # Set axis limits
        self.ax.set_xlim([0, r_max])
        self.ax.set_ylim([0, z_max])

        self.ax.invert_yaxis()

        # Redraw canvas
        self.canvas.draw_idle()

    def simulation_button_clicked(self):
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

        # Disable the run button to prevent re-entrant runs while the solver is executing
        self.show_simulation_plot()
        self.simulation_button.setText("Simulation Complete! Click to Run Another.")
        self.simulation_button.clicked.disconnect(self.simulation_button_clicked)
        self.simulation_button.clicked.connect(self.redo_simulation_button_clicked)

    def redo_simulation_button_clicked(self):
        self.fig.clear()
        self.simulation_button.setText("Run Simulation")
        self.simulation_button.clicked.disconnect(self.redo_simulation_button_clicked)
        self.simulation_button.clicked.connect(self.simulation_button_clicked)

        self.canvas.deleteLater()
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

        self.update_parameter_plot()

    def show_simulation_plot(self):
        """Clears and re-runs the solver plot."""
        self.ax.clear()
        cfg = self.config.as_dict()

        try:
            run_solver_from_gui(cfg, self.canvas.figure, self.ax, progress_callback=self.update_progress)
        except Exception:
            # Ensure the progress bar is cleaned up and the GUI is left in a sane state
            if getattr(self, 'progress_bar', None) is not None:
                self.progress_bar.deleteLater()
                self.progress_bar = None
            self.simulation_button.setText("Run Simulation")
            self.canvas.draw()
            raise

        # Clean up progress bar and restore UI state
        if getattr(self, 'progress_bar', None) is not None:
            self.progress_bar.deleteLater()
            self.progress_bar = None

        self.canvas.draw()
        self.canvas.show()

    def update_progress(self, i, N):
        percent = int((i+1) / N * 100)
        self.progress_bar.setValue(percent)
        QApplication.processEvents()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
