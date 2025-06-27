# agi_prototype/ui/gui.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils.visualization import NetworkVisualizer
from config import REFRESH_RATE_MS, DT, NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS, EXC_INH_RATIO

class AGIPrototypeGUI(QWidget):
    def __init__(self, simulation_manager):
        super().__init__()
        self.simulation_manager = simulation_manager
        self.visualizer = None # Will be created in init_ui
        self.init_ui()
        self.reset_visualizer() # Create the first instance

    def init_ui(self):
        """Initializes the window and all its widgets."""
        self.setWindowTitle("Bio-Hyper-SNN AGI Prototype")
        self.setGeometry(100, 100, 1800, 1000)

        self.main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        controls_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("Reset Simulation")
        self.reset_button.clicked.connect(self.reset_simulation)
        controls_layout.addWidget(self.reset_button)
        
        self.main_layout.addLayout(controls_layout)
        self.setLayout(self.main_layout)

        self.timer = QTimer()
        self.timer.setInterval(REFRESH_RATE_MS)
        self.timer.timeout.connect(self.update_gui)

    def reset_visualizer(self):
        """Creates or recreates the visualizer and canvas."""
        if hasattr(self, 'canvas_widget') and self.canvas_widget:
            self.main_layout.removeWidget(self.canvas_widget)
            self.canvas_widget.deleteLater()
            self.canvas_widget = None

        num_hidden_exc = int(NUM_HIDDEN_NEURONS * EXC_INH_RATIO)
        num_hidden_inh = NUM_HIDDEN_NEURONS - num_hidden_exc
        
        self.visualizer = NetworkVisualizer(
            num_input=NUM_INPUT_NEURONS,
            num_hidden_exc=num_hidden_exc,
            num_hidden_inh=num_hidden_inh,
            num_output=NUM_OUTPUT_NEURONS,
            dt=DT
        )
        self.simulation_manager.set_visualizer(self.visualizer)
        
        self.canvas = FigureCanvas(self.visualizer.fig)
        self.canvas_widget = self.canvas # Keep a reference to the widget
        self.main_layout.addWidget(self.canvas_widget)

    def update_gui(self):
        """Fetches data from the simulation manager and updates the plots."""
        if not self.simulation_manager.has_data: return

        # This method in network.py must exist and return a dictionary of numpy arrays
        weights = self.simulation_manager.network.get_all_synaptic_weights()

        all_data = {
            'synaptic_weights': weights,
            'membrane_potential_history': self.simulation_manager.membrane_potential_history_example_neuron,
            'dopamine_history': self.simulation_manager.dopamine_history,
            'acetylcholine_history': self.simulation_manager.acetylcholine_history,
            'serotonin_history': self.simulation_manager.serotonin_history,
        }
        
        self.visualizer.update_plots(self.simulation_manager.current_time_ms, all_data)
        self.canvas.draw()

    def start_simulation(self):
        self.simulation_manager.start_simulation()
        self.timer.start()

    def stop_simulation(self):
        self.simulation_manager.stop_simulation()
        self.timer.stop()

    def reset_simulation(self):
        self.stop_simulation()
        self.simulation_manager.reset_simulation()
        # The manager's reset now handles creating a new visualizer via the GUI
        self.update_gui()
        self.canvas.draw()
