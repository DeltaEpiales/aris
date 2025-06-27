# agi_prototype/main.py

import torch
import numpy as np
import time
import sys
import threading
from collections import deque
from config import *
from src.network import AGINetwork
from src.data_encoders import DataEncoder
from src.hdc import HDC
from utils.visualization import NetworkVisualizer
from utils.metrics import calculate_sparsity
from ui.gui import AGIPrototypeGUI
from PyQt5.QtWidgets import QApplication

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(RANDOM_SEED)

class SimulationManager:
    def __init__(self, gui_instance=None):
        print(f"Initializing Simulation Manager on device: {DEVICE}")
        self.gui = gui_instance
        self.hdc_system = HDC()
        self.data_encoder = DataEncoder(num_neurons_for_pattern_encoding=NUM_INPUT_NEURONS, hdc_system=self.hdc_system)
        self.network = AGINetwork(hdc_system=self.hdc_system, data_encoder=self.data_encoder).to(DEVICE)
        self.visualizer = None
        self.current_time_ms = 0.0
        self.sim_step = 0
        self.is_running = False
        self.stop_event = threading.Event()
        self.sim_thread = None
        self.has_data = False
        history_len = int(RASTER_WINDOW_SIZE_MS / DT)
        self.dopamine_history = deque(maxlen=history_len)
        self.acetylcholine_history = deque(maxlen=history_len)
        self.serotonin_history = deque(maxlen=history_len)
        self.membrane_potential_history_example_neuron = [deque(maxlen=history_len) for _ in range(NUM_EXAMPLE_NEURONS_TO_PLOT)]
        self.epistemic_uncertainty = 0.0
        
        # Task related state
        self.current_pattern_id = 'A'
        self.pattern_timer = 0.0
        self.last_sparsity = 1.0 # Initialize to max sparsity

    def set_visualizer(self, visualizer):
        self.visualizer = visualizer

    def _run_developmental_phase(self):
        print("Starting developmental phase...")
        start_time = time.time()
        developmental_steps = int(DEVELOPMENT_PHASE_DURATION_MS / DT)
        num_hidden_exc = int(NUM_HIDDEN_NEURONS * EXC_INH_RATIO)
        avg_input_spikes = torch.zeros(NUM_INPUT_NEURONS, device=DEVICE)
        avg_hidden_exc_spikes = torch.zeros(num_hidden_exc, device=DEVICE)
        for step in range(developmental_steps):
            noisy_input_current = torch.rand(NUM_INPUT_NEURONS, device=DEVICE) * INPUT_CURRENT_SCALE
            output = self.network(noisy_input_current, self.current_time_ms, stdp_enabled=False)
            if output:
                avg_input_spikes += output['input_spikes']
                avg_hidden_exc_spikes += output['hidden_exc_spikes']
            self.current_time_ms += DT
        avg_input_spikes /= developmental_steps
        avg_hidden_exc_spikes /= developmental_steps
        self.network.apply_developmental_bias(avg_input_spikes, avg_hidden_exc_spikes)
        print(f"Developmental phase completed in {time.time() - start_time:.2f} seconds.")
        self.current_time_ms = 0.0
        self.sim_step = 0

    def _generate_task_input(self):
        """Generates input based on the current pattern for the task."""
        self.pattern_timer += DT
        if self.pattern_timer > PATTERN_PRESENTATION_MS:
            self.pattern_timer = 0.0
            self.current_pattern_id = 'B' if self.current_pattern_id == 'A' else 'A'
            print(f"Switching to Pattern {self.current_pattern_id} at {self.current_time_ms:.0f}ms")

        input_signal = torch.zeros(NUM_INPUT_NEURONS, device=DEVICE)
        if self.current_pattern_id == 'A':
            input_signal[PATTERN_A_INDICES] = 1.0
        else: # Pattern B
            input_signal[PATTERN_B_INDICES] = 1.0
            
        # Add a small amount of noise
        noise = torch.rand(NUM_INPUT_NEURONS, device=DEVICE) * 0.1
        return (input_signal + noise) * INPUT_CURRENT_SCALE

    def simulation_loop(self):
        try:
            self._run_developmental_phase()
            while not self.stop_event.is_set() and self.current_time_ms < SIMULATION_DURATION:
                loop_start_time = time.time()
                input_current = self._generate_task_input()
                
                # Run network
                network_output = self.network(input_current, self.current_time_ms)
                
                # --- Task Logic and Neuromodulation ---
                reward = 0.0
                serotonin_pulse = 0.0
                if network_output:
                    output_spikes = network_output['output_spikes']
                    hidden_spikes = network_output['hidden_exc_spikes']
                    
                    # Check for correct response for reward
                    if self.current_pattern_id == 'A' and output_spikes[TARGET_NEURON_A] > 0:
                        reward = DOPAMINE_MOD['reward_boost']
                        serotonin_pulse = 0.1 # Also pulse serotonin on reward
                        print(f"Correct! Rewarding for pattern A at {self.current_time_ms:.0f}ms")
                    elif self.current_pattern_id == 'B' and output_spikes[TARGET_NEURON_B] > 0:
                        reward = DOPAMINE_MOD['reward_boost']
                        serotonin_pulse = 0.1 # Also pulse serotonin on reward
                        print(f"Correct! Rewarding for pattern B at {self.current_time_ms:.0f}ms")
                    
                    # Uncertainty is now driven by CHANGE in sparsity
                    current_sparsity = calculate_sparsity(hidden_spikes)
                    sparsity_change = abs(current_sparsity - self.last_sparsity)
                    self.epistemic_uncertainty = sparsity_change * 5 # Amplify the change signal
                    self.last_sparsity = current_sparsity
                    
                # Update neuromodulators
                self.network.neuromodulators.set_dopamine_pulse(reward)
                self.network.neuromodulators.set_acetylcholine_pulse(self.epistemic_uncertainty * ACETYLCHOLINE_MOD['uncertainty_gain'])
                self.network.neuromodulators.set_serotonin_pulse(serotonin_pulse)


                if network_output:
                    self.has_data = True
                    self._update_history_data(network_output)

                self.current_time_ms += DT
                self.sim_step += 1
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, (DT / 1000.0) - elapsed)
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Error in simulation loop: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            print("Simulation loop terminated.")

    def _update_history_data(self, network_output):
        if self.visualizer is None: return
        self.dopamine_history.append(network_output.get('dopamine_level', 0.0))
        self.acetylcholine_history.append(network_output.get('acetylcholine_level', 0.0))
        self.serotonin_history.append(network_output.get('serotonin_level', 0.0))
        hidden_exc_potentials = self.network.exc_hidden_neurons.v.detach().cpu()
        num_to_plot = min(NUM_EXAMPLE_NEURONS_TO_PLOT, len(hidden_exc_potentials))
        for i in range(num_to_plot):
            self.membrane_potential_history_example_neuron[i].append(hidden_exc_potentials[i].item())
        self.visualizer.spike_history_input.append(network_output.get('input_spikes').cpu())
        self.visualizer.spike_history_hidden_exc.append(network_output.get('hidden_exc_spikes').cpu())
        self.visualizer.spike_history_hidden_inh.append(network_output.get('hidden_inh_spikes').cpu())
        self.visualizer.spike_history_output.append(network_output.get('output_spikes').cpu())

    def start_simulation(self):
        if not self.is_running:
            print("Starting simulation...")
            self.is_running = True
            self.stop_event.clear()
            self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.sim_thread.start()

    def stop_simulation(self):
        if self.is_running:
            print("Stopping simulation...")
            self.stop_event.set()
            if self.sim_thread and self.sim_thread.is_alive():
                self.sim_thread.join(timeout=2.0)
            self.is_running = False
            print("Simulation stopped.")

    def reset_simulation(self):
        self.stop_simulation()
        print("Resetting simulation...")
        self.__init__(gui_instance=self.gui)
        if self.gui:
            self.gui.reset_visualizer()
        print("Simulation reset complete.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = SimulationManager()
    gui = AGIPrototypeGUI(manager)
    manager.gui = gui
    gui.show()
    sys.exit(app.exec_())
