# ---
# agi_prototype/utils/visualization.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from collections import deque
from config import *

class NetworkVisualizer:
    def __init__(self, num_input, num_hidden_exc, num_hidden_inh, num_output, dt=DT):
        self.num_input = num_input
        self.num_hidden_exc = num_hidden_exc
        self.num_hidden_inh = num_hidden_inh
        self.num_output = num_output
        self.num_example_neurons_to_plot = min(NUM_EXAMPLE_NEURONS_TO_PLOT, num_hidden_exc if num_hidden_exc > 0 else 1)
        self.dt = dt
        self.raster_window_steps = int(RASTER_WINDOW_SIZE_MS / self.dt)

        self._create_data_histories()
        
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = Figure(figsize=(16, 10), tight_layout=True)
        gs = self.fig.add_gridspec(3, 2)

        self.ax_raster_input = self.fig.add_subplot(gs[0, 0])
        self.ax_raster_hidden = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raster_input)
        self.ax_raster_output = self.fig.add_subplot(gs[2, 0], sharex=self.ax_raster_input)
        self.ax_weights = self.fig.add_subplot(gs[0, 1])
        self.ax_membrane = self.fig.add_subplot(gs[1, 1], sharex=self.ax_raster_input)
        self.ax_neuromod = self.fig.add_subplot(gs[2, 1], sharex=self.ax_raster_input)

        self._init_plots()

    def _create_data_histories(self):
        self.spike_history_input = deque(maxlen=self.raster_window_steps)
        self.spike_history_hidden_exc = deque(maxlen=self.raster_window_steps)
        self.spike_history_hidden_inh = deque(maxlen=self.raster_window_steps)
        self.spike_history_output = deque(maxlen=self.raster_window_steps)

    def _init_plots(self):
        self.ax_raster_input.set_title("Input Layer Spikes")
        self.ax_raster_input.set_ylabel("Neuron Index")
        self.ax_raster_input.set_ylim(-0.5, self.num_input - 0.5)
        
        self.ax_raster_hidden.set_title("Hidden Layer Spikes (Exc/Inh)")
        self.ax_raster_hidden.set_ylabel("Neuron Index")
        self.ax_raster_hidden.set_ylim(-0.5, self.num_hidden_exc + self.num_hidden_inh - 0.5)
        self.ax_raster_hidden.axhline(y=self.num_hidden_exc - 0.5, color='white', linestyle='--', linewidth=1.5, zorder=10)

        self.ax_raster_output.set_title("Output Layer Spikes")
        self.ax_raster_output.set_xlabel("Time (ms)")
        self.ax_raster_output.set_ylabel("Neuron Index")
        self.ax_raster_output.set_ylim(-0.5, self.num_output - 0.5)

        self.ax_weights.set_title("Synaptic Weights (Input -> Hidden Exc)")
        self.im_weights = self.ax_weights.imshow(np.zeros((self.num_hidden_exc, self.num_input)), cmap='viridis', aspect='auto', vmin=0, vmax=1)
        self.fig.colorbar(self.im_weights, ax=self.ax_weights, label="Weight Strength")

        self.ax_membrane.set_title(f"Membrane Potential (First {self.num_example_neurons_to_plot} Hidden Neurons)")
        self.ax_membrane.set_ylabel("Potential (mV)")
        self.membrane_lines = [self.ax_membrane.plot([], [], lw=1)[0] for _ in range(self.num_example_neurons_to_plot)]

        self.ax_neuromod.set_title("Neuromodulator Levels")
        self.ax_neuromod.set_ylabel("Level (a.u.)")
        self.ax_neuromod.set_ylim(-0.1, 1.1)
        self.dopamine_line, = self.ax_neuromod.plot([], [], label='Dopamine')
        self.acetylcholine_line, = self.ax_neuromod.plot([], [], label='Acetylcholine')
        self.serotonin_line, = self.ax_neuromod.plot([], [], label='Serotonin')
        self.ax_neuromod.legend(fontsize='small')

    def update_plots(self, current_time_ms, all_data):
        if not all_data: return

        synaptic_weights = all_data.get('synaptic_weights', {})
        membrane_potential_history = all_data.get('membrane_potential_history', [])
        dopamine_history = all_data.get('dopamine_history', deque())
        acetylcholine_history = all_data.get('acetylcholine_history', deque())
        serotonin_history = all_data.get('serotonin_history', deque())
        
        def plot_raster(ax, spike_history, color='black', offset=0):
            try:
                num_steps = len(spike_history)
                if num_steps == 0: return

                time_axis = np.linspace(max(0, current_time_ms - (num_steps - 1) * self.dt), current_time_ms, num_steps)
                
                spikes_list = list(spike_history)
                spikes_np = torch.stack(spikes_list).cpu().numpy().T

                spike_times_idx, neuron_indices = np.where(spikes_np > 0)

                valid_indices = spike_times_idx < len(time_axis)
                ax.scatter(time_axis[spike_times_idx[valid_indices]], neuron_indices[valid_indices] + offset, s=2, c=color, marker='|')
            except (RuntimeError, IndexError):
                pass # Gracefully skip plot update if a threading issue occurs

        # --- Update Rasters ---
        self.ax_raster_input.clear(); self.ax_raster_input.set_title("Input Layer Spikes"); self.ax_raster_input.set_ylim(-0.5, self.num_input - 0.5)
        plot_raster(self.ax_raster_input, self.spike_history_input)
        
        self.ax_raster_hidden.clear(); self.ax_raster_hidden.set_title("Hidden Layer Spikes (Exc/Inh)"); self.ax_raster_hidden.set_ylim(-0.5, self.num_hidden_exc + self.num_hidden_inh - 0.5)
        plot_raster(self.ax_raster_hidden, self.spike_history_hidden_exc, color='blue')
        plot_raster(self.ax_raster_hidden, self.spike_history_hidden_inh, color='red', offset=self.num_hidden_exc)
        self.ax_raster_hidden.axhline(y=self.num_hidden_exc - 0.5, color='white', linestyle='--', linewidth=1.5, zorder=10)

        self.ax_raster_output.clear(); self.ax_raster_output.set_title("Output Layer Spikes"); self.ax_raster_output.set_ylim(-0.5, self.num_output - 0.5)
        plot_raster(self.ax_raster_output, self.spike_history_output)
        
        max_len = len(dopamine_history)
        if max_len < 2: return
        
        master_time_axis = np.linspace(max(0, current_time_ms - (max_len - 1) * self.dt), current_time_ms, max_len)
        
        for ax in [self.ax_raster_input, self.ax_raster_hidden, self.ax_raster_output, self.ax_membrane, self.ax_neuromod]:
             ax.set_xlim(master_time_axis[0], master_time_axis[-1])

        if 'input_to_hidden_exc' in synaptic_weights:
            weights_np = synaptic_weights['input_to_hidden_exc']
            if weights_np.size > 0:
                self.im_weights.set_data(weights_np)
                self.im_weights.set_clim(vmin=np.min(weights_np), vmax=np.max(weights_np))

        # --- FIX: Robust plotting for line graphs to prevent shape mismatch ---
        def plot_line(line_artist, data_history, master_axis):
            try:
                data_list = list(data_history)
                len_data = len(data_list)
                if len_data < 2: 
                    line_artist.set_data([],[])
                    return

                time_axis = master_axis[-len_data:]
                # Final check to ensure lengths match before plotting
                if len(time_axis) == len(data_list):
                    line_artist.set_data(time_axis, data_list)
                else: # Fallback if there's still a mismatch
                    line_artist.set_data([],[])
            except Exception:
                line_artist.set_data([],[])

        for i in range(self.num_example_neurons_to_plot):
            if i < len(membrane_potential_history):
                plot_line(self.membrane_lines[i], membrane_potential_history[i], master_time_axis)
        self.ax_membrane.relim(); self.ax_membrane.autoscale_view(True, True, True)

        plot_line(self.dopamine_line, dopamine_history, master_time_axis)
        plot_line(self.acetylcholine_line, acetylcholine_history, master_time_axis)
        plot_line(self.serotonin_line, serotonin_history, master_time_axis)
