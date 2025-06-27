# src/network.py

import torch
import torch.nn as nn
from .neurons import LIFNeuron
from .synapses import Synapse
from .neuromodulation import NeuromodulatorSystem
from .memory import HippocampalModule, NeocorticalModule
from config import *

class AGINetwork(nn.Module):
    def __init__(self, hdc_system, data_encoder):
        super().__init__()
        
        self.num_hidden_exc = int(NUM_HIDDEN_NEURONS * EXC_INH_RATIO)
        self.num_hidden_inh = NUM_HIDDEN_NEURONS - self.num_hidden_exc

        self.neuromodulators = NeuromodulatorSystem()
        self.hippocampus = HippocampalModule(hdc_system, data_encoder)
        self.neocortex = NeocorticalModule(hdc_system)
        
        self.input_neurons = LIFNeuron(NUM_INPUT_NEURONS)
        self.exc_hidden_neurons = LIFNeuron(self.num_hidden_exc)
        self.inh_hidden_neurons = LIFNeuron(self.num_hidden_inh)
        self.output_neurons = LIFNeuron(NUM_OUTPUT_NEURONS)
        self.neuron_layers = nn.ModuleList([self.input_neurons, self.exc_hidden_neurons, self.inh_hidden_neurons, self.output_neurons])

        # --- FIX: Stronger initial weights to prevent dead network ---
        self.synapses = nn.ModuleDict({
            'input_to_hidden_exc': Synapse(NUM_INPUT_NEURONS, self.num_hidden_exc, connection_prob=FF_PROB, weight_scale=0.8, weight_offset=0.2),
            # --- FIX: Specifically strengthening the hidden->output pathway ---
            'hidden_exc_to_output': Synapse(self.num_hidden_exc, NUM_OUTPUT_NEURONS, connection_prob=FF_PROB, weight_scale=1.0, weight_offset=0.4),
            'hidden_exc_to_hidden_inh': Synapse(self.num_hidden_exc, self.num_hidden_inh, connection_prob=REC_PROB, weight_scale=0.5, weight_offset=0.1),
            'hidden_inh_to_hidden_exc': Synapse(self.num_hidden_inh, self.num_hidden_exc, connection_prob=REC_PROB, weight_scale=0.5, weight_offset=0.1),
            'recurrent_hidden_exc': Synapse(self.num_hidden_exc, self.num_hidden_exc, connection_prob=REC_PROB, weight_scale=0.5, weight_offset=0.1),
        })
        
        self.latest_spikes = {}

    def forward(self, input_current, current_time_ms, stdp_enabled=True):
        input_spikes = self.input_neurons(input_current, current_time_ms)
        
        current_exc = self.synapses['input_to_hidden_exc'](input_spikes)
        if 'hidden_exc_spikes' in self.latest_spikes:
            current_exc += self.synapses['recurrent_hidden_exc'](self.latest_spikes['hidden_exc_spikes'])
        if 'hidden_inh_spikes' in self.latest_spikes:
             current_exc -= self.synapses['hidden_inh_to_hidden_exc'](self.latest_spikes['hidden_inh_spikes'])
        
        exc_hidden_spikes = self.exc_hidden_neurons(current_exc, current_time_ms)
        
        current_inh = self.synapses['hidden_exc_to_hidden_inh'](exc_hidden_spikes)
        inh_hidden_spikes = self.inh_hidden_neurons(current_inh, current_time_ms)

        output_current = self.synapses['hidden_exc_to_output'](exc_hidden_spikes)
        output_spikes = self.output_neurons(output_current, current_time_ms)
        
        self.latest_spikes = {
            'input_spikes': input_spikes,
            'hidden_exc_spikes': exc_hidden_spikes,
            'hidden_inh_spikes': inh_hidden_spikes,
            'output_spikes': output_spikes
        }

        if stdp_enabled:
            self.synapses['input_to_hidden_exc'].update_stdp(self.input_neurons.last_spike_times, self.exc_hidden_neurons.last_spike_times, self.neuromodulators)
            self.synapses['hidden_exc_to_output'].update_stdp(self.exc_hidden_neurons.last_spike_times, self.output_neurons.last_spike_times, self.neuromodulators)
            self.synapses['hidden_exc_to_hidden_inh'].update_stdp(self.exc_hidden_neurons.last_spike_times, self.inh_hidden_neurons.last_spike_times, self.neuromodulators)
            self.synapses['hidden_inh_to_hidden_exc'].update_stdp(self.inh_hidden_neurons.last_spike_times, self.exc_hidden_neurons.last_spike_times, self.neuromodulators)
            self.synapses['recurrent_hidden_exc'].update_stdp(self.exc_hidden_neurons.last_spike_times, self.exc_hidden_neurons.last_spike_times, self.neuromodulators)

        if torch.sum(exc_hidden_spikes) > (self.num_hidden_exc * 0.01):
            reward_val = self.neuromodulators.dopamine_level.item()
            # Pass current time to create a temporal context
            self.hippocampus.encode_pattern(exc_hidden_spikes, {
                'reward_val': reward_val, 
                'time_val': current_time_ms
            })

        if int(current_time_ms) % 10 == 0:
            self.hippocampus.trigger_replay(self.neocortex, self.neuromodulators)
            self.neuromodulators.update_decay()

        return self.latest_spikes | {
            'dopamine_level': self.neuromodulators.dopamine_level.item(),
            'acetylcholine_level': self.neuromodulators.acetylcholine_level.item(),
            'serotonin_level': self.neuromodulators.serotonin_level.item(),
        }

    def apply_developmental_bias(self, avg_input_spikes, avg_hidden_exc_spikes):
        self.synapses['input_to_hidden_exc'].initialize_developmental_bias(
            avg_input_spikes, avg_hidden_exc_spikes, DEVELOPMENTAL_BIAS_STRENGTH
        )

    def get_latest_spikes(self, layer_name):
        return self.latest_spikes.get(layer_name)

    def get_all_synaptic_weights(self):
        return {name: synapse.get_weights() for name, synapse in self.synapses.items()}
        
    def reset(self):
        for layer in self.neuron_layers:
            layer.reset()
        for name, synapse in self.synapses.items():
             synapse.__init__(synapse.num_pre, synapse.num_post, connection_prob=synapse.connection_prob, weight_scale=synapse.weight_scale, weight_offset=synapse.weight_offset)
        self.neuromodulators.reset()
        self.hippocampus.reset()
        self.neocortex.reset()
        self.latest_spikes.clear()
