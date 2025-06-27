# agi_prototype/src/neurons.py

import torch
import torch.nn as nn
from config import LIF_PARAMS, DT, DEVICE

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron model with refractory period
    and intrinsic plasticity (dynamic threshold).
    """
    def __init__(self, num_neurons: int):
        super().__init__()
        self.num_neurons = num_neurons

        # --- LIF Parameters ---
        self.v_rest = LIF_PARAMS['v_rest']
        self.v_reset = LIF_PARAMS['v_reset']
        self.tau_m = LIF_PARAMS['tau_m']
        self.tau_ref = LIF_PARAMS['tau_ref']
        
        # --- FIX: TypeError resolved by converting the float argument to a tensor.
        # Membrane potential decay factor, calculated once
        self.decay_factor = torch.exp(torch.tensor(-DT / self.tau_m, device=DEVICE))

        # --- Intrinsic Plasticity (Dynamic Threshold) Parameters ---
        self.adaptation_rate = LIF_PARAMS['adaptation_rate']
        self.min_thresh = LIF_PARAMS['min_thresh']
        self.max_thresh = LIF_PARAMS['max_thresh']
        
        # --- State Variables ---
        self.v = torch.full((self.num_neurons,), self.v_rest, device=DEVICE)
        self.dynamic_threshold = torch.full((self.num_neurons,), LIF_PARAMS['v_thresh'], device=DEVICE)
        self.refractory_countdown = torch.zeros(self.num_neurons, device=DEVICE)
        self.last_spike_times = torch.full((self.num_neurons,), -float('inf'), device=DEVICE)

    def forward(self, postsynaptic_current: torch.Tensor, current_time_ms: float):
        """
        Updates the neuron states for one time step.
        """
        not_in_refractory = self.refractory_countdown <= 0

        # Update membrane potential using the decay factor formulation
        self.v[not_in_refractory] = self.v_rest + (self.v[not_in_refractory] - self.v_rest) * self.decay_factor + postsynaptic_current[not_in_refractory]
        
        # Clamp voltage to prevent extreme values, can help with stability
        self.v.clamp_(min=-100.0, max=0.0)

        # Check for spikes against the dynamic threshold
        spikes = (self.v >= self.dynamic_threshold) & not_in_refractory
        
        if torch.any(spikes):
            self.last_spike_times[spikes] = current_time_ms
            self.v[spikes] = self.v_reset
            self.refractory_countdown[spikes] = self.tau_ref

        # Decay refractory countdown
        self.refractory_countdown -= DT
        self.refractory_countdown.clamp_(min=0)

        # Update dynamic threshold (intrinsic plasticity)
        target_thresh = torch.where(spikes, self.max_thresh, self.min_thresh)
        self.dynamic_threshold += self.adaptation_rate * (target_thresh - self.dynamic_threshold)
        
        return spikes.float()

    def get_state(self):
        """Returns neuron states as numpy arrays for visualization."""
        return {
            'membrane_potential': self.v.detach().cpu().numpy(),
            'spikes': (self.refractory_countdown > (self.tau_ref - DT)).detach().cpu().numpy(),
        }
        
    def reset(self):
        """Resets all state variables to their initial values."""
        self.v.fill_(self.v_rest)
        self.dynamic_threshold.fill_(LIF_PARAMS['v_thresh'])
        self.refractory_countdown.zero_()
        self.last_spike_times.fill_(-float('inf'))
