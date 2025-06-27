# agi_prototype/src/data_encoders.py

import torch
from config import DEVICE, HD_DIM
from .hdc import HDC

class DataEncoder:
    """Handles encoding/decoding between SNN spike patterns and HDC hypervectors."""
    def __init__(self, num_neurons_for_pattern_encoding: int, hdc_system: HDC):
        self.hdc = hdc_system
        self.num_neurons = num_neurons_for_pattern_encoding
        # Ensure basis vectors for all neurons are created
        self.hdc.init_basis_vectors(self.num_neurons, prefix="neuron")

    def encode_rate_to_hypervector(self, rates: torch.Tensor):
        """Encodes an analog rate vector into a single hypervector."""
        weighted_hvs = []
        for i, rate in enumerate(rates):
            if rate > 0:
                neuron_hv = self.hdc.get_basis_vector(f"neuron_{i}")
                weighted_hvs.append(neuron_hv * rate.item())
        
        if not weighted_hvs:
            return torch.zeros(self.hdc.dim, device=DEVICE)
            
        return self.hdc.bundle(weighted_hvs)

    def encode_spike_pattern_to_hypervector(self, spike_pattern: torch.Tensor):
        """Encodes a binary spike pattern into a single hypervector."""
        active_neuron_indices = torch.where(spike_pattern > 0)[0]
        if len(active_neuron_indices) == 0:
            return torch.zeros(self.hdc.dim, device=DEVICE)

        active_hvs = [self.hdc.get_basis_vector(f"neuron_{i.item()}") for i in active_neuron_indices]
        return self.hdc.bundle(active_hvs)
        