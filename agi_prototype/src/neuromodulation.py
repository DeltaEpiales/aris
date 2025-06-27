# agi_prototype/src/neuromodulation.py

import torch
from config import DT, DOPAMINE_MOD, ACETYLCHOLINE_MOD, SEROTONIN_MOD, DEVICE

class NeuromodulatorSystem:
    """
    Simulates levels of key neuromodulators, which decay over time and can be
    pulsed by specific events, influencing system-wide learning and behavior.
    """
    def __init__(self):
        self.dopamine_level = torch.tensor(DOPAMINE_MOD['baseline_dopamine'], device=DEVICE)
        self.acetylcholine_level = torch.tensor(0.0, device=DEVICE)
        self.serotonin_level = torch.tensor(0.0, device=DEVICE)

        self.dopamine_decay = torch.exp(torch.tensor(-DT / DOPAMINE_MOD['decay_tau'], device=DEVICE))
        self.acetylcholine_decay = torch.exp(torch.tensor(-DT / ACETYLCHOLINE_MOD['decay_tau'], device=DEVICE))
        self.serotonin_decay = torch.exp(torch.tensor(-DT / SEROTONIN_MOD['decay_tau'], device=DEVICE))

    def update_decay(self):
        """Applies exponential decay to all neuromodulator levels."""
        self.dopamine_level *= self.dopamine_decay
        self.dopamine_level.clamp_(min=DOPAMINE_MOD['baseline_dopamine'])
        
        self.acetylcholine_level *= self.acetylcholine_decay
        self.acetylcholine_level.clamp_(min=0.0)

        self.serotonin_level *= self.serotonin_decay
        self.serotonin_level.clamp_(min=0.0)

    def set_dopamine_pulse(self, magnitude: float):
        """Increases dopamine level (e.g., due to prediction surprise)."""
        self.dopamine_level += magnitude
        self.dopamine_level.clamp_(max=1.0)

    def set_acetylcholine_pulse(self, magnitude: float):
        """Increases acetylcholine level (e.g., due to uncertainty)."""
        self.acetylcholine_level += magnitude
        self.acetylcholine_level.clamp_(max=1.0)

    def set_serotonin_pulse(self, magnitude: float):
        """Increases serotonin level (e.g., due to redundancy/cost)."""
        self.serotonin_level += magnitude
        self.serotonin_level.clamp_(max=1.0)
        
    def reset(self):
        """Resets levels to their initial state."""
        self.dopamine_level.fill_(DOPAMINE_MOD['baseline_dopamine'])
        self.acetylcholine_level.zero_()
        self.serotonin_level.zero_()
