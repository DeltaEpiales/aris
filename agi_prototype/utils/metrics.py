# agi_prototype/utils/metrics.py

import torch
from collections import deque
from config import DEVICE

def calculate_firing_rate(spikes_history: deque, num_neurons: int, window_ms: float, dt: float):
    """
    Calculates the average firing rate for a population of neurons over a given time window.
    Args:
        spikes_history (collections.deque): Deque of spike tensors (0s and 1s).
        num_neurons (int): Number of neurons in the population.
        window_ms (float): Time window in milliseconds.
        dt (float): Simulation time step in milliseconds.
    Returns:
        float: Average firing rate in Hz for the entire population.
    """
    if not spikes_history or num_neurons == 0:
        return 0.0

    num_steps = len(spikes_history)
    if num_steps == 0:
        return 0.0

    # Stack the recent spikes from the deque into a single tensor
    spikes_tensor = torch.stack(list(spikes_history))
    
    # Calculate total spikes in the window
    total_spikes = torch.sum(spikes_tensor).item()
    
    # Total time window in seconds
    window_sec = (num_steps * dt) / 1000.0
    if window_sec == 0:
        return 0.0
        
    # Calculate average firing rate in Hz for the population
    # Rate = (Total Spikes) / (Number of Neurons * Time Window in Seconds)
    avg_rate_hz = total_spikes / (num_neurons * window_sec)
    
    return avg_rate_hz

def calculate_sparsity(spikes: torch.Tensor):
    """
    Calculates the sparsity of firing activity in a given tensor of spikes.
    Sparsity = 1 - (fraction of active neurons)
    A value of 1.0 means no neurons are firing.
    A value of 0.0 means all neurons are firing.
    """
    if spikes is None or spikes.numel() == 0:
        return 1.0
    
    active_fraction = (spikes > 0).float().sum() / spikes.numel()
    return 1.0 - active_fraction.item()

def calculate_network_stability(membrane_potentials: torch.Tensor, threshold: torch.Tensor):
    """
    Evaluates network stability. A stable network avoids runaway excitation or inactivity.
    Returns standard deviation of potentials and average distance to threshold.
    """
    if membrane_potentials.numel() == 0:
        return 0.0, 0.0
    
    std_v = torch.std(membrane_potentials).item()
    
    # Average distance to threshold for all neurons
    avg_dist_to_thresh = torch.mean(threshold - membrane_potentials).item()

    return std_v, avg_dist_to_thresh

def calculate_weight_change_magnitude(old_weights: torch.Tensor, new_weights: torch.Tensor):
    """
    Calculates the average magnitude of weight changes.
    """
    if old_weights.shape != new_weights.shape:
        return 0.0
    return torch.mean(torch.abs(new_weights - old_weights)).item()
