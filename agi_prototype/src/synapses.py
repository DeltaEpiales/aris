# src/synapses.py

import torch
import torch.nn as nn
from config import STDP_PARAMS, DT, DEVICE

class Synapse(nn.Module):
    def __init__(self, num_pre: int, num_post: int, connection_prob: float = 1.0, weight_scale=0.5, weight_offset=0.1):
        super().__init__()
        self.num_pre = num_pre
        self.num_post = num_post
        self.connection_prob = connection_prob
        self.weight_scale = weight_scale
        self.weight_offset = weight_offset

        mask = torch.rand(num_post, num_pre) < connection_prob
        initial_weights = torch.rand(num_post, num_pre) * self.weight_scale + self.weight_offset
        self.weights = nn.Parameter(initial_weights.to(DEVICE))
        self.weights.data[~mask] = 0.0

        self.A_plus = STDP_PARAMS['A_plus']
        self.A_minus = STDP_PARAMS['A_minus']
        self.tau_plus = STDP_PARAMS['tau_plus']
        self.tau_minus = STDP_PARAMS['tau_minus']
        self.w_min = STDP_PARAMS['w_min']
        self.w_max = STDP_PARAMS['w_max']
        self.decay_rate = STDP_PARAMS['decay_rate']

    def forward(self, pre_spikes: torch.Tensor):
        return torch.matmul(self.weights, pre_spikes)

    def update_stdp(self, pre_last_spikes, post_last_spikes, neuromodulators):
        pre_times_matrix = pre_last_spikes.unsqueeze(0).expand(self.num_post, -1)
        post_times_matrix = post_last_spikes.unsqueeze(1).expand(-1, self.num_pre)
        dt_matrix = post_times_matrix - pre_times_matrix
        
        ltd_mask = (dt_matrix > 0) & (dt_matrix < self.tau_plus * 5)
        dw_plus = self.A_plus * torch.exp(-dt_matrix / self.tau_plus)
        
        ltp_mask = (dt_matrix < 0) & (dt_matrix > -self.tau_minus * 5)
        dw_minus = -self.A_minus * torch.exp(dt_matrix / self.tau_minus)

        learning_rate = neuromodulators.dopamine_level.item()
        attention_boost = neuromodulators.acetylcholine_level.item()
        
        self.weights.data[ltd_mask] += dw_plus[ltd_mask] * learning_rate * (1 + attention_boost)
        self.weights.data[ltp_mask] += dw_minus[ltp_mask] * learning_rate

        self.weights.data -= self.decay_rate * (self.weights.data - self.w_min) * DT
        self.weights.data.clamp_(min=self.w_min, max=self.w_max)

    def initialize_developmental_bias(self, avg_pre_spikes, avg_post_spikes, bias_strength):
        co_activity = torch.outer(avg_post_spikes, avg_pre_spikes)
        self.weights.data += co_activity * bias_strength
        self.weights.data.clamp_(min=self.w_min, max=self.w_max)

    def get_weights(self):
        return self.weights.detach().cpu().numpy()
