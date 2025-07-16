#pragma once

#include <cuda_runtime.h>

void updateLIFNeuronsCUDA(
    float* d_v, float* d_dynamic_thresh, float* d_refractory_countdown,
    float* d_last_spike_times, const float* d_input_current,
    bool* d_spikes, int num_neurons, float current_time_ms
);

void updateSTDPCUDA(
    float* d_weights, const float* d_pre_last_spikes, const float* d_post_last_spikes,
    int num_pre, int num_post, float dopamine_level, float acetylcholine_level
);

void matMulCUDA(const float* d_A, const bool* d_x, float* d_y, int rows, int cols);