#pragma once

#include <cuda_runtime.h>

/**
 * @brief CUDA kernel to update the state of all LIF neurons in parallel.
 * Each thread handles one neuron.
 */
void updateLIFNeuronsCUDA(
    float* d_v, float* d_dynamic_thresh, float* d_refractory_countdown,
    float* d_last_spike_times, const float* d_input_current,
    bool* d_spikes, int num_neurons, float current_time_ms
);

/**
 * @brief CUDA kernel to update all synaptic weights based on STDP.
 * Each thread handles one synapse in a 2D grid.
 */
void updateSTDPCUDA(
    float* d_weights, const float* d_pre_last_spikes, const float* d_post_last_spikes,
    int num_pre, int num_post, float dopamine_level, float acetylcholine_level
);

/**
 * @brief CUDA kernel for matrix-vector multiplication (Spike Propagation).
 * This is a simplified implementation. For production, cuBLAS should be used.
 */
void matMulCUDA(const float* d_A, const bool* d_x, float* d_y, int rows, int cols);