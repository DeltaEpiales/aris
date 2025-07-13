#include "kernels.h"
#include "config.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>

// A simple macro for robust CUDA error checking.
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// --- CUDA Kernel for LIF Neuron Update ---
// This kernel computes the state of all neurons for one timestep in parallel.
// Each thread is responsible for a single neuron.
__global__ void lif_update_kernel(
    float* v, float* dynamic_thresh, float* refractory_countdown,
    float* last_spike_times, const float* input_current,
    bool* spikes, int num_neurons, float current_time_ms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    // Pre-calculate decay factor from config constants for efficiency.
    float decay_factor = expf(-config::DT / config::TAU_M);

    // Only update neurons that are not in their refractory period.
    if (refractory_countdown[idx] <= 0.0f) {
        // Update membrane potential based on the leaky integrate model.
        v[idx] = config::V_REST + (v[idx] - config::V_REST) * decay_factor + input_current[idx];

        // Check if the neuron's potential crosses the firing threshold.
        if (v[idx] >= dynamic_thresh[idx]) {
            spikes[idx] = true;
            v[idx] = config::V_RESET; // Reset potential
            refractory_countdown[idx] = config::TAU_REF; // Start refractory period
            last_spike_times[idx] = current_time_ms; // Record spike time for STDP
        } else {
            spikes[idx] = false;
        }
    } else {
        spikes[idx] = false;
    }

    // Decay the refractory timer for all neurons.
    refractory_countdown[idx] -= config::DT;
    if (refractory_countdown[idx] < 0.0f) refractory_countdown[idx] = 0.0f;

    // Update the dynamic threshold (intrinsic plasticity).
    float target_thresh = spikes[idx] ? config::MAX_THRESH : config::MIN_THRESH;
    dynamic_thresh[idx] += config::ADAPTATION_RATE * (target_thresh - dynamic_thresh[idx]);
    dynamic_thresh[idx] = fmaxf(config::MIN_THRESH, fminf(config::MAX_THRESH, dynamic_thresh[idx]));
}

// Host wrapper function to launch the LIF update kernel.
void updateLIFNeuronsCUDA(
    float* d_v, float* d_dynamic_thresh, float* d_refractory_countdown,
    float* d_last_spike_times, const float* d_input_current,
    bool* d_spikes, int num_neurons, float current_time_ms)
{
    int threads_per_block = 256;
    int blocks_per_grid = (num_neurons + threads_per_block - 1) / threads_per_block;
    lif_update_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_v, d_dynamic_thresh, d_refractory_countdown, d_last_spike_times,
        d_input_current, d_spikes, num_neurons, current_time_ms
    );
}

// --- CUDA Kernel for STDP Update ---
// This kernel updates all synapses in parallel using a 2D grid of threads.
// Each thread is responsible for a single synapse (weight).
__global__ void stdp_update_kernel(
    float* weights, const float* pre_last_spikes, const float* post_last_spikes,
    int num_pre, int num_post, float dopamine_level, float acetylcholine_level)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Post-synaptic neuron index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Pre-synaptic neuron index

    if (i >= num_post || j >= num_pre) return;

    int weight_idx = i * num_pre + j;
    float current_weight = weights[weight_idx];

    // Only update weights that are not at their boundaries to save computation.
    if (current_weight > config::W_MIN && current_weight < config::W_MAX) {
        float dt = post_last_spikes[i] - pre_last_spikes[j];
        float dw = 0.0f;

        // LTP (Long-Term Potentiation): Pre-synaptic spike occurs before post-synaptic.
        if (dt > 0 && dt < config::TAU_PLUS * 5.0f) {
            dw = config::A_PLUS * expf(-dt / config::TAU_PLUS);
            // Learning is gated by neuromodulators.
            dw *= dopamine_level * (1.0f + acetylcholine_level);
        }
        // LTD (Long-Term Depression): Post-synaptic spike occurs before pre-synaptic.
        else if (dt < 0 && dt > -config::TAU_MINUS * 5.0f) {
            dw = -config::A_MINUS * expf(dt / config::TAU_MINUS);
            dw *= dopamine_level; // Dopamine gates both LTP and LTD.
        }

        // Apply weight change and structural plasticity (decay).
        float new_weight = current_weight + dw;
        new_weight -= config::DECAY_RATE * (new_weight - config::W_MIN) * config::DT;

        // Clamp the weight to its valid range.
        weights[weight_idx] = fmaxf(config::W_MIN, fminf(config::W_MAX, new_weight));
    }
}

// Host wrapper function to launch the STDP update kernel.
void updateSTDPCUDA(
    float* d_weights, const float* d_pre_last_spikes, const float* d_post_last_spikes,
    int num_pre, int num_post, float dopamine_level, float acetylcholine_level)
{
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (num_pre + threads_per_block.x - 1) / threads_per_block.x,
        (num_post + threads_per_block.y - 1) / threads_per_block.y
    );
    stdp_update_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_weights, d_pre_last_spikes, d_post_last_spikes, num_pre, num_post,
        dopamine_level, acetylcholine_level
    );
}

// --- CUDA Kernel for Matrix-Vector Multiplication (Spike Propagation) ---
// This is a simplified, textbook implementation. For peak performance, this should
// be replaced with a call to the highly optimized cuBLAS library (e.g., cublasSgemv).
__global__ void mat_mul_kernel(const float* A, const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            sum += A[row * cols + i] * x[i];
        }
        y[row] = sum;
    }
}

// Helper kernel to convert a boolean spike vector to a float vector for matrix multiplication.
__global__ void boolToFloatKernel(const bool* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] ? 1.0f : 0.0f;
    }
}

// Host wrapper for the matrix multiplication.
void matMulCUDA(const float* d_A, const bool* d_x_bool, float* d_y, int rows, int cols) {
    float* d_x_float;
    CUDA_CHECK(cudaMalloc(&d_x_float, cols * sizeof(float)));

    int threads = 256;
    int blocks = (cols + threads - 1) / threads;
    boolToFloatKernel<<<blocks, threads>>>(d_x_bool, d_x_float, cols);

    blocks = (rows + threads - 1) / threads;
    mat_mul_kernel<<<blocks, threads>>>(d_A, d_x_float, d_y, rows, cols);

    cudaFree(d_x_float);
}