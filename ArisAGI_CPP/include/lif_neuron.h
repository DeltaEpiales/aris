#pragma once

#include <Eigen/Dense>
#include "kernels.h"

/**
 * @class LIFNeuron
 * @brief Manages a population of Leaky Integrate-and-Fire neurons on the GPU.
 * This class handles the allocation of CUDA memory and calls the appropriate
 * kernels to update the neuron states in parallel.
 */
class LIFNeuron {
public:
    explicit LIFNeuron(int num_neurons);
    ~LIFNeuron();

    // Disallow copy and assignment to prevent shallow copies of GPU pointers.
    LIFNeuron(const LIFNeuron&) = delete;
    LIFNeuron& operator=(const LIFNeuron&) = delete;

    /**
     * @brief The main update function. Copies input current to the GPU and launches the update kernel.
     * @param input_current The total synaptic current for each neuron.
     * @param current_time_ms The current simulation time, needed for STDP.
     */
    void update(const Eigen::VectorXf& input_current, float current_time_ms);

    // --- Getters for Device Pointers (for inter-kernel operations) ---
    const bool* getSpikesDevicePtr() const { return d_spikes; }
    const float* getLastSpikeTimesDevicePtr() const { return d_last_spike_times; }

    // --- Getters for Host Data (for visualization and CPU logic) ---
    Eigen::VectorXf getPotentialsHost() const;
    Eigen::VectorXf getSpikesHost() const;

    void reset();

private:
    int m_num_neurons;

    // Device-side (GPU) vectors for high-performance computation.
    float* d_v;                     // Membrane potentials
    float* d_dynamic_thresh;        // Adaptive thresholds
    float* d_refractory_countdown;  // Refractory timers
    float* d_last_spike_times;      // Timestamps of last spikes for STDP
    bool* d_spikes;                // Spike output (boolean)
    float* d_input_current;         // Buffer for incoming current
};