#include "lif_neuron.h"
#include "config.h"
#include <cuda_runtime.h>
#include <iostream>

// A simple macro for robust CUDA error checking.
#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

LIFNeuron::LIFNeuron(int num_neurons) : m_num_neurons(num_neurons) {
    // Allocate all necessary memory on the GPU device.
    CUDA_CHECK(cudaMalloc(&d_v, m_num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dynamic_thresh, m_num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_refractory_countdown, m_num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last_spike_times, m_num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spikes, m_num_neurons * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_input_current, m_num_neurons * sizeof(float)));
    reset(); // Set initial values.
}

LIFNeuron::~LIFNeuron() {
    // Free all GPU memory to prevent leaks.
    cudaFree(d_v);
    cudaFree(d_dynamic_thresh);
    cudaFree(d_refractory_countdown);
    cudaFree(d_last_spike_times);
    cudaFree(d_spikes);
    cudaFree(d_input_current);
}

void LIFNeuron::reset() {
    // Create temporary host vectors with initial values from the config file.
    Eigen::VectorXf h_v_temp = Eigen::VectorXf::Constant(m_num_neurons, config::V_REST);
    Eigen::VectorXf h_thresh_temp = Eigen::VectorXf::Constant(m_num_neurons, config::V_THRESH);
    Eigen::VectorXf h_refractory_temp = Eigen::VectorXf::Zero(m_num_neurons);
    Eigen::VectorXf h_last_spike_temp = Eigen::VectorXf::Constant(m_num_neurons, -1e9f); // Large negative number for "never spiked"

    // Copy these initial values from the host (CPU) to the device (GPU).
    CUDA_CHECK(cudaMemcpy(d_v, h_v_temp.data(), m_num_neurons * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dynamic_thresh, h_thresh_temp.data(), m_num_neurons * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_refractory_countdown, h_refractory_temp.data(), m_num_neurons * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_last_spike_times, h_last_spike_temp.data(), m_num_neurons * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_spikes, 0, m_num_neurons * sizeof(bool))); // Initialize spikes to false.
}

void LIFNeuron::update(const Eigen::VectorXf& input_current, float current_time_ms) {
    // Copy the latest input current from the host to the device's input buffer.
    CUDA_CHECK(cudaMemcpy(d_input_current, input_current.data(), m_num_neurons * sizeof(float), cudaMemcpyHostToDevice));
    // Launch the CUDA kernel to perform the neuron state updates in parallel on the GPU.
    updateLIFNeuronsCUDA(d_v, d_dynamic_thresh, d_refractory_countdown, d_last_spike_times, d_input_current, d_spikes, m_num_neurons, current_time_ms);
}

Eigen::VectorXf LIFNeuron::getPotentialsHost() const {
    Eigen::VectorXf h_v_temp(m_num_neurons);
    // Copy membrane potentials from GPU back to CPU for visualization.
    CUDA_CHECK(cudaMemcpy(h_v_temp.data(), d_v, m_num_neurons * sizeof(float), cudaMemcpyDeviceToHost));
    return h_v_temp;
}

Eigen::VectorXf LIFNeuron::getSpikesHost() const {
    Eigen::VectorXf h_spikes_temp(m_num_neurons);
    bool* h_bool_spikes = new bool[m_num_neurons];
    // Copy spike data from GPU back to CPU for visualization.
    CUDA_CHECK(cudaMemcpy(h_bool_spikes, d_spikes, m_num_neurons * sizeof(bool), cudaMemcpyDeviceToHost));
    for(int i=0; i<m_num_neurons; ++i) {
        h_spikes_temp(i) = h_bool_spikes[i] ? 1.0f : 0.0f;
    }
    delete[] h_bool_spikes;
    return h_spikes_temp;
}