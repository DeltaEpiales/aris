#pragma once

#include <Eigen/Dense>
#include "neuromodulation.h"
#include "kernels.h"

/**
 * @class Synapse
 * @brief Manages a matrix of synaptic weights connecting two neuron populations on the GPU.
 * This class is responsible for signal propagation (forward pass) and learning (STDP).
 */
class Synapse {
public:
    Synapse(int num_pre, int num_post, float connection_prob, float weight_scale, float weight_offset);
    ~Synapse();

    Synapse(const Synapse&) = delete;
    Synapse& operator=(const Synapse&) = delete;

    /**
     * @brief Propagates spikes from a presynaptic population to generate postsynaptic current.
     * @param d_pre_spikes Device pointer to the boolean spike vector of the presynaptic layer.
     * @param d_output_current Device pointer to the output current vector to be populated.
     */
    void forward(const bool* d_pre_spikes, float* d_output_current);

    /**
     * @brief Updates all synaptic weights according to the STDP rule, gated by neuromodulators.
     */
    void updateSTDP(const float* d_pre_last_spikes, const float* d_post_last_spikes, const NeuromodulatorSystem& neuromodulators);

    /**
     * @brief Applies a Hebbian-like bias to weights based on early activity patterns.
     */
    void initializeDevelopmentalBias(const Eigen::VectorXf& pre_activity, const Eigen::VectorXf& post_activity);

    Eigen::MatrixXf getWeights() const;
    void setWeights(const Eigen::MatrixXf& weights);

private:
    int m_num_pre;
    int m_num_post;
    float* d_weights; // Synaptic weights stored on the GPU
};