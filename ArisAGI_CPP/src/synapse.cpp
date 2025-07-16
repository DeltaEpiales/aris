#include "synapse.h"
#include "config.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

Synapse::Synapse(int num_pre, int num_post, float connection_prob, float weight_scale, float weight_offset)
    : m_num_pre(num_pre), m_num_post(num_post) {

    CUDA_CHECK(cudaMalloc(&d_weights, m_num_pre * m_num_post * sizeof(float)));

    // Weights are initialized on the host (CPU) and then copied to the device (GPU).
    Eigen::MatrixXf h_weights_temp(m_num_post, m_num_pre);

    std::default_random_engine generator(config::RANDOM_SEED);
    std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<float> weight_dist(0.0, 1.0);

    // Initialize weights with sparsity based on connection_prob.
    for (int i = 0; i < m_num_post; ++i) {
        for (int j = 0; j < m_num_pre; ++j) {
            if (prob_dist(generator) < connection_prob) {
                h_weights_temp(i, j) = weight_dist(generator) * weight_scale + weight_offset;
            } else {
                h_weights_temp(i, j) = 0.0f;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights_temp.data(), m_num_pre * m_num_post * sizeof(float), cudaMemcpyHostToDevice));
}

Synapse::~Synapse() {
    cudaFree(d_weights);
}

void Synapse::forward(const bool* d_pre_spikes, float* d_output_current) {
    // Propagate spikes by performing a matrix-vector multiplication on the GPU.
    // Note: For peak performance, this custom kernel should be replaced with a
    // highly optimized library call like cuBLAS's sgemv.
    matMulCUDA(d_weights, d_pre_spikes, d_output_current, m_num_post, m_num_pre);
}

void Synapse::updateSTDP(const float* d_pre_last_spikes, const float* d_post_last_spikes, const NeuromodulatorSystem& neuromodulators) {
    // Launch the CUDA kernel to update all weights in parallel.
    updateSTDPCUDA(d_weights, d_pre_last_spikes, d_post_last_spikes, m_num_pre, m_num_post,
                   neuromodulators.getDopamineLevel(), neuromodulators.getAcetylcholineLevel());
}

Eigen::MatrixXf Synapse::getWeights() const {
    Eigen::MatrixXf h_weights_temp(m_num_post, m_num_pre);
    // Copy weights from GPU to CPU for visualization.
    CUDA_CHECK(cudaMemcpy(h_weights_temp.data(), d_weights, m_num_pre * m_num_post * sizeof(float), cudaMemcpyDeviceToHost));
    return h_weights_temp;
}

void Synapse::setWeights(const Eigen::MatrixXf& weights) {
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), m_num_pre * m_num_post * sizeof(float), cudaMemcpyHostToDevice));
}

void Synapse::initializeDevelopmentalBias(const Eigen::VectorXf& pre_activity, const Eigen::VectorXf& post_activity) {
    // This is a one-time host-side operation.
    Eigen::MatrixXf h_weights = getWeights();
    Eigen::MatrixXf co_activity = post_activity * pre_activity.transpose();
    h_weights += co_activity * config::DEVELOPMENTAL_BIAS_STRENGTH;
    h_weights = h_weights.cwiseMax(config::W_MIN).cwiseMin(config::W_MAX);
    setWeights(h_weights);
}