#pragma once

#include <Eigen/Dense>
#include "neuromodulation.h"
#include "kernels.h"

class Synapse {
public:
    Synapse(int num_pre, int num_post, float connection_prob, float weight_scale, float weight_offset);
    ~Synapse();

    Synapse(const Synapse&) = delete;
    Synapse& operator=(const Synapse&) = delete;

    void forward(const bool* d_pre_spikes, float* d_output_current);
    void updateSTDP(const float* d_pre_last_spikes, const float* d_post_last_spikes, const NeuromodulatorSystem& neuromodulators);
    void initializeDevelopmentalBias(const Eigen::VectorXf& pre_activity, const Eigen::VectorXf& post_activity);

    Eigen::MatrixXf getWeights() const;
    void setWeights(const Eigen::MatrixXf& weights);

private:
    int m_num_pre;
    int m_num_post;
    float* d_weights;
};