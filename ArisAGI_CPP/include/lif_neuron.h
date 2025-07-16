#pragma once

#include <Eigen/Dense>
#include "kernels.h"

class LIFNeuron {
public:
    explicit LIFNeuron(int num_neurons);
    ~LIFNeuron();

    LIFNeuron(const LIFNeuron&) = delete;
    LIFNeuron& operator=(const LIFNeuron&) = delete;

    void update(const Eigen::VectorXf& input_current, float current_time_ms);

    const bool* getSpikesDevicePtr() const { return d_spikes; }
    const float* getLastSpikeTimesDevicePtr() const { return d_last_spike_times; }

    Eigen::VectorXf getPotentialsHost() const;
    Eigen::VectorXf getSpikesHost() const;

    void reset();

private:
    int m_num_neurons;

    float* d_v;
    float* d_dynamic_thresh;
    float* d_refractory_countdown;
    float* d_last_spike_times;
    bool* d_spikes;
    float* d_input_current;
};