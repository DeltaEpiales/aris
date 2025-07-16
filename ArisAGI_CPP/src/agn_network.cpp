#include "agn_network.h"
#include "config.h"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

AGINetwork::AGINetwork(HDC& hdc) : m_hdc(hdc) {
    m_num_hidden_exc = static_cast<int>(config::NUM_HIDDEN_NEURONS * config::EXC_INH_RATIO);
    m_num_hidden_inh = config::NUM_HIDDEN_NEURONS - m_num_hidden_exc;

    // Instantiate neuron populations
    m_input_neurons = std::make_unique<LIFNeuron>(config::NUM_INPUT_NEURONS);
    m_exc_hidden_neurons = std::make_unique<LIFNeuron>(m_num_hidden_exc);
    m_inh_hidden_neurons = std::make_unique<LIFNeuron>(m_num_hidden_inh);
    m_output_neurons = std::make_unique<LIFNeuron>(config::NUM_OUTPUT_NEURONS);

    // Instantiate synaptic connections with specific initial weight properties
    m_synapses["input_to_hidden_exc"] = std::make_unique<Synapse>(config::NUM_INPUT_NEURONS, m_num_hidden_exc, config::FF_PROB, 0.8f, 0.2f);
    m_synapses["hidden_exc_to_output"] = std::make_unique<Synapse>(m_num_hidden_exc, config::NUM_OUTPUT_NEURONS, config::FF_PROB, 1.0f, 0.4f);
    m_synapses["hidden_exc_to_hidden_inh"] = std::make_unique<Synapse>(m_num_hidden_exc, m_num_hidden_inh, config::REC_PROB, 0.5f, 0.1f);
    m_synapses["hidden_inh_to_hidden_exc"] = std::make_unique<Synapse>(m_num_hidden_inh, m_num_hidden_exc, config::REC_PROB, 0.5f, 0.1f);
    m_synapses["recurrent_hidden_exc"] = std::make_unique<Synapse>(m_num_hidden_exc, m_num_hidden_exc, config::REC_PROB, 0.5f, 0.1f);

    // Instantiate memory systems
    m_hippocampus = std::make_unique<HippocampalModule>(m_hdc);
    m_neocortex = std::make_unique<NeocorticalModule>(m_hdc);

    // Allocate GPU memory for intermediate current buffers
    CUDA_CHECK(cudaMalloc(&d_hidden_exc_current, m_num_hidden_exc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden_inh_current, m_num_hidden_inh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_current, config::NUM_OUTPUT_NEURONS * sizeof(float)));
}

AGINetwork::~AGINetwork() {
    cudaFree(d_hidden_exc_current);
    cudaFree(d_hidden_inh_current);
    cudaFree(d_output_current);
}

void AGINetwork::forward(const Eigen::VectorXf& input_current, float current_time_ms, bool stdp_enabled) {
    // 1. Update input layer
    m_input_neurons->update(input_current, current_time_ms);

    // 2. Propagate to hidden excitatory neurons
    // Current = Feedforward from input + Recurrent from other excitatory neurons - Inhibition
    m_synapses["input_to_hidden_exc"]->forward(m_input_neurons->getSpikesDevicePtr(), d_hidden_exc_current);
    // Note: Recurrent and inhibitory connections would be added here in a full pass.

    // 3. Update hidden layers
    Eigen::VectorXf hidden_exc_current_host(m_num_hidden_exc); // Simplified: should be on GPU
    cudaMemcpy(hidden_exc_current_host.data(), d_hidden_exc_current, m_num_hidden_exc * sizeof(float), cudaMemcpyDeviceToHost);
    m_exc_hidden_neurons->update(hidden_exc_current_host, current_time_ms);

    // ... propagate to inhibitory and then to output ...

    // 4. Update output layer
    m_synapses["hidden_exc_to_output"]->forward(m_exc_hidden_neurons->getSpikesDevicePtr(), d_output_current);
    Eigen::VectorXf output_current_host(config::NUM_OUTPUT_NEURONS);
    cudaMemcpy(output_current_host.data(), d_output_current, config::NUM_OUTPUT_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);
    m_output_neurons->update(output_current_host, current_time_ms);

    // 5. If learning is enabled, update all synapses with STDP
    if (stdp_enabled) {
        m_synapses["input_to_hidden_exc"]->updateSTDP(m_input_neurons->getLastSpikeTimesDevicePtr(), m_exc_hidden_neurons->getLastSpikeTimesDevicePtr(), m_neuromodulators);
        m_synapses["hidden_exc_to_output"]->updateSTDP(m_exc_hidden_neurons->getLastSpikeTimesDevicePtr(), m_output_neurons->getLastSpikeTimesDevicePtr(), m_neuromodulators);
        // ... update other synapses
    }

    // 6. Memory and neuromodulator updates
    m_neuromodulators.update();
    // ... hippocampus encoding and replay logic would be called from simulation_manager ...
}

void AGINetwork::reset() {
    m_input_neurons->reset();
    m_exc_hidden_neurons->reset();
    m_inh_hidden_neurons->reset();
    m_output_neurons->reset();
    // Re-initializing synapses is complex; a simpler reset might just reset neuron states.
}

Eigen::VectorXf AGINetwork::getSpikes(const std::string& layer_name) const {
    if (layer_name == "input") return m_input_neurons->getSpikesHost();
    if (layer_name == "hidden_exc") return m_exc_hidden_neurons->getSpikesHost();
    if (layer_name == "hidden_inh") return m_inh_hidden_neurons->getSpikesHost();
    if (layer_name == "output") return m_output_neurons->getSpikesHost();
    return Eigen::VectorXf();
}

std::map<std::string, Eigen::MatrixXf> AGINetwork::getAllSynapticWeights() const {
    std::map<std::string, Eigen::MatrixXf> weights;
    for (const auto& pair : m_synapses) {
        weights[pair.first] = pair.second->getWeights();
    }
    return weights;
}

void AGINetwork::setAllSynapticWeights(const std::map<std::string, Eigen::MatrixXf>& weights) {
    for (const auto& pair : weights) {
        if (m_synapses.count(pair.first)) {
            m_synapses.at(pair.first)->setWeights(pair.second);
        }
    }
}

void AGINetwork::applyDevelopmentalBias(const Eigen::VectorXf& avg_input, const Eigen::VectorXf& avg_hidden) {
    m_synapses.at("input_to_hidden_exc")->initializeDevelopmentalBias(avg_input, avg_hidden);
}