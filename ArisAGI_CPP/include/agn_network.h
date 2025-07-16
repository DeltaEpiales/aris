#pragma once
#include <memory>
#include <map>
#include <string>
#include "lif_neuron.h"
#include "synapse.h"
#include "neuromodulation.h"
#include "memory.h"
#include "hdc.h"

/**
 * @class AGINetwork
 * @brief The central class that integrates all components of the Bio-Hyper-SNN.
 * It orchestrates the flow of information between neuron layers, synapses,
 * memory systems, and neuromodulators.
 */
class AGINetwork {
public:
    explicit AGINetwork(HDC& hdc);
    ~AGINetwork();

    AGINetwork(const AGINetwork&) = delete;
    AGINetwork& operator=(const AGINetwork&) = delete;

    /**
     * @brief Executes one full simulation step of the entire network.
     * @param input_current The sensory input for the current timestep.
     * @param current_time_ms The global simulation time.
     * @param stdp_enabled A flag to enable/disable learning (e.g., for developmental phase).
     */
    void forward(const Eigen::VectorXf& input_current, float current_time_ms, bool stdp_enabled = true);

    void applyDevelopmentalBias(const Eigen::VectorXf& avg_input, const Eigen::VectorXf& avg_hidden);
    void reset();

    // --- Getters for state and visualization ---
    Eigen::VectorXf getSpikes(const std::string& layer_name) const;
    std::map<std::string, Eigen::MatrixXf> getAllSynapticWeights() const;
    NeuromodulatorSystem& getNeuromodulators() { return m_neuromodulators; }
    HippocampalModule& getHippocampus() { return *m_hippocampus; }
    NeocorticalModule& getNeocortex() { return *m_neocortex; }

    // --- For saving/loading state ---
    void setAllSynapticWeights(const std::map<std::string, Eigen::MatrixXf>& weights);


private:
    HDC& m_hdc;
    int m_num_hidden_exc;
    int m_num_hidden_inh;

    // --- Core Components ---
    std::unique_ptr<LIFNeuron> m_input_neurons;
    std::unique_ptr<LIFNeuron> m_exc_hidden_neurons;
    std::unique_ptr<LIFNeuron> m_inh_hidden_neurons;
    std::unique_ptr<LIFNeuron> m_output_neurons;

    std::map<std::string, std::unique_ptr<Synapse>> m_synapses;

    NeuromodulatorSystem m_neuromodulators;

    std::unique_ptr<HippocampalModule> m_hippocampus;
    std::unique_ptr<NeocorticalModule> m_neocortex;

    // --- Intermediate buffers for computation ---
    float* d_hidden_exc_current;
    float* d_hidden_inh_current;
    float* d_output_current;
};