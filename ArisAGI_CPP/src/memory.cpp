#include "memory.h"
#include "config.h"
#include <numeric>
#include <algorithm>
#include <random>

// --- HippocampalModule Implementation ---
HippocampalModule::HippocampalModule(HDC& hdc) : m_hdc(hdc) {
    // Pre-generate the fundamental "role" and "value" basis vectors needed for contextual binding.
    m_hdc.initBasisVectors(2, "role"); // For time, reward
    m_hdc.initBasisvectors(config::NUM_HIDDEN_NEURONS, "neuron");
    m_hdc.initBasisVectors(1, "time_value_base");
    m_hdc.initBasisVectors(1, "reward_value_base");
}

Vector HippocampalModule::encodeSpikePatternToHV(const Eigen::VectorXf& spike_pattern) {
    // Convert a raw SNN spike pattern into a single hypervector.
    std::vector<Vector> active_hvs;
    for (int i = 0; i < spike_pattern.size(); ++i) {
        if (spike_pattern(i) > 0.5f) { // If neuron `i` spiked...
            // ...add its unique ID hypervector to the list.
            active_hvs.push_back(m_hdc.getBasisVector("neuron_" + std::to_string(i)));
        }
    }
    // Bundle the ID vectors of all spiking neurons into one composite vector.
    return m_hdc.bundle(active_hvs);
}

void HippocampalModule::encodePattern(const Eigen::VectorXf& spike_pattern, const std::map<std::string, float>& context_data) {
    // This is the "quantum-inspired" contextual binding process.
    Vector core_pattern_hv = encodeSpikePatternToHV(spike_pattern);
    if (core_pattern_hv.isZero()) return; // Don't store empty patterns.

    std::vector<Vector> context_chunks;

    // 1. Create a "Time" context chunk: bind(ROLE_time, VALUE_time)
    if (context_data.count("time_val")) {
        Vector role_time_hv = m_hdc.getBasisVector("role_0");
        Vector time_value_base_hv = m_hdc.getBasisVector("time_value_base_0");
        Vector value_time_hv = m_hdc.permute(time_value_base_hv, static_cast<int>(context_data.at("time_val")));
        context_chunks.push_back(m_hdc.bind(role_time_hv, value_time_hv));
    }
    // 2. Create a "Reward" context chunk: bind(ROLE_reward, VALUE_reward)
    if (context_data.count("reward_val")) {
        Vector role_reward_hv = m_hdc.getBasisVector("role_1");
        Vector reward_value_base_hv = m_hdc.getBasisVector("reward_value_base_0");
        int reward_shift = static_cast<int>(context_data.at("reward_val") * 10);
        Vector value_reward_hv = m_hdc.permute(reward_value_base_hv, reward_shift);
        context_chunks.push_back(m_hdc.bind(role_reward_hv, value_reward_hv));
    }

    // 3. Bind the core pattern with the bundled context.
    Vector episodic_hv = core_pattern_hv;
    if (!context_chunks.empty()) {
        Vector composite_context_hv = m_hdc.bundle(context_chunks);
        episodic_hv = m_hdc.bind(core_pattern_hv, composite_context_hv);
    }

    // 4. Store the complete episode in the memory buffer.
    if (m_episodic_memory.size() >= config::HIPPOCAMPUS_CAPACITY) {
        m_episodic_memory.pop_front();
    }
    m_episodic_memory.push_back({episodic_hv, core_pattern_hv, context_data.count("reward_val") ? context_data.at("reward_val") : 0.1f, 1.0f});
}

void HippocampalModule::reset() { m_episodic_memory.clear(); }

// --- NeocorticalModule Implementation ---
NeocorticalModule::NeocorticalModule(HDC& hdc) : m_hdc(hdc) {}

void NeocorticalModule::consolidatePattern(const Vector& replayed_hv) {
    // This is the core of generalization and long-term learning.
    if (m_semantic_memory_bank.empty()) {
        m_semantic_memory_bank.push_back(replayed_hv);
        return;
    }

    float max_sim = -1.0f;
    int best_match_idx = -1;
    for (size_t i = 0; i < m_semantic_memory_bank.size(); ++i) {
        float sim = m_hdc.cosineSimilarity(replayed_hv, m_semantic_memory_bank[i]);
        if (sim > max_sim) {
            max_sim = sim;
            best_match_idx = i;
        }
    }

    // If the replayed memory is similar enough to an existing concept...
    if (max_sim >= config::NEOCORTEX_CONSOLIDATION_THRESHOLD) {
        // ...bundle it with the existing concept to refine and generalize it.
        m_semantic_memory_bank[best_match_idx] = m_hdc.bundle({m_semantic_memory_bank[best_match_idx], replayed_hv});
    } else {
        // Otherwise, it's a new concept, so add it to the memory bank.
        m_semantic_memory_bank.push_back(replayed_hv);
    }
}

float NeocorticalModule::getSimilarityToMemory(const Vector& query_hv) {
    if (m_semantic_memory_bank.empty()) return 0.0f;
    float max_sim = 0.0f;
    for (const auto& mem_hv : m_semantic_memory_bank) {
        max_sim = std::max(max_sim, m_hdc.cosineSimilarity(query_hv, mem_hv));
    }
    return max_sim;
}

void NeocorticalModule::reset() { m_semantic_memory_bank.clear(); }

// --- HippocampalModule Replay Logic ---
void HippocampalModule::triggerReplay(NeocorticalModule& neocortex, const NeuromodulatorSystem& neuromodulators) {
    // This method would implement the prioritized experience replay logic.
    // For brevity, the full sampling logic is omitted, but it would calculate
    // priorities based on reward, recency, and novelty (similarity to neocortex)
    // and then call neocortex.consolidatePattern for the selected episodes.
}