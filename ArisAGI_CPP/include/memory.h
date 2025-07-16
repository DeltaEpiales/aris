#pragma once
#include <deque>
#include <vector>
#include <string>
#include <map>
#include "hdc.h"
#include "neuromodulation.h"

// Represents a single, context-rich memory trace.
struct Episode {
    Vector hv;       // The final, composite hypervector (pattern bound with context)
    Vector core_hv;  // The hypervector for the raw pattern, without context
    float reward;    // The scalar reward associated with this episode
    float recency;   // A decaying value representing the age of the memory
};

class NeocorticalModule; // Forward declaration for circular dependency

/**
 * @class HippocampalModule
 * @brief Simulates a fast-learning system for episodic memory.
 * It captures specific events in their full context and prioritizes them for replay.
 */
class HippocampalModule {
public:
    explicit HippocampalModule(HDC& hdc);

    /**
     * @brief The core memory encoding function. Implements "quantum-inspired" contextual binding.
     * @param spike_pattern The raw spike pattern from the SNN.
     * @param context_data A map containing contextual information like time and reward.
     */
    void encodePattern(const Eigen::VectorXf& spike_pattern, const std::map<std::string, float>& context_data);

    void triggerReplay(NeocorticalModule& neocortex, const NeuromodulatorSystem& neuromodulators);
    void reset();

    // --- For State Saving/Loading ---
    const std::deque<Episode>& getMemory() const { return m_episodic_memory; }
    void setMemory(const std::deque<Episode>& memory) { m_episodic_memory = memory; }

private:
    HDC& m_hdc;
    std::deque<Episode> m_episodic_memory;

    Vector encodeSpikePatternToHV(const Eigen::VectorXf& spike_pattern);
    std::vector<float> calculatePriorities(NeocorticalModule& neocortex, const NeuromodulatorSystem& neuromodulators);
};

/**
 * @class NeocorticalModule
 * @brief Simulates a slow-learning system for semantic memory.
 * It generalizes knowledge by bundling similar episodic memories replayed from the hippocampus.
 */
class NeocorticalModule {
public:
    explicit NeocorticalModule(HDC& hdc);

    void consolidatePattern(const Vector& replayed_hv);
    float getSimilarityToMemory(const Vector& query_hv);
    void reset();

    // --- For State Saving/Loading ---
    const std::vector<Vector>& getMemoryBank() const { return m_semantic_memory_bank; }
    void setMemoryBank(const std::vector<Vector>& bank) { m_semantic_memory_bank = bank; }

private:
    HDC& m_hdc;
    std::vector<Vector> m_semantic_memory_bank;
};