#pragma once
#include <deque>
#include <vector>
#include <string>
#include <map>
#include "hdc.h"
#include "neuromodulation.h"

struct Episode {
    Vector hv;
    Vector core_hv;
    float reward;
    float recency;
};

class NeocorticalModule;

class HippocampalModule {
public:
    explicit HippocampalModule(HDC& hdc);

    void encodePattern(const Eigen::VectorXf& spike_pattern, const std::map<std::string, float>& context_data);
    void triggerReplay(NeocorticalModule& neocortex, const NeuromodulatorSystem& neuromodulators);
    void reset();

    const std::deque<Episode>& getMemory() const { return m_episodic_memory; }
    void setMemory(const std::deque<Episode>& memory) { m_episodic_memory = memory; }

private:
    HDC& m_hdc;
    std::deque<Episode> m_episodic_memory;
    Vector encodeSpikePatternToHV(const Eigen::VectorXf& spike_pattern);
};

class NeocorticalModule {
public:
    explicit NeocorticalModule(HDC& hdc);

    void consolidatePattern(const Vector& replayed_hv);
    float getSimilarityToMemory(const Vector& query_hv);
    void reset();

    const std::vector<Vector>& getMemoryBank() const { return m_semantic_memory_bank; }
    void setMemoryBank(const std::vector<Vector>& bank) { m_semantic_memory_bank = bank; }

private:
    HDC& m_hdc;
    std::vector<Vector> m_semantic_memory_bank;
};