#pragma once
#include <map>
#include <string>

/**
 * @class NeuromodulatorSystem
 * @brief Simulates the levels of key "brain chemicals" that influence learning and behavior.
 * This class models the slow decay and rapid pulsing of neuromodulators like
 * Dopamine (reward), Acetylcholine (attention/uncertainty), and Serotonin (cost/saliency).
 */
class NeuromodulatorSystem {
public:
    NeuromodulatorSystem();

    // Called every simulation step to apply natural decay.
    void update();

    // --- Pulse Methods ---
    // Called by the simulation manager in response to specific events.
    void setDopaminePulse(float magnitude);
    void setAcetylcholinePulse(float magnitude);
    void setSerotoninPulse(float magnitude);

    // --- Getters ---
    float getDopamineLevel() const { return m_dopamine_level; }
    float getAcetylcholineLevel() const { return m_acetylcholine_level; }
    float getSerotoninLevel() const { return m_serotonin_level; }

    // For saving/loading state
    std::map<std::string, float> getLevels() const;
    void loadLevels(const std::map<std::string, float>& levels);
    void reset();

private:
    float m_dopamine_level;
    float m_acetylcholine_level;
    float m_serotonin_level;

    // Pre-calculated decay factors for efficiency
    float m_dopamine_decay;
    float m_acetylcholine_decay;
    float m_serotonin_decay;
};