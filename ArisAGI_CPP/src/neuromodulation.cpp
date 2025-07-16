#include "neuromodulation.h"
#include "config.h"
#include <cmath>
#include <algorithm>

NeuromodulatorSystem::NeuromodulatorSystem() {
    reset();
    // Pre-calculate decay factors for efficiency.
    m_dopamine_decay = expf(-config::DT / config::DOPAMINE_DECAY_TAU);
    m_acetylcholine_decay = expf(-config::DT / config::ACETYLCHOLINE_DECAY_TAU);
    m_serotonin_decay = expf(-config::DT / config::SEROTONIN_DECAY_TAU);
}

void NeuromodulatorSystem::reset() {
    m_dopamine_level = config::DOPAMINE_BASELINE;
    m_acetylcholine_level = 0.0f;
    m_serotonin_level = 0.0f;
}

void NeuromodulatorSystem::update() {
    // Apply natural exponential decay to all levels each timestep.
    m_dopamine_level *= m_dopamine_decay;
    m_dopamine_level = std::max(m_dopamine_level, config::DOPAMINE_BASELINE);

    m_acetylcholine_level *= m_acetylcholine_decay;
    m_serotonin_level *= m_serotonin_decay;
}

void NeuromodulatorSystem::setDopaminePulse(float magnitude) {
    // A dopamine burst, e.g., from a reward signal.
    m_dopamine_level = std::min(1.0f, m_dopamine_level + magnitude);
}

void NeuromodulatorSystem::setAcetylcholinePulse(float magnitude) {
    // An acetylcholine burst, e.g., from an uncertainty/attention signal.
    m_acetylcholine_level = std::min(1.0f, m_acetylcholine_level + magnitude);
}

void NeuromodulatorSystem::setSerotoninPulse(float magnitude) {
    // A serotonin burst, e.g., from a cost or saliency signal.
    m_serotonin_level = std::min(1.0f, m_serotonin_level + magnitude);
}

std::map<std::string, float> NeuromodulatorSystem::getLevels() const {
    return {
        {"dopamine", m_dopamine_level},
        {"acetylcholine", m_acetylcholine_level},
        {"serotonin", m_serotonin_level}
    };
}

void NeuromodulatorSystem::loadLevels(const std::map<std::string, float>& levels) {
    if (levels.count("dopamine")) m_dopamine_level = levels.at("dopamine");
    if (levels.count("acetylcholine")) m_acetylcholine_level = levels.at("acetylcholine");
    if (levels.count("serotonin")) m_serotonin_level = levels.at("serotonin");
}