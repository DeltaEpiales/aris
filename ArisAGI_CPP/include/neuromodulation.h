#pragma once
#include <map>
#include <string>

class NeuromodulatorSystem {
public:
    NeuromodulatorSystem();

    void update();
    void setDopaminePulse(float magnitude);
    void setAcetylcholinePulse(float magnitude);
    void setSerotoninPulse(float magnitude);

    float getDopamineLevel() const { return m_dopamine_level; }
    float getAcetylcholineLevel() const { return m_acetylcholine_level; }
    float getSerotoninLevel() const { return m_serotonin_level; }

    std::map<std::string, float> getLevels() const;
    void loadLevels(const std::map<std::string, float>& levels);
    void reset();

private:
    float m_dopamine_level;
    float m_acetylcholine_level;
    float m_serotonin_level;

    float m_dopamine_decay;
    float m_acetylcholine_decay;
    float m_serotonin_decay;
};