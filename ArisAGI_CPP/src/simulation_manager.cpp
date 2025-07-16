#include "simulation_manager.h"
#include "main_window.h" 
#include "config.h"
#include <iostream>
#include <QFileDialog>
#include <QMessageBox>
#include <chrono>
#include <fstream>

// --- SimulationManager (GUI Thread) Implementation ---

SimulationManager::SimulationManager(QObject *parent) : QObject(parent) {
    m_worker = new SimulationWorker;
    m_worker->moveToThread(&m_worker_thread);

    connect(&m_worker_thread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(this, &SimulationManager::startWorker, m_worker, &SimulationWorker::run);
    connect(this, &SimulationManager::stopWorker, m_worker, &SimulationWorker::stop, Qt::DirectConnection);
    connect(this, &SimulationManager::resetWorker, m_worker, &SimulationWorker::reset);
    connect(this, &SimulationManager::pulseDopamineWorker, m_worker, &SimulationWorker::manualDopaminePulse);
    connect(this, &SimulationManager::saveStateWorker, m_worker, &SimulationWorker::saveState);
    connect(this, &SimulationManager::loadStateWorker, m_worker, &SimulationWorker::loadState);

    m_worker_thread.start();
}

SimulationManager::~SimulationManager() {
    m_worker_thread.quit();
    m_worker_thread.wait();
}

void SimulationManager::setMainWindow(MainWindow* window) {
    m_main_window = window;
}

void SimulationManager::startSimulation() { emit startWorker(); }
void SimulationManager::stopSimulation() { emit stopWorker(); }
void SimulationManager::resetSimulation() { emit resetWorker(); }
void SimulationManager::manualDopaminePulse(int value) { emit pulseDopamineWorker(value / 100.0f); }

VizData SimulationManager::getVizData() {
    return m_worker->getLatestVizData();
}

void SimulationManager::saveState() {
    QString filePath = QFileDialog::getSaveFileName(m_main_window, "Save Model State", "", "Aris State Files (*.aris)");
    if (!filePath.isEmpty()) {
        emit saveStateWorker(filePath);
    }
}

void SimulationManager::loadState() {
    QString filePath = QFileDialog::getOpenFileName(m_main_window, "Load Model State", "", "Aris State Files (*.aris)");
    if (!filePath.isEmpty()) {
        emit loadStateWorker(filePath);
    }
}


// --- SimulationWorker (Background Thread) Implementation ---

SimulationWorker::SimulationWorker() : m_running(false) {
    m_hdc = std::make_unique<HDC>(config::HD_DIM);
    m_network = std::make_unique<AGINetwork>(*m_hdc);
    reset();
}
SimulationWorker::~SimulationWorker() = default;

void SimulationWorker::reset() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_network->reset();
    m_current_time_ms = 0.0f;
    m_current_pattern_id = 'A';
    m_pattern_timer = 0.0f;
    m_last_sparsity = 1.0f;
}

void SimulationWorker::run() {
    if (m_running) return;
    m_running = true;
    std::cout << "Simulation thread started." << std::endl;

    developmentalPhase();

    while(m_running) {
        auto loop_start_time = std::chrono::high_resolution_clock::now();

        Eigen::VectorXf input_current = generateTaskInput();
        m_network->forward(input_current, m_current_time_ms);

        // --- Task and Neuromodulation Logic ---
        float reward = 0.0f;
        float serotonin_pulse = 0.0f;

        Eigen::VectorXf output_spikes = m_network->getSpikes("output");
        Eigen::VectorXf hidden_spikes = m_network->getSpikes("hidden_exc");

        if (m_current_pattern_id == 'A' && output_spikes(config::TARGET_NEURON_A) > 0.5f) {
            reward = config::DOPAMINE_REWARD_BOOST;
            serotonin_pulse = 0.1f;
        } else if (m_current_pattern_id == 'B' && output_spikes(config::TARGET_NEURON_B) > 0.5f) {
            reward = config::DOPAMINE_REWARD_BOOST;
            serotonin_pulse = 0.1f;
        }

        float current_sparsity = calculateSparsity(hidden_spikes);
        float sparsity_change = std::abs(current_sparsity - m_last_sparsity);
        float epistemic_uncertainty = sparsity_change * 5.0f;
        m_last_sparsity = current_sparsity;

        m_network->getNeuromodulators().setDopaminePulse(reward);
        m_network->getNeuromodulators().setAcetylcholinePulse(epistemic_uncertainty * config::ACETYLCHOLINE_UNCERTAINTY_GAIN);
        m_network->getNeuromodulators().setSerotoninPulse(serotonin_pulse);

        m_current_time_ms += config::DT;

        // Update visualization data under a lock
        {
            std::lock_guard<std::mutex> lock(m_data_mutex);
            m_viz_data.currentTime = m_current_time_ms;
            m_viz_data.spikes["input"] = m_network->getSpikes("input");
            m_viz_data.spikes["hidden_exc"] = m_network->getSpikes("hidden_exc");
            m_viz_data.spikes["output"] = m_network->getSpikes("output");
            m_viz_data.neuromodulators = m_network->getNeuromodulators().getLevels();
        }
        emit dataReady();

        auto loop_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = loop_end_time - loop_start_time;
        QThread::msleep(std::max(0.0f, config::DT - elapsed.count()));
    }
    emit finished();
}

void SimulationWorker::stop() {
    m_running = false;
}

void SimulationWorker::developmentalPhase() {
    std::cout << "Starting developmental phase..." << std::endl;
    int num_steps = static_cast<int>(config::DEVELOPMENT_PHASE_DURATION_MS / config::DT);
    int num_hidden_exc = static_cast<int>(config::NUM_HIDDEN_NEURONS * config::EXC_INH_RATIO);
    Eigen::VectorXf avg_input = Eigen::VectorXf::Zero(config::NUM_INPUT_NEURONS);
    Eigen::VectorXf avg_hidden = Eigen::VectorXf::Zero(num_hidden_exc);

    for(int i = 0; i < num_steps; ++i) {
        Eigen::VectorXf input_current = Eigen::VectorXf::Random(config::NUM_INPUT_NEURONS).cwiseAbs() * config::INPUT_CURRENT_SCALE;
        m_network->forward(input_current, m_current_time_ms, false);
        avg_input += m_network->getSpikes("input");
        avg_hidden += m_network->getSpikes("hidden_exc");
        m_current_time_ms += config::DT;
    }
    avg_input /= num_steps;
    avg_hidden /= num_steps;
    m_network->applyDevelopmentalBias(avg_input, avg_hidden);
    std::cout << "Developmental phase complete." << std::endl;
}

Eigen::VectorXf SimulationWorker::generateTaskInput() {
    m_pattern_timer += config::DT;
    if (m_pattern_timer > config::PATTERN_PRESENTATION_MS) {
        m_pattern_timer = 0.0f;
        m_current_pattern_id = (m_current_pattern_id == 'A') ? 'B' : 'A';
    }

    Eigen::VectorXf input_signal = Eigen::VectorXf::Zero(config::NUM_INPUT_NEURONS);
    const auto& indices = (m_current_pattern_id == 'A') ? config::PATTERN_A_INDICES : config::PATTERN_B_INDICES;
    for (int idx : indices) {
        input_signal(idx) = 1.0f;
    }

    input_signal += Eigen::VectorXf::Random(config::NUM_INPUT_NEURONS).cwiseAbs() * 0.1f;
    return input_signal * config::INPUT_CURRENT_SCALE;
}

float SimulationWorker::calculateSparsity(const Eigen::VectorXf& spikes) {
    if (spikes.size() == 0) return 1.0f;
    return 1.0f - (spikes.sum() / spikes.size());
}

VizData SimulationWorker::getLatestVizData() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_viz_data;
}

void SimulationWorker::manualDopaminePulse(float magnitude) {
    m_network->getNeuromodulators().setDopaminePulse(magnitude);
}

void SimulationWorker::saveState(const QString& filePath) {
    std::cout << "Worker saving state to: " << filePath.toStdString() << std::endl;
    // This is a placeholder for a proper serialization library (e.g., Boost.Serialization, Cereal)
    std::ofstream ofs(filePath.toStdString(), std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << filePath.toStdString() << std::endl;
        return;
    }
    // ... serialization logic ...
}

void SimulationWorker::loadState(const QString& filePath) {
    std::cout << "Worker loading state from: " << filePath.toStdString() << std::endl;
    // This is a placeholder for a proper serialization library
    std::ifstream ifs(filePath.toStdString(), std::ios::binary);
    if (!ifs) {
        std::cerr << "Error opening file for reading: " << filePath.toStdString() << std::endl;
        return;
    }
    // ... deserialization logic ...
}