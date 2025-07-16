#include "simulation_manager.h"
#include "main_window.h" 
#include "config.h"
#include <iostream>
#include <QFileDialog>
#include <QMessageBox>

// --- SimulationManager (GUI Thread) Implementation ---

SimulationManager::SimulationManager(QObject *parent) : QObject(parent) {
    m_worker = new SimulationWorker;
    m_worker->moveToThread(&m_worker_thread);

    // Connect signals and slots for thread communication
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
        // This logic is simplified here but would mirror the Python implementation.
        // It would get output spikes, check for correctness, and pulse neuromodulators.

        m_current_time_ms += config::DT;

        // Update visualization data under a lock
        {
            std::lock_guard<std::mutex> lock(m_data_mutex);
            m_viz_data.currentTime = m_current_time_ms;
            // ... populate other m_viz_data fields ...
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
    // Implementation would mirror the Python version
    std::cout << "Developmental phase complete." << std::endl;
}

Eigen::VectorXf SimulationWorker::generateTaskInput() {
    // Implementation would mirror the Python version
    Eigen::VectorXf input_signal = Eigen::VectorXf::Zero(config::NUM_INPUT_NEURONS);
    return input_signal;
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
    // Implementation would serialize network and memory states to a file.
}

void SimulationWorker::loadState(const QString& filePath) {
    std::cout << "Worker loading state from: " << filePath.toStdString() << std::endl;
    // Implementation would deserialize state and load it into the network.
}