#pragma once

#include <QObject>
#include <QThread>
#include <memory>
#include <mutex>
#include <atomic>
#include <Eigen/Dense>
#include <map>
#include <string>

// Forward declarations
class AGINetwork;
class HDC;
class MainWindow;

// Data structure to pass visualization data from worker to GUI thread safely.
struct VizData {
    std::map<std::string, Eigen::MatrixXf> weights;
    std::map<std::string, Eigen::VectorXf> spikes;
    std::map<std::string, Eigen::VectorXf> potentials;
    std::map<std::string, float> neuromodulators;
    float currentTime;
};

/**
 * @class SimulationWorker
 * @brief Runs the entire simulation loop in a separate thread to keep the GUI responsive.
 * This object lives on a different thread than the main GUI.
 */
class SimulationWorker : public QObject {
    Q_OBJECT
public:
    SimulationWorker();
    ~SimulationWorker();
    VizData getLatestVizData();

public slots:
    void run();
    void stop();
    void reset();
    void manualDopaminePulse(float magnitude);
    void saveState(const QString& filePath);
    void loadState(const QString& filePath);

signals:
    void finished();
    void dataReady();

private:
    std::atomic<bool> m_running;
    std::mutex m_data_mutex;
    std::unique_ptr<HDC> m_hdc;
    std::unique_ptr<AGINetwork> m_network;

    float m_current_time_ms;
    char m_current_pattern_id;
    float m_pattern_timer;
    float m_last_sparsity;
    VizData m_viz_data;

    void developmentalPhase();
    Eigen::VectorXf generateTaskInput();
    float calculateSparsity(const Eigen::VectorXf& spikes);
};

/**
 * @class SimulationManager
 * @brief The main controller that lives in the GUI thread.
 * It manages the SimulationWorker and its thread, and acts as the interface
 * between the GUI and the simulation backend.
 */
class SimulationManager : public QObject {
    Q_OBJECT
public:
    explicit SimulationManager(QObject *parent = nullptr);
    ~SimulationManager();

    void setMainWindow(MainWindow* window);
    VizData getVizData();

public slots:
    void startSimulation();
    void stopSimulation();
    void resetSimulation();
    void manualDopaminePulse(int value);
    void saveState();
    void loadState();

signals:
    void startWorker();
    void stopWorker();
    void resetWorker();
    void pulseDopamineWorker(float magnitude);
    void saveStateWorker(const QString& filePath);
    void loadStateWorker(const QString& filePath);

private:
    QThread m_worker_thread;
    SimulationWorker* m_worker;
    MainWindow* m_main_window = nullptr;
};