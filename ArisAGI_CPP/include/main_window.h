#pragma once
#include <QMainWindow>
#include <memory>

// Forward declarations
class QTabWidget;
class QPushButton;
class QSlider;
class QLabel;
class QTimer;
class SimulationManager;
class QCustomPlot; // Placeholder for a real plotting widget

/**
 * @class MainWindow
 * @brief The main graphical user interface for the Aris prototype.
 * It provides controls to run the simulation and will display the visualization.
 */
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void updateGUI();

private:
    void setupUI();
    void setupControlTab();

    std::unique_ptr<SimulationManager> m_sim_manager;
    QTimer* m_update_timer;

    QTabWidget* m_tabs;
    QWidget* m_viz_tab;
    QWidget* m_control_tab;

    // Simulation controls
    QPushButton* m_start_button;
    QPushButton* m_stop_button;
    QPushButton* m_reset_button;

    // System controls
    QPushButton* m_save_button;
    QPushButton* m_load_button;
    QPushButton* m_pulse_button;
    QSlider* m_dopamine_slider;
    QLabel* m_dopamine_value_label;

    // For now, we use labels as placeholders for the complex plots.
    // A full implementation would use a library like QCustomPlot or integrate Matplotlib.
    QLabel* m_viz_placeholder;
};