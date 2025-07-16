#include "main_window.h"
#include "simulation_manager.h"
#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QPushButton>
#include <QSlider>
#include <QLabel>
#include <QTimer>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    m_sim_manager = std::make_unique<SimulationManager>(this);
    m_sim_manager->setMainWindow(this);
    setupUI();

    m_update_timer = new QTimer(this);
    connect(m_update_timer, &QTimer::timeout, this, &MainWindow::updateGUI);
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUI() {
    setWindowTitle("Aris AGI C++ Prototype");
    setMinimumSize(1200, 800);

    auto* central_widget = new QWidget;
    auto* main_layout = new QHBoxLayout(central_widget);

    m_tabs = new QTabWidget;
    m_viz_tab = new QWidget;
    m_control_tab = new QWidget;

    m_tabs->addTab(m_viz_tab, "Live Visualization");
    m_tabs->addTab(m_control_tab, "System Control");

    // Setup viz tab
    auto* viz_layout = new QVBoxLayout(m_viz_tab);
    m_viz_placeholder = new QLabel("Visualization will appear here.\n(Requires a plotting library like QCustomPlot)");
    m_viz_placeholder->setAlignment(Qt::AlignCenter);
    m_viz_placeholder->setStyleSheet("QLabel { background-color : #333; color : white; font-size: 18px; }");
    viz_layout->addWidget(m_viz_placeholder);

    // Setup control tab
    setupControlTab();

    main_layout->addWidget(m_tabs, 3);

    // --- Right Panel ---
    auto* right_panel_layout = new QVBoxLayout;

    auto* main_controls_group = new QGroupBox("Simulation Control");
    auto* main_controls_layout = new QVBoxLayout;

    m_start_button = new QPushButton("Start Simulation");
    m_stop_button = new QPushButton("Stop Simulation");
    m_reset_button = new QPushButton("Reset Simulation");

    main_controls_layout->addWidget(m_start_button);
    main_controls_layout->addWidget(m_stop_button);
    main_controls_layout->addWidget(m_reset_button);
    main_controls_group->setLayout(main_controls_layout);

    right_panel_layout->addWidget(main_controls_group);
    right_panel_layout->addStretch();

    main_layout->addLayout(right_panel_layout, 1);

    setCentralWidget(central_widget);

    // --- Connections ---
    connect(m_start_button, &QPushButton::clicked, [this](){
        m_sim_manager->startSimulation();
        m_update_timer->start(config::REFRESH_RATE_MS);
    });
    connect(m_stop_button, &QPushButton::clicked, [this](){
        m_sim_manager->stopSimulation();
        m_update_timer->stop();
    });
    connect(m_reset_button, &QPushButton::clicked, m_sim_manager.get(), &SimulationManager::resetSimulation);
}

void MainWindow::setupControlTab() {
    auto* control_layout = new QVBoxLayout(m_control_tab);

    auto* state_group = new QGroupBox("Model Checkpointing");
    auto* state_layout = new QHBoxLayout;
    m_save_button = new QPushButton("Save State");
    m_load_button = new QPushButton("Load State");
    state_layout->addWidget(m_save_button);
    state_layout->addWidget(m_load_button);
    state_group->setLayout(state_layout);
    control_layout->addWidget(state_group);

    auto* neuro_group = new QGroupBox("Interactive Neuromodulation");
    auto* neuro_layout = new QFormLayout;

    m_dopamine_slider = new QSlider(Qt::Horizontal);
    m_dopamine_slider->setRange(0, 100);
    auto* dopamine_hbox = new QHBoxLayout;
    m_dopamine_value_label = new QLabel("0.00");
    dopamine_hbox->addWidget(m_dopamine_slider);
    dopamine_hbox->addWidget(m_dopamine_value_label);

    m_pulse_button = new QPushButton("Apply Pulse");

    neuro_layout->addRow(new QLabel("Manual Dopamine Pulse:"), dopamine_hbox);
    neuro_layout->addRow(m_pulse_button);
    neuro_group->setLayout(neuro_layout);
    control_layout->addWidget(neuro_group);

    control_layout->addStretch();

    // --- Connections for Control Tab ---
    connect(m_save_button, &QPushButton::clicked, m_sim_manager.get(), &SimulationManager::saveState);
    connect(m_load_button, &QPushButton::clicked, m_sim_manager.get(), &SimulationManager::loadState);
    connect(m_pulse_button, &QPushButton::clicked, [this](){
        m_sim_manager->manualDopaminePulse(m_dopamine_slider->value());
    });
    connect(m_dopamine_slider, &QSlider::valueChanged, [this](int value){
        m_dopamine_value_label->setText(QString::number(value / 100.0, 'f', 2));
    });
}

void MainWindow::updateGUI() {
    // This is where you would get data from the simulation manager
    // and update the plotting widgets.
    // VizData data = m_sim_manager->getVizData();
    // m_my_plot_widget->update(data);
}