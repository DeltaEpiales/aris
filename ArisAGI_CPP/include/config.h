#pragma once

#include <string>
#include <vector>
#include <numeric> // Required for std::iota

namespace config {

// --- Global Simulation Parameters ---
// These parameters define the temporal resolution and overall duration of the simulation.
constexpr float DT = 1.0f; // Simulation time step in milliseconds (ms)
constexpr float SIMULATION_DURATION = 60000.0f; // ms
constexpr int REFRESH_RATE_MS = 50; // How often the GUI updates.
constexpr int RANDOM_SEED = 42; // For reproducible results.

// --- Neuron Parameters (Leaky Integrate-and-Fire - LIF) ---
// These values define the core behavior of our spiking neurons, based on the LIF model.
constexpr float V_REST = -65.0f;    // Resting membrane potential (mV)
constexpr float V_RESET = -70.0f;   // Reset potential after a spike (mV)
constexpr float V_THRESH = -50.0f;  // Firing threshold (mV)
constexpr float TAU_M = 10.0f;      // Membrane time constant (ms), defines the "leakiness"
constexpr float TAU_REF = 2.0f;     // Refractory period (ms) after a spike
constexpr float ADAPTATION_RATE = 0.01f; // Rate for intrinsic threshold adaptation
constexpr float MIN_THRESH = -52.0f; // Floor for the dynamic threshold
constexpr float MAX_THRESH = -48.0f; // Ceiling for the dynamic threshold
constexpr float INPUT_CURRENT_SCALE = 2.0f; // Multiplier for input signal strength

// --- Synapse Parameters (STDP) ---
// These values control the learning rule for synapses.
constexpr float A_PLUS = 0.015f;    // Learning rate for Long-Term Potentiation (LTP)
constexpr float A_MINUS = 0.017f;   // Learning rate for Long-Term Depression (LTD)
constexpr float TAU_PLUS = 20.0f;   // Time constant for the LTP window (ms)
constexpr float TAU_MINUS = 20.0f;  // Time constant for the LTD window (ms)
constexpr float W_MIN = 0.0f;       // Minimum synaptic weight
constexpr float W_MAX = 1.0f;       // Maximum synaptic weight
constexpr float DECAY_RATE = 0.0001f; // "Use it or lose it" decay rate for structural plasticity

// --- Neuromodulation Parameters ---
// Defines the behavior of our simulated "brain chemicals".
constexpr float DOPAMINE_BASELINE = 0.1f;
constexpr float DOPAMINE_DECAY_TAU = 300.0f;
constexpr float DOPAMINE_REWARD_BOOST = 0.5f; // Dopamine pulse on correct action
constexpr float ACETYLCHOLINE_DECAY_TAU = 400.0f;
constexpr float ACETYLCHOLINE_UNCERTAINTY_GAIN = 0.9f; // How much uncertainty affects ACh
constexpr float SEROTONIN_DECAY_TAU = 700.0f;

// --- Task Parameters ---
// Defines the simple pattern recognition task for the agent.
const std::vector<int> PATTERN_A_INDICES = []{ std::vector<int> v(100); std::iota(v.begin(), v.end(), 100); return v; }();
const std::vector<int> PATTERN_B_INDICES = []{ std::vector<int> v(100); std::iota(v.begin(), v.end(), 500); return v; }();
constexpr int TARGET_NEURON_A = 2;
constexpr int TARGET_NEURON_B = 7;
constexpr float PATTERN_PRESENTATION_MS = 300.0f;

// --- Hyperdimensional Computing (HDC) Parameters ---
constexpr int HD_DIM = 10000; // Dimensionality of our hypervectors.

// --- Network Topology ---
// Defines the size and connectivity of the neural network.
constexpr int NUM_INPUT_NEURONS = 784;
constexpr int NUM_HIDDEN_NEURONS = 400;
constexpr int NUM_OUTPUT_NEURONS = 10;
constexpr float EXC_INH_RATIO = 0.8f; // 80% excitatory, 20% inhibitory
constexpr float REC_PROB = 0.05f;     // Recurrent connection probability
constexpr float FF_PROB = 0.1f;       // Feedforward connection probability

// --- Memory System Parameters ---
constexpr int HIPPOCAMPUS_CAPACITY = 1000; // Max number of episodic memories
constexpr float NEOCORTEX_CONSOLIDATION_THRESHOLD = 0.5f; // Similarity to merge memories
constexpr int REPLAY_BATCH_SIZE = 10;
constexpr float RECENCY_DECAY = 0.998f;
constexpr float RECENCY_WEIGHT = 0.3f;
constexpr float REWARD_WEIGHT = 0.5f;
constexpr float NOVELTY_WEIGHT = 0.2f;
constexpr float SEROTONIN_INFLUENCE = 0.5f;

// --- Developmental Phase ---
constexpr float DEVELOPMENT_PHASE_DURATION_MS = 500.0f;
constexpr float DEVELOPMENTAL_BIAS_STRENGTH = 0.05f;

} // namespace config