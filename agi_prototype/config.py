# agi_prototype/config.py

import torch

# --- Global Simulation Parameters ---
DT = 1.0  # Simulation time step in milliseconds (ms)
SIMULATION_DURATION = 60000  # Total simulation time in ms (e.g., 60000ms = 60 seconds)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
REFRESH_RATE_MS = 50

# --- Visualization Parameters ---
RASTER_WINDOW_SIZE_MS = 2000
NUM_EXAMPLE_NEURONS_TO_PLOT = 5

# --- Neuron Parameters (Leaky Integrate-and-Fire - LIF) ---
LIF_PARAMS = {
    'v_rest': -65.0,
    'v_reset': -70.0,
    'v_thresh': -50.0,
    'tau_m': 10.0,
    'tau_ref': 2.0,
    'adaptation_rate': 0.01,
    'min_thresh': -52.0,
    'max_thresh': -48.0,
}
# --- FIX: Increased scale to ensure a strong input signal that propagates through all layers ---
INPUT_CURRENT_SCALE = 2.0 

# --- Synapse Parameters (STDP) ---
STDP_PARAMS = {
    'A_plus': 0.015, # Increased LTP for faster learning on task
    'A_minus': 0.017,
    'tau_plus': 20.0,
    'tau_minus': 20.0,
    'w_min': 0.0,
    'w_max': 1.0,
    'decay_rate': 0.0001,
}

# --- Neuromodulation Parameters ---
DOPAMINE_MOD = {
    'baseline_dopamine': 0.1,
    'decay_tau': 300.0,
    'reward_boost': 0.5, # Explicit boost for correct task performance
}
ACETYLCHOLINE_MOD = {
    'decay_tau': 400.0,
    'uncertainty_gain': 0.9,
}
SEROTONIN_MOD = {
    'decay_tau': 700.0,
}

# --- Task Parameters ---
PATTERN_A_INDICES = list(range(100, 200))
PATTERN_B_INDICES = list(range(500, 600))
TARGET_NEURON_A = 2
TARGET_NEURON_B = 7
PATTERN_PRESENTATION_MS = 300 # How long each pattern is shown

# --- Hyperdimensional Computing (HDC) Parameters ---
HD_DIM = 10000
# --- NEW: Parameters for contextual binding using permutation ---
CONTEXT_PERMUTATION_SHIFTS = {
    'time': 1,
    'reward': 2,
    # Add other context types and their unique shift values here
}


# --- Network Topology Parameters ---
NUM_INPUT_NEURONS = 784
NUM_HIDDEN_NEURONS = 400
NUM_OUTPUT_NEURONS = 10
EXC_INH_RATIO = 0.8
REC_PROB = 0.05
FF_PROB = 0.1

# --- Memory System Parameters ---
HIPPOCAMPUS_CAPACITY = 1000
NEOCORTEX_CONSOLIDATION_THRESHOLD = 0.5
REPLAY_BATCH_SIZE = 10
RECENCY_DECAY = 0.998
RECENCY_WEIGHT = 0.3
REWARD_WEIGHT = 0.5
NOVELTY_WEIGHT = 0.2
SEROTONIN_INFLUENCE = 0.5

# --- Developmental Phase ---
DEVELOPMENT_PHASE_DURATION_MS = 500
DEVELOPMENTAL_BIAS_STRENGTH = 0.05

