# Aris: A Lightweight Bio-Inspired AGI Prototype (Aristotle)

![Aris GUI Screenshot - Placeholder](Figure1.png)
*^ Placeholder image. Replace with a real screenshot of Aris running. ^*

##  Project Overview

**Aris** (short for **Aristotle**) is an ambitious open-source project aimed at exploring the foundational principles for building a **lightweight Artificial General Intelligence (AGI)** capable of running on conventional home computing hardware, specifically leveraging modern GPUs like the NVIDIA RTX 4070 Studio.

Unlike traditional large-scale deep learning models that rely on immense computational power and data, Aris draws inspiration directly from the incredible efficiency of biological brains. Our core philosophy is that by carefully designing the AI's architecture and learning mechanisms to mimic the brain's resourcefulness, we can create a system that inherently "figures out how to store information in a set amount of space" and operates with remarkable efficiency.

##  Motivation & Core Philosophy

The human brain operates on approximately 20 watts, yet performs feats of general intelligence unmatched by any current AI. This stark contrast highlights the potential for energy- and space-efficient AGI. Aris is built on the belief that by imposing constraints (like limited computational resources typical of a laptop), we can drive the development of inherently more efficient and brain-like intelligence.

Key principles guiding Aris:
* **Biological Plausibility:** Integrating insights from neuroscience regarding neuron behavior, synaptic plasticity, neuromodulation, and memory systems.
* **Energy & Space Efficiency:** Leveraging event-driven Spiking Neural Networks (SNNs) and compact Hyperdimensional Computing (HDC) representations.
* **Continuous & Lifelong Learning:** Designing mechanisms for ongoing adaptation without catastrophic forgetting, inspired by biological memory consolidation.
* **Emergent Intelligence:** Aiming for higher-level cognitive functions to emerge from the complex interactions of simpler, biologically-inspired components.

##  Key Features & Architectural Components

Aris is built around a **Bio-Hyper-SNN** architecture, a novel integration of Spiking Neural Networks with Hyperdimensional Computing.

### **2.1. Core SNN Modules**
* **Leaky Integrate-and-Fire (LIF) Neurons:** Efficiently simulates neuron membrane potential, firing threshold, and refractory periods. Includes basic intrinsic plasticity for dynamic thresholds.
* **Synaptic Connections (`Synapse` class):** Manages connections between neuron populations.
* **Spike-Timing-Dependent Plasticity (STDP):** A biologically plausible learning rule where synaptic strength changes based on the precise timing of pre- and post-synaptic spikes. Modulated by neuromodulator levels.
* **Sparse Probabilistic Connectivity:** Synaptic layers are initialized with a given connection probability, fostering a sparse network structure from the outset.

### **2.2. Hyperdimensional Computing (HDC)**
* **High-Dimensional Vector Space:** Information (concepts, patterns, contexts) is represented as robust, distributed hypervectors (e.g., 10,000 dimensions).
* **Core Operations:** Efficient binding, bundling (superposition), and permutation for symbolic manipulation and associative memory.
* **Basis Vectors:** Unique random hypervectors are pre-generated for each input neuron or conceptual entity, enabling SNN activity to be translated into HDC space.

### **2.3. Brain-Inspired Modularity**
* **Input Layer:** Processes external sensory signals into SNN spike patterns.
* **Hidden Layer (Cortical-like):** A large, recurrent SNN population divided into excitatory and inhibitory neurons, serving as a conceptual cortical processing unit.
* **Output Layer:** Generates responses or actions based on processed information.
* **Hippocampal Module:** Rapidly encodes and stores episodic memories.
* **Neocortical Module:** Performs slow, long-term memory consolidation and generalization.

### **2.4. Neuromodulatory System**
* Simulates the global influence of key neuromodulators:
    * **Dopamine:** Associated with reward and prediction surprise.
    * **Acetylcholine:** Associated with attention and uncertainty.
    * **Serotonin:** Associated with cost, exploration, and redundancy.
* These levels dynamically decay and are influenced by internal network states and conceptual "information signals."

##  Novel Algorithms & Innovations (Beyond the Standard)

Aris is not just a simulation; it's a testbed for novel, speculative algorithms pushing the boundaries of bio-inspired AI:

1.  **Quantum-Inspired Contextual Binding & Retrieval (Simplified):**
    * **Concept:** Moving beyond simple HDC binding, this aims for more dynamic, context-dependent memory formation. When an event is encoded, it's not just stored; its hypervector is "bound" with conceptual contextual hypervectors (like time, emotional state, location).
    * **Implementation:** In `HippocampalModule.encode_pattern`, core pattern HDVs are sequentially bound with context hypervectors (e.g., a time-tag HV generated by permutation). This aims to create richer, context-sensitive episodic memories.

2.  **Autoregressive Structural Plasticity (Simplified):**
    * **Concept:** The network's physical connectivity (its "structure") is not fixed but dynamically adapts based on activity and efficiency pressure.
    * **Implementation:** A `_run_developmental_phase` in `main.py` applies an *adaptive developmental bias* to initial synaptic weights based on early, noisy activity. This provides a "proto-structure" before formal learning. Simultaneously, a continuous `synaptic decay` in `Synapse.update_stdp` actively prunes less used connections, promoting an efficient, sparse network over time.

3.  **Information-Theoretic Neuromodulatory Control (Simplified):**
    * **Concept:** Neuromodulators are driven by internal "information signals" (not just external rewards) to optimize learning efficiency.
    * **Implementation:** In `AGINetwork.forward`, conceptual `prediction_surprise`, `epistemic_uncertainty`, and `info_redundancy` signals are generated (currently simplified) and directly influence Dopamine, Acetylcholine, and Serotonin levels. This aims to intrinsically guide the network to learn efficiently by prioritizing novel/uncertain information and discouraging redundancy.

4.  **Cost-Minimizing Replay Selection (New Algorithm):**
    * **Concept:** During "rest" or "sleep" phases, the system doesn't randomly replay memories. Instead, the `HippocampalModule.trigger_replay` prioritizes memories based on factors like recency, associated reward/surprise, and a conceptual "consolidation urgency" feedback signal from the `NeocorticalModule`. This ensures efficient, targeted memory consolidation.

5.  **HDC-based Memory Consolidation (New Application):**
    * **Concept:** The `NeocorticalModule` distills episodic memories into generalized, space-efficient semantic knowledge.
    * **Implementation:** When a replayed episodic hypervector is received, the `NeocorticalModule.consolidate_pattern` attempts to `hdc.bundle()` it with an existing similar semantic concept in its `semantic_memory_bank`. This combines multiple experiences into a single, more generalized HDC representation, effectively compressing information and demonstrating the brain's efficient knowledge organization.

## 🛠️ Installation

To set up and run Aris on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Aris.git](https://github.com/your-username/Aris.git)  # Replace with your actual repo URL
    cd Aris
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    * **PyTorch with CUDA:** This is crucial for leveraging your RTX 4070. Visit the official PyTorch installation page ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)), select your OS, `Pip`, and **your specific CUDA version (e.g., CUDA 12.1 for `cu121`)**. The command will look similar to this:
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```
        (Replace `cu121` with your chosen CUDA version).
    * **Other libraries:**
        ```bash
        pip install numpy matplotlib PyQt5
        ```

4.  **Verify CUDA Installation:**
    Run these Python commands in your activated environment:
    ```python
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    ```
    You should see `CUDA available: True` and your RTX 4070 listed. If not, revisit PyTorch installation and ensure your NVIDIA GPU drivers are up to date.

##  Usage

To start the Aris prototype and view its real-time visualization:

1.  **Navigate to the `agi_prototype` directory:**
    ```bash
    cd agi_prototype
    ```

2.  **Run the main application:**
    ```bash
    python main.py
    ```
    A GUI window titled "Bio-Hyper-SNN AGI Prototype" should appear.

3.  **Interact with the GUI:**
    * **"Start Simulation"**: Begins the SNN simulation loop. You should see spikes appear in the raster plots, membrane potentials fluctuate, weights change, and neuromodulator levels update.
    * **"Stop Simulation"**: Pauses the simulation.
    * **"Reset Simulation"**: Resets the entire network to its initial state (including rerunning the developmental phase).
    * **"Dopamine Pulse" Slider:** Conceptually injects a dopamine pulse, demonstrating real-time neuromodulatory control (will affect learning rates in the network).

## 📁 Project Structure

* `agi_prototype/`
    * `main.py`
    * `config.py`
    * `src/`
        * `neurons.py`
        * `synapses.py`
        * `network.py`
        * `hdc.py`
        * `neuromodulation.py`
        * `memory.py`
        * `data_encoders.py`
    * `utils/`
        * `visualization.py`
        * `metrics.py`
    * `ui/`
        * `gui.py`


## ⚠️ Current Status & Limitations

Aris is a **research prototype**. While it incorporates advanced conceptual algorithms and aims for efficiency, it is **not yet a true AGI**. Key limitations include:

* **Conceptual Algorithms:** Many "novel" algorithms (e.g., Information-Theoretic Neuromodulation, Quantum-Inspired Binding) are implemented conceptually, representing the *principle* rather than a fully solved, mathematically rigorous, or biologically exhaustive mechanism. They serve as a foundation for further research.
* **Scalability:** While designed for efficiency, simulating brain-scale networks (billions of neurons, trillions of synapses) is currently beyond consumer hardware. This prototype operates at a much smaller scale.
* **Learning Task Complexity:** The current input `_generate_example_input` is simplistic. Real AGI requires complex, multi-modal, open-ended learning environments.
* **Parameter Tuning:** The parameters in `config.py` are initial values and will require extensive tuning and experimentation to observe optimal or desired emergent behaviors.
* **Debugging:** Complex, dynamic SNNs are inherently challenging to debug.

## 📈 Future Work & How to Contribute

This project is a starting point for an exciting journey. We welcome contributions from researchers, developers, and enthusiasts!

Possible areas for future development:

* **Refine Core Algorithms:**
    * Develop more sophisticated `Prediction_surprise`, `Epistemic_uncertainty`, and `Info_redundancy` calculation methods based on network state and environmental feedback.
    * Implement more advanced forms of contextual binding in HDC.
    * Explore reinforcement learning loops for goal-directed behavior, driven by neuromodulators.
* **Memory System Enhancement:**
    * Integrate a truly trainable SNN within the Neocortical Module that learns from replayed HDVs.
    * Develop mechanisms for hierarchical memory organization and retrieval.
* **Sensory & Motor Integration:**
    * Connect to real-world sensory inputs (e.g., event-based cameras, microphones).
    * Develop motor control modules to enable embodied interaction.
* **Advanced Learning Tasks:** Implement complex, open-ended learning environments (e.g., mini-games, simulated robots) to test emergent AGI capabilities.
* **Performance Optimization:** Explore PyTorch's advanced features for sparse tensor operations, JIT compilation, and potentially porting critical sections to custom CUDA kernels for extreme efficiency.
* **Theoretical Development:** Further formalize the mathematical underpinnings of the conceptual algorithms.

If you're interested in contributing, please feel free to fork the repository, open issues, and submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Inspired by the intricate complexity and astounding efficiency of the biological brain, and guided by the principles of Spiking Neural Networks and Hyperdimensional Computing. Special thanks to the open-source community for foundational libraries like PyTorch, NumPy, Matplotlib, and PyQt5.
