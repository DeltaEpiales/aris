# ==========================================================================
# CMakeLists.txt
# This file manages the entire build process for the Aris C++ prototype.
# It finds all necessary dependencies (CUDA, Qt6, Eigen) and compiles
# the C++, CUDA, and Qt-specific files into a single executable.
# ==========================================================================

cmake_minimum_required(VERSION 3.18)
project(ArisAGI LANGUAGES CXX CUDA)

# --- Find Dependencies ---

# Find the NVIDIA CUDA toolkit. This is essential for our GPU kernels.
find_package(CUDA REQUIRED)

# Find the Qt6 framework for the GUI.
# Note: The user must have Qt6 installed and CMAKE_PREFIX_PATH configured.
# Example for user: export CMAKE_PREFIX_PATH=/path/to/Qt/6.x.x/gcc_64/
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

# Find the Eigen3 library for high-performance linear algebra on the CPU.
find_package(Eigen3 REQUIRED)


# --- Project Configuration ---

# Add our own 'include' directory and Eigen's include directory to the project.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Enforce modern C++ and CUDA standards.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# These settings are crucial for Qt projects. They automate the handling of
# Qt's Meta-Object Compiler (MOC), User Interface Compiler (UIC), and
# Resource Compiler (RCC), which are needed for signals, slots, and UI files.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)


# --- Define Source Files ---

# List all C++ source files that need to be compiled.
set(SOURCES
    src/main.cpp
    src/simulation_manager.cpp
    src/agn_network.cpp
    src/lif_neuron.cpp
    src/synapse.cpp
    src/hdc.cpp
    src/memory.cpp
    src/neuromodulation.cpp
    src/main_window.cpp
)

# List all CUDA source files (.cu) that need to be compiled by NVCC.
set(CUDA_SOURCES
    src/kernels.cu
)


# --- Create the Executable ---

# This command creates the final executable named 'ArisAGI' from all our source files.
add_executable(ArisAGI ${SOURCES} ${CUDA_SOURCES})

# Link all the necessary libraries to our executable.
target_link_libraries(ArisAGI
    PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Eigen3::Eigen
    # Note: CUDA libraries are linked automatically by CMake when using `project(... LANGUAGES CUDA)`
)

# --- Installation (Optional) ---
# This defines a simple 'make install' rule to copy the executable to a 'bin' directory.
install(TARGETS ArisAGI DESTINATION bin)