#include "main_window.h"
#include <QApplication>
#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    // It's good practice to check for a CUDA-capable device at startup.
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "CUDA Error: cudaGetDeviceCount returned " << static_cast<int>(error_id) 
                  << " -> " << cudaGetErrorString(error_id) << std::endl;
        std::cerr << "Please ensure your NVIDIA drivers and CUDA toolkit are installed correctly." << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-enabled devices were found on your system." << std::endl;
        return 1;
    }

    std::cout << "CUDA is available. Found " << deviceCount << " device(s)." << std::endl;
    cudaSetDevice(0); // Explicitly use the first GPU.

    // Standard Qt application setup.
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}