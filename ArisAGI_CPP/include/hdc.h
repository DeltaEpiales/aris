#pragma once
#include <Eigen/Dense>
#include <unordered_map>
#include <string>
#include <vector>

// Type aliases for clarity
using Vector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;

/**
 * @class HDC
 * @brief Implements the core algebra for Hyperdimensional Computing.
 * This class provides the fundamental tools (binding, bundling, permutation)
 * for representing and manipulating concepts as high-dimensional vectors.
 */
class HDC {
public:
    explicit HDC(int dim);

    // --- Core HDC Operations ---
    Vector bind(const Vector& v1, const Vector& v2);
    Vector bundle(const std::vector<Vector>& vecs);
    Vector permute(const Vector& v, int shift = 1);
    float cosineSimilarity(const Vector& v1, const Vector& v2);

    // --- Memory and Encoding ---
    void initBasisVectors(int num_entities, const std::string& prefix);
    Vector getBasisVector(const std::string& key);

private:
    int m_dim;
    // The "Item Memory" that stores the fundamental, atomic hypervectors.
    std::unordered_map<std::string, Vector> m_basis_vectors;

    Vector createRandomHypervector();
};