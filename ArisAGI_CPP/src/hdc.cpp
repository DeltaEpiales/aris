#include "hdc.h"
#include <random>
#include "config.h"

HDC::HDC(int dim) : m_dim(dim) {}

Vector HDC::createRandomHypervector() {
    // Creates a single bipolar {-1, 1} hypervector.
    Vector hv(m_dim);
    // Use a static generator to ensure different vectors are created on subsequent calls.
    static std::mt19937 gen(config::RANDOM_SEED);
    std::uniform_int_distribution<> distrib(0, 1);
    for (int i = 0; i < m_dim; ++i) {
        hv(i) = distrib(gen) == 0 ? -1.0f : 1.0f;
    }
    return hv;
}

void HDC::initBasisVectors(int num_entities, const std::string& prefix) {
    // Pre-generates and stores unique hypervectors for fundamental concepts.
    for (int i = 0; i < num_entities; ++i) {
        std::string key = prefix + "_" + std::to_string(i);
        if (m_basis_vectors.find(key) == m_basis_vectors.end()) {
            m_basis_vectors[key] = createRandomHypervector();
        }
    }
}

Vector HDC::getBasisVector(const std::string& key) {
    if (m_basis_vectors.count(key)) {
        return m_basis_vectors.at(key);
    }
    // Return a zero vector if the key is not found to prevent crashes.
    return Vector::Zero(m_dim);
}

Vector HDC::bind(const Vector& v1, const Vector& v2) {
    // Binding is element-wise multiplication (XOR for bipolar vectors).
    // It creates a new vector dissimilar to its inputs, representing their association.
    return v1.cwiseProduct(v2);
}

Vector HDC::bundle(const std::vector<Vector>& vecs) {
    // Bundling is element-wise addition followed by a sign function.
    // It creates a new vector that is similar to all its inputs (a superposition).
    if (vecs.empty()) {
        return Vector::Zero(m_dim);
    }
    Vector sum = Vector::Zero(m_dim);
    for (const auto& vec : vecs) {
        sum += vec;
    }
    // Apply sign function to keep the vector bipolar.
    return sum.unaryExpr([](float x) { return (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f); });
}

Vector HDC::permute(const Vector& v, int shift) {
    // Permutation is a cyclic shift. It creates a new vector that is
    // dissimilar to the original but preserves distance to other vectors.
    // Used for encoding sequences.
    Vector result(m_dim);
    int s = shift % m_dim;
    if (s < 0) s += m_dim;
    for (int i = 0; i < m_dim; ++i) {
        result(i) = v((i - s + m_dim) % m_dim);
    }
    return result;
}

float HDC::cosineSimilarity(const Vector& v1, const Vector& v2) {
    // Measures the similarity between two hypervectors.
    float norm_prod = v1.norm() * v2.norm();
    if (norm_prod == 0) return 0.0f;
    return v1.dot(v2) / norm_prod;
}