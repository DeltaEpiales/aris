#pragma once
#include <Eigen/Dense>
#include <unordered_map>
#include <string>
#include <vector>

using Vector = Eigen::VectorXf;

class HDC {
public:
    explicit HDC(int dim);

    Vector bind(const Vector& v1, const Vector& v2);
    Vector bundle(const std::vector<Vector>& vecs);
    Vector permute(const Vector& v, int shift = 1);
    float cosineSimilarity(const Vector& v1, const Vector& v2);

    void initBasisVectors(int num_entities, const std::string& prefix);
    Vector getBasisVector(const std::string& key);

private:
    int m_dim;
    std::unordered_map<std::string, Vector> m_basis_vectors;
    Vector createRandomHypervector();
};