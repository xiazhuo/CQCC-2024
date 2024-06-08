#include "QPanda.h"
#include <stdio.h>
#include <cmath>
#include "block_encoding.cpp"
using namespace QPanda;
using namespace std;

const int step = 200; // delta_s = 1/step
const double TT = 25; // 演化时间

struct qdal_res
{
    Eigen::VectorXcd state;
    std::complex<double> fidelity;
};

Eigen::MatrixXcd H_s(double s, Eigen::MatrixXcd H_0, Eigen::MatrixXcd H_1)
{
    Eigen::MatrixXcd H_s_matrix = (1.0 - s) * H_0 + s * H_1;
    return H_s_matrix;
}

// 使用一阶近似得到演化矩阵
Eigen::MatrixXcd generate_H(Eigen::MatrixXcd matrix_A, Eigen::VectorXcd vector_b)
{
    matrix_A = -1.0 * matrix_A;

    int n = vector_b.rows();
    Eigen::MatrixXcd identity_matrix = Eigen::MatrixXcd::Identity(n, n);
    Eigen::MatrixXcd Q_b = identity_matrix - vector_b * vector_b.transpose();
    Eigen::MatrixXcd H_0 = Eigen::MatrixXcd::Zero(2 * n, 2 * n);
    H_0.block(n, 0, n, n) = Q_b;
    H_0.block(0, n, n, n) = Q_b;
    Eigen::MatrixXcd H_1 = Eigen::MatrixXcd::Zero(2 * n, 2 * n);
    H_1.block(n, 0, n, n) = Q_b * matrix_A;
    H_1.block(0, n, n, n) = matrix_A * Q_b;

    double delta_s = 1.0 / step;
    std::complex<double> image_(0.0, 1.0);
    Eigen::MatrixXcd return_matrix = Eigen::MatrixXcd::Identity(2 * n, 2 * n);
    identity_matrix = Eigen::MatrixXcd::Identity(2 * n, 2 * n);

    // 数值计算
    // for (int i = 1; i <= step; ++i)
    // {
    //     double s = (i * 1.0 - 0.5) * delta_s;
    //     return_matrix = expMat(-image_, H_s(s, H_0, H_1), TT * delta_s) * return_matrix;
    // }

    // 一阶近似
    for (int i = 1; i <= step; ++i)
    {
        double s = (i - 0.5) * delta_s;
        return_matrix = (identity_matrix - image_ * H_s(s, H_0, H_1) * TT * delta_s) * return_matrix;
    }

    return return_matrix;
}

qdal_res qda_linear_solver(Eigen::MatrixXcd matrix_A, Eigen::VectorXcd vector_b)
{
    // 获得归一化后的精确解 x_r
    Eigen::VectorXcd x_r = matrix_A.colPivHouseholderQr().solve(vector_b);
    x_r /= x_r.norm();

    vector_b /= vector_b.norm();
    // 获得一阶近似后的演化矩阵 H
    Eigen::MatrixXcd return_H = generate_H(matrix_A, vector_b);
    // 将 H 嵌入到酉矩阵 U_H 中
    Eigen::MatrixXcd U_H = block_encoding_method(return_H);

    auto qvm = new CPUQVM();
    qvm->init();
    int n_qubits = log2(U_H.rows());
    QProg prog;
    auto q = qvm->qAllocMany(n_qubits);

    Encode encode_b;
    std::vector<std::complex<double>> data(vector_b.size());
    for (int i = 0; i < vector_b.size(); ++i)
    {
        data[i] = vector_b(i);
    }
    encode_b.efficient_sparse(q[0], data);

    QCircuit circuit = matrix_decompose_qr(q, Eigen_to_QStat(U_H), false);

    prog << encode_b.get_circuit() << circuit;
    qvm->probRunDict(prog, q);
    auto res = qvm->getQState();

    // 获得 “000000” 态和 “000001” 态的振幅并进行归一化，即是求解得到的 x
    Eigen::VectorXcd x(2);
    x << res[0], res[1];
    x /= x.norm();

    qdal_res result;
    result.state = x;
    result.fidelity = x.dot(x_r);

    destroyQuantumMachine(qvm);
    return result;
}
