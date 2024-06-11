#include "QPanda.h"
#include <stdio.h>
#include "block_encoding.cpp"
#include "qda_linear_solver.hpp"
using namespace QPanda;
using namespace std;

Eigen::MatrixXcd H_s(double s, Eigen::MatrixXcd H_0, Eigen::MatrixXcd H_1)
{
    Eigen::MatrixXcd H_s_matrix = (1.0 - s) * H_0 + s * H_1;
    return H_s_matrix;
}

// 使用一阶近似得到演化矩阵
Eigen::MatrixXcd generate_H(Eigen::MatrixXcd matrix_A, Eigen::VectorXcd vector_b)
{
    int n = vector_b.rows();
    Eigen::MatrixXcd identity_matrix = Eigen::MatrixXcd::Identity(n, n);
    Eigen::MatrixXcd Q_b = identity_matrix - vector_b * vector_b.adjoint();
    Eigen::MatrixXcd H_0 = Eigen::MatrixXcd::Zero(2 * n, 2 * n);
    H_0.block(n, 0, n, n) = Q_b;
    H_0.block(0, n, n, n) = Q_b;
    Eigen::MatrixXcd H_1 = Eigen::MatrixXcd::Zero(2 * n, 2 * n);
    H_1.block(n, 0, n, n) = Q_b * matrix_A;
    H_1.block(0, n, n, n) = matrix_A * Q_b;

    const int step = 200;
    double delta_s = 1.0 / step;
    std::complex<double> image_(0.0, 1.0);

    Eigen::MatrixXcd return_H;
    identity_matrix = Eigen::MatrixXcd::Identity(2 * n, 2 * n);
    Eigen::VectorXcd ket_0(2);
    ket_0 << 1, 0;
    double obs_min = 1;

    for (int TT = 1; TT < 200; TT += 1)
    {
        Eigen::MatrixXcd return_matrix = Eigen::MatrixXcd::Identity(2 * n, 2 * n);
        Eigen::VectorXcd psi = kroneckerProduct(ket_0, vector_b);

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

        psi = return_matrix * psi;
        // obs 的最小值对应 fidelity 的最大值
        double obs = abs((psi.adjoint() * H_1 * H_1 * psi)(0, 0));

        if (obs < obs_min)
        {
            obs_min = obs;
            return_H = return_matrix;
        }
    }

    return return_H;
}

qdal_res qda_linear_solver(Eigen::MatrixXcd matrix_A, Eigen::VectorXcd vector_b)
{
    vector_b /= vector_b.norm();
    // 获得一阶近似后的演化矩阵 H
    Eigen::MatrixXcd return_H = generate_H(matrix_A, vector_b);

    // 验证解的量子态
    // Eigen::VectorXcd ket_0(2);
    // ket_0 << 1, 0;
    // Eigen::VectorXcd psi = kroneckerProduct(ket_0, vector_b);
    // psi = return_H * psi;
    // Eigen::VectorXcd x = psi.head(vector_b.size());
    // x = x / x.norm();

    // 将 H 嵌入到酉矩阵 U_H 中
    Eigen::MatrixXcd U_H = block_encoding_method(return_H);

    auto qvm = new CPUQVM();
    qvm->init();
    int n_encode = log2(vector_b.rows());
    int n_ancillary = log2(U_H.rows()) - n_encode;
    QProg prog;
    auto q1 = qvm->qAllocMany(n_encode);
    auto q2 = qvm->qAllocMany(n_ancillary);

    Encode encode_b;
    std::vector<std::complex<double>> data(vector_b.size());
    for (int i = 0; i < vector_b.size(); ++i)
    {
        data[i] = vector_b(i);
    }
    encode_b.efficient_sparse(q1, data);

    QCircuit circuit = matrix_decompose_qr(q1 + q2, Eigen_to_QStat(U_H), false); // 分解很慢

    prog << encode_b.get_circuit() << circuit;
    qvm->probRunDict(prog, q1 + q2);
    auto state = qvm->getQState();

    // 获得辅助量子位状态为0时，原始态的振幅并进行归一化，即是求解得到的 x
    Eigen::VectorXcd x(vector_b.rows());
    for (int i = 0; i < vector_b.rows(); ++i)
    {
        x(i) = state[i];
    }
    x /= x.norm();

    // 获得归一化后的精确解 x_r
    Eigen::VectorXcd x_r = matrix_A.colPivHouseholderQr().solve(vector_b);
    x_r /= x_r.norm();

    qdal_res result;
    result.state = x;
    result.fidelity = x.dot(x_r);

    destroyQuantumMachine(qvm);
    return result;
}