#include "QPanda.h"
#include "QAlg/Encode/Encode.h"
#include <stdio.h>
using namespace QPanda;
using namespace std;

// Linear Combination of Unitaries
// 前一项是正实系数，后一项是对应的酉矩阵
typedef std::vector<std::pair<double, Eigen::MatrixXcd>> LCU;

// 获得由pauli矩阵的张量组成的一组正交基
std::vector<Eigen::MatrixXcd> get_pauli_bases(int n_qubits)
{
    std::complex<double> image_(0.0, 1.0);
    Eigen::MatrixXcd matrix_I(2, 2);
    matrix_I << 1, 0, 0, 1;
    Eigen::MatrixXcd matrix_X(2, 2);
    matrix_X << 0, 1, 1, 0;
    Eigen::MatrixXcd matrix_Y(2, 2);
    matrix_Y << 0, -image_, image_, 0;
    Eigen::MatrixXcd matrix_Z(2, 2);
    matrix_Z << 1, 0, 0, -1;

    std::vector<Eigen::MatrixXcd> pauli_basis = {matrix_I, matrix_X, matrix_Y, matrix_Z};
    // std::vector<std::string> pauli_labels = {"I", "X", "Y", "Z"};
    std::vector<Eigen::MatrixXcd> pauli_tensors;
    // std::vector<std::string> pauli_tensor_labels;

    for (int i = 0; i < (1 << (2 * n_qubits)); ++i)
    {
        Eigen::MatrixXcd tensor_product = Eigen::MatrixXcd::Identity(1, 1);
        for (int j = 0; j < n_qubits; ++j)
        {
            // 提取第 j 位数字, 每 4 轮一个循环
            int idx = (i >> (2 * j)) & 0b11;
            tensor_product = Eigen::kroneckerProduct(tensor_product, pauli_basis[idx]).eval();
            // label += pauli_labels[idx];
        }
        pauli_tensors.push_back(tensor_product);
        // pauli_tensor_labels.push_back(label);
    }

    return pauli_tensors;
}

// 将任意一个2^n x 2^n形状的矩阵分解为一系列酉矩阵的线性组合
LCU linear_combination_pauli(Eigen::MatrixXcd hamiltonian)
{
    int n_qubits = log2(hamiltonian.rows());
    assert(1 << n_qubits == hamiltonian.rows());

    // 生成n量子比特的Pauli矩阵的张量积
    auto pauli_tensors = get_pauli_bases(n_qubits);

    LCU decompose;

    // 分解Hamiltonian
    for (const auto &pauli_matrix : pauli_tensors)
    {
        std::complex<double> coefficient = (pauli_matrix.adjoint() * hamiltonian).trace() / static_cast<std::complex<double>>(hamiltonian.rows());
        double abs_coefficient = abs(coefficient);
        std::complex<double> phase = coefficient / abs_coefficient; // 相位因子 e^{i\theta}
        if (abs_coefficient > 1e-10)                                // 避免添加零系数
        {
            Eigen::MatrixXcd adjusted_pauli_matrix = phase * pauli_matrix; // 应用相位因子，将系数转为正实数
            decompose.push_back({abs_coefficient, adjusted_pauli_matrix});
        }
    }

    return decompose;
}

// 获得数据编码电路的酉矩阵，以得到 W
Eigen::MatrixXcd get_init_matrix(std::vector<double> data)
{
    auto qvm = new CPUQVM();
    qvm->init();
    int n_qubits = ceil(log2(data.size()));
    auto qlist = qvm->qAllocMany(n_qubits);

    Encode encode;
    encode.schmidt_encode(qlist, data);
    QCircuit cir = encode.get_circuit();
    QStat cir_matrix = getCircuitMatrix(cir);

    destroyQuantumMachine(qvm);
    return QStat_to_Eigen(cir_matrix);
}

// 获得酉矩阵 V, 这里使用张量积代替控制操作
Eigen::MatrixXcd get_control_matrix(std::vector<Eigen::MatrixXcd> V_vec, int n)
{
    int m = V_vec.size();
    int n_ancillary = ceil(log2(m));
    Eigen::MatrixXcd V;
    V.setZero(1 << (n_ancillary + n), 1 << (n_ancillary + n));

    for (int i = 0; i < m; ++i)
    {
        // 当控制比特测量结果为 i 时，对目标比特作用酉矩阵 V_vec[i]
        Eigen::MatrixXcd control_i;
        control_i.setZero(1 << n_ancillary, 1 << n_ancillary);
        control_i(i, i) = 1;
        V += Eigen::kroneckerProduct(control_i, V_vec[i]);
    }
    return V;
}

// 使用 W, V, W.dagger 得到块编码矩阵
Eigen::MatrixXcd get_block_matrix(Eigen::MatrixXcd W, Eigen::MatrixXcd V, int n, int n_ancillary)
{
    auto qvm = new CPUQVM();
    qvm->init();
    QProg prog;
    auto q1 = qvm->qAllocMany(n);
    auto q2 = qvm->qAllocMany(n_ancillary);

    // 对辅助量子位作用 W
    auto state = Eigen_to_QStat(W);
    QCircuit circuit1 = matrix_decompose_qr(q2, state);
    prog << circuit1;

    // 以辅助量子位为控制量子比特，当辅助量子位为 i 时，对 q1 作用酉矩阵 V_vec[i]
    state = Eigen_to_QStat(V);
    QCircuit circuit2 = matrix_decompose_qr(q1 + q2, state, false);
    prog << circuit2;

    // 对辅助量子位作用 W.dagger
    prog << circuit1.dagger();

    QStat cir_matrix = getCircuitMatrix(prog, true);
    destroyQuantumMachine(qvm);
    return QStat_to_Eigen(cir_matrix);
}

Eigen::MatrixXcd block_encoding_method(Eigen::MatrixXcd hamiltonian)
{
    // 将输入矩阵分解为若干酉矩阵的线性组合（LCU）
    LCU M = linear_combination_pauli(hamiltonian);
    int n = int(log2(hamiltonian.rows()));
    int m = M.size();                // 分解得到的酉矩阵个数
    int n_ancillary = ceil(log2(m)); // 辅助量子比特数

    std::vector<double> alpha_vec;       // 存储实系数
    std::vector<Eigen::MatrixXcd> V_vec; // 存储酉矩阵

    double sum_alpha = 0;
    for (int i = 0; i < m; ++i)
    {
        sum_alpha += M[i].first;
        V_vec.push_back(M[i].second);
    }
    for (int i = 0; i < (1 << n_ancillary); ++i)
    {
        if (i < m)
            alpha_vec.push_back(sqrt(M[i].first / sum_alpha));
        else
            alpha_vec.push_back(0);
    }

    Eigen::MatrixXcd W, V;
    Eigen::MatrixXcd block_encode;
    W = get_init_matrix(alpha_vec);
    V = get_control_matrix(V_vec, n);

    block_encode = get_block_matrix(W, V, n, n_ancillary);
    return block_encode;
}