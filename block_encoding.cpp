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
        // string label = "";
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
QCircuit get_prep_circuit(QVec qlist, std::vector<double> data)
{
    int n_qubits = ceil(log2(data.size()));
    Encode encode;
    encode.schmidt_encode(qlist, data);
    QCircuit cir = encode.get_circuit();
    return cir;
}

// 当control_qlist为 j 时，对 target_qlist 作用酉矩阵 V_vec[j]
QCircuit get_control_circuit(QVec control_qlist, QVec target_qlist, Eigen::MatrixXcd V_j, int j)
{
    int n_ancillary = control_qlist.size();
    QCircuit cir, cir1, cir2, cir3, cir4;
    for (int i = 0; i < n_ancillary; i++)
    {
        if (((j >> i) & 1) == 0)
        {
            cir << X(control_qlist[i]);
        }
    }
    cir << matrix_decompose_qr(target_qlist, Eigen_to_QStat(V_j), false).control(control_qlist); // 太慢了
    for (int i = 0; i < n_ancillary; i++)
    {
        if (((j >> i) & 1) == 0)
        {
            cir << X(control_qlist[i]);
        }
    }
    return cir;
}

Eigen::MatrixXcd block_encoding_method(Eigen::MatrixXcd hamiltonian)
{
    // 将输入矩阵分解为若干酉矩阵的线性组合（LCU）
    LCU M = linear_combination_pauli(hamiltonian);
    int n = int(log2(hamiltonian.rows()));
    int m = M.size(); // 分解得到的酉矩阵个数
    if (m == 1)
        return hamiltonian;
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

    auto qvm = new CPUQVM();
    qvm->init();
    QProg prog;
    auto q1 = qvm->qAllocMany(n);
    auto q2 = qvm->qAllocMany(n_ancillary);

    // 对辅助量子位作用 W
    QCircuit circuit1 = get_prep_circuit(q2, alpha_vec);
    prog << circuit1;

    // 以辅助量子位为控制量子比特，当辅助量子位为 i 时，对 q1 作用酉矩阵 V_vec[i]
    for (int i = 0; i < m; i++)
    {
        auto circuit2 = get_control_circuit(q2, q1, V_vec[i], i);
        prog << circuit2;
    }

    // 对辅助量子位作用 W.dagger
    prog << circuit1.dagger();

    QStat cir_matrix = getCircuitMatrix(prog, true);
    destroyQuantumMachine(qvm);

    return QStat_to_Eigen(cir_matrix);
}