#ifndef QDA_LINEAR_SOLVER_H
#define QDA_LINEAR_SOLVER_H

#include "QPanda.h"

struct qdal_res
{
    Eigen::VectorXcd state;
    std::complex<double> fidelity;
};

// 函数声明
qdal_res qda_linear_solver(Eigen::MatrixXcd matrix_A, Eigen::VectorXcd vector_b);

#endif
