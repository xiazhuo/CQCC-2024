#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include "qda_linear_solver.cpp"
using namespace QPanda;
using namespace std;

int main()
{
    Eigen::MatrixXcd matrix_A(2, 2);
    matrix_A << 2, 1, 1, 0;
    Eigen::VectorXcd vector_b(2);
    vector_b << 3, 1;

    qdal_res result = qda_linear_solver(matrix_A, vector_b);
    cout << result.state << "\n"
         << result.fidelity << endl;
    return 0;
}