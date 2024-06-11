#include "qda_linear_solver.cpp"
using namespace QPanda;
using namespace std;

// 生成随机复数
std::complex<double> randomComplex()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    double real = dis(gen);
    double imag = dis(gen);
    return std::complex<double>(real, imag);
}

// 生成随机厄密矩阵
Eigen::MatrixXcd generateRandomHermitianMatrix(int size)
{
    Eigen::MatrixXcd A(size, size);

    // 生成随机矩阵 A
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            A(i, j) = randomComplex();
        }
    }

    // 构造厄密矩阵 H = 0.5 * (A + A^H)
    Eigen::MatrixXcd H = 0.5 * (A + A.adjoint());

    return H;
}

// 生成随机复数向量
Eigen::VectorXcd generateRandomComplexVector(int size)
{
    Eigen::VectorXcd vec(size);

    for (int i = 0; i < size; ++i)
    {
        vec[i] = randomComplex();
    }

    return vec;
}

int main()
{
    double sum_acc = 0;
    int n = 5;
    for (int i = 0; i < n; i++)
    {
        Eigen::MatrixXcd matrix_A = generateRandomHermitianMatrix(2);
        // Eigen::MatrixXcd matrix_A(2, 2);
        // matrix_A << 2, 1, 1, 0;
        Eigen::VectorXcd vector_b = generateRandomComplexVector(2);
        // Eigen::VectorXcd vector_b(2);
        // vector_b << 3, 1;

        qdal_res result = qda_linear_solver(matrix_A, vector_b);
        cout << result.state << "\n"
             << result.fidelity << "\n\n";
        sum_acc += abs(result.fidelity);
    }
    cout << sum_acc / n << "\n";
    return 0;
}