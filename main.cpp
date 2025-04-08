#include <iostream>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;


// relative error 
double err(const VectorXd& C, const VectorXd& V){
    return (C-V).norm()/V.norm();
}


// forward
VectorXd forwardSub(const MatrixXd& L, const VectorXd& b) {
    int n = L.rows();
    VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        y(i) = b(i);
        for (int j = 0; j < i; ++j) {
            y(i) -= L(i, j) * y(j);
        }
    }
    return y;
}


// backward
VectorXd backwardSub(const MatrixXd& U, const VectorXd& y) {
    int n = U.rows();
    VectorXd x(n);
    for (int i = n - 1; i >= 0; --i) {
        x(i) = y(i);
        for (int j = i + 1; j < n; ++j) {
            x(i) -= U(i, j) * x(j);
        }
        x(i) /= U(i, i);
    }
    return x;
}


// QR resolution
VectorXd sQR(const MatrixXd& A, const VectorXd& b){
    HouseholderQR<MatrixXd> qr(A);
    MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixQR().triangularView<Upper>();

    // actual calculation
    VectorXd y = Q.transpose() * b;
    VectorXd x = backwardSub(R, y);
    return x;
}


// PALU resolution
VectorXd sPALU(const MatrixXd& A, const VectorXd& b){
    PartialPivLU<MatrixXd> lu(A);
    MatrixXd L = lu.matrixL();
    MatrixXd U = lu.matrixU();
    MatrixXd P = lu.permutationP();

    // actual calculation
    VectorXd Pb = P * b;
    VectorXd y = forwardSub(L, Pb);
    VectorXd x = backwardSub(U, y);
    return x;
}





int main(){

// expected real vector
Vector2d V(-1.0, -1.0); 

// given matrices and vectors
Matrix2d A1;
A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
Matrix2d A2;
A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
Matrix2d A3; 
A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;

Vector2d b1;
b1 << -5.169911863249772e-01, 1.672384680188350e-01;
Vector2d b2;
b2 << -6.394645785530173e-04, 4.259549612877223e-04;
Vector2d b3;
b3 << -6.400391328043042e-10, 4.266924591433963e-10;


// A1
VectorXd x1_palu = sPALU(A1, b1);
VectorXd x1_qr = sQR(A1, b1);
cout << "------ A1 ------" << endl;
cout << "my PALU solution: " << x1_palu.transpose() << endl;
cout << "PALU relative error: " << err(x1_palu, V) << endl;
cout << "my QR solution:   " << x1_qr.transpose() << endl;
cout << "QR relative error:   " << err(x1_qr, V) << endl;

// A2
VectorXd x2_palu = sPALU(A2, b2);
VectorXd x2_qr = sQR(A2, b2);
cout << "------ A2 ------" << endl;
cout << "my PALU solution: " << x2_palu.transpose() << endl;
cout << "PALU relative error: " << err(x2_palu, V) << endl;
cout << "my QR solution:   " << x2_qr.transpose() << endl;
cout << "QR relative error:   " << err(x2_qr, V) << endl;

// A3
VectorXd x3_palu = sPALU(A3, b3);
VectorXd x3_qr = sQR(A3, b3);
cout << "------ A3 ------" << endl;
cout << "my PALU solution: " << x3_palu.transpose() << endl;
cout << "PALU relative error: " << err(x3_palu, V) << endl;
cout << "my QR solution:   " << x3_qr.transpose() << endl;
cout << "QR relative error:   " << err(x3_qr, V) << endl;

return 0;
}
