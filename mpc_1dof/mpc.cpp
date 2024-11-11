#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

// System parameters
const double RC = 1; // Adjust RC value if known
const double A_cont = -1.0 / RC;
const double B_cont = 1.0 / RC;
const double C = 1.0;
const double D = 0.0;
const double Ts = 0.1; // Sampling time for discretization
const int mpcWindow = 10;
const int simulationSteps = 330; // 33 seconds / Ts

// Discrete-time system matrices (for single SISO system)
double A_discrete, B_discrete;

void discretizeSystem() {
  A_discrete = std::exp(A_cont * Ts);
  B_discrete = (1.0 - A_discrete) / -A_cont;
  std::cout << "Discrete-time system matrices: A = " << A_discrete
            << ", B = " << B_discrete << std::endl;
}

// double referenceTrajectory(double t) {
//     return -0.17 + 0.85 * t - 0.13 * std::pow(t, 2) + 0.008407 * std::pow(t,
//     3) - 0.00016 * std::pow(t, 4);
// }

double referenceTrajectory(double t) { return 1; }

void castMPCToQPHessian(double Q, double R, int mpcWindow,
                        Eigen::SparseMatrix<double> &hessianMatrix) {
  hessianMatrix.resize(mpcWindow + 1 + mpcWindow, mpcWindow + 1 + mpcWindow);
  for (int i = 0; i < mpcWindow + 1; i++)
    hessianMatrix.insert(i, i) = Q; // State cost

  for (int i = mpcWindow + 1; i < mpcWindow + 1 + mpcWindow; i++)
    hessianMatrix.insert(i, i) = R; // Input cost
}

void castMPCToQPGradient(double Q, const Eigen::VectorXd &xRef, int mpcWindow,
                         Eigen::VectorXd &gradient) {
  gradient = Eigen::VectorXd::Zero(mpcWindow + 1 + mpcWindow);
  for (int i = 0; i < mpcWindow + 1; i++)
    gradient(i) = -Q * xRef(i);
}

void castMPCToQPConstraintMatrix(
    double A, double B, int mpcWindow,
    Eigen::SparseMatrix<double> &constraintMatrix) {
  constraintMatrix.resize(2 * (mpcWindow + 1) + mpcWindow,
                          mpcWindow + 1 + mpcWindow);

  for (int i = 0; i < mpcWindow + 1; i++) {
    constraintMatrix.insert(i, i) = -1;
    constraintMatrix.insert(mpcWindow + 1 + i, i) = 1;
  }

  for (int i = 0; i < mpcWindow; i++) {
    constraintMatrix.insert(i + 1, i) = A;
    constraintMatrix.insert(i + 1, mpcWindow + 1 + i) = B;
    constraintMatrix.insert(mpcWindow + 2 + mpcWindow + i, mpcWindow + 1 + i) =
        1;
  }
}

void castMPCToQPConstraintVectors(const Eigen::VectorXd &xMin,
                                  const Eigen::VectorXd &xMax,
                                  const Eigen::VectorXd &uMin,
                                  const Eigen::VectorXd &uMax, double x0,
                                  int mpcWindow, Eigen::VectorXd &lowerBound,
                                  Eigen::VectorXd &upperBound) {
  lowerBound = Eigen::VectorXd::Zero(2 * (mpcWindow + 1) + mpcWindow);
  upperBound = Eigen::VectorXd::Zero(2 * (mpcWindow + 1) + mpcWindow);

  lowerBound(0) = -x0;
  upperBound(0) = -x0;

  for (int i = 1; i < mpcWindow + 1; i++) {
    lowerBound(i) = xMin(i);
    upperBound(i) = xMax(i);
  }

  for (int i = 0; i < mpcWindow; i++) {
    lowerBound(mpcWindow + 1 + i) = 0;
    upperBound(mpcWindow + 1 + i) = 0;
  }

  for (int i = 0; i < mpcWindow; i++) {
    lowerBound(2 * (mpcWindow + 1) + i) = uMin(i);
    upperBound(2 * (mpcWindow + 1) + i) = uMax(i);
  }
}

void updateConstraintVectors(double x0, Eigen::VectorXd &lowerBound,
                             Eigen::VectorXd &upperBound) {
  lowerBound(0) = -x0;
  upperBound(0) = -x0;
}

int main() {
  discretizeSystem();

  // Set up constraints
  Eigen::VectorXd xMax =
      Eigen::VectorXd::Constant(mpcWindow + 1, OsqpEigen::INFTY);
  Eigen::VectorXd xMin =
      Eigen::VectorXd::Constant(mpcWindow + 1, -OsqpEigen::INFTY);
  Eigen::VectorXd uMax = Eigen::VectorXd::Constant(mpcWindow, 10.0);
  Eigen::VectorXd uMin = Eigen::VectorXd::Constant(mpcWindow, -10.0);

  // Weight matrices
  double Q = 1.0;
  double R = 0.00001;

  // Allocate QP problem matrices and vectors
  Eigen::SparseMatrix<double> hessian;
  Eigen::VectorXd gradient;
  Eigen::SparseMatrix<double> constraintMatrix;
  Eigen::VectorXd lowerBound;
  Eigen::VectorXd upperBound;

  // Initial and reference states
  double x0 = 0.5;
  Eigen::VectorXd xRef = Eigen::VectorXd::Zero(mpcWindow + 1);

  // Set reference trajectory
  for (int i = 0; i < mpcWindow + 1; i++) {
    double time = i * Ts;
    xRef(i) = referenceTrajectory(time);
  }

  // Cast the MPC problem to QP form
  castMPCToQPHessian(Q, R, mpcWindow, hessian);
  castMPCToQPGradient(Q, xRef, mpcWindow, gradient);
  castMPCToQPConstraintMatrix(A_discrete, B_discrete, mpcWindow,
                              constraintMatrix);
  castMPCToQPConstraintVectors(xMin, xMax, uMin, uMax, x0, mpcWindow,
                               lowerBound, upperBound);

  std::cout << "Hessian matrix: " << hessian << std::endl;
  std::cout << "Gradient: " << gradient.transpose() << std::endl;
  std::cout << "Constraint matrix: " << constraintMatrix << std::endl;
  std::cout << "Lower bound: " << lowerBound.transpose() << std::endl;
  std::cout << "Upper bound: " << upperBound.transpose() << std::endl;

  // Set up the QP solver
  OsqpEigen::Solver solver;
  solver.settings()->setWarmStart(true);

  solver.data()->setNumberOfVariables(mpcWindow + 1 + mpcWindow);
  solver.data()->setNumberOfConstraints(2 * (mpcWindow + 1) + mpcWindow);

  if (!solver.data()->setHessianMatrix(hessian))
    return 1;
  if (!solver.data()->setGradient(gradient))
    return 1;
  if (!solver.data()->setLinearConstraintsMatrix(constraintMatrix))
    return 1;
  if (!solver.data()->setLowerBound(lowerBound))
    return 1;
  if (!solver.data()->setUpperBound(upperBound))
    return 1;

  // instantiate the solver
  if (!solver.initSolver())
    return 1;

  // Controller input and QP solution
  double ctr = 0.0;
  Eigen::VectorXd QPSolution;

  // Simulation loop for trajectory tracking
  for (int i = 0; i < simulationSteps; i++) {
    double currentTime = i * Ts;

    for (int j = 0; j < mpcWindow + 1; j++) {
      double time = (i + j) * Ts;
      xRef(j) = referenceTrajectory(time);
    }

    castMPCToQPGradient(Q, xRef, mpcWindow, gradient);
    solver.updateGradient(gradient);
    std::cout << "xRef: " << xRef.transpose() << std::endl;

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
      return 1;

    QPSolution = solver.getSolution();
    ctr = QPSolution(mpcWindow + 1);
    if (i == 0)
      std::cout << QPSolution << std::endl;
    // ctr = 1;

    x0 = A_discrete * x0 + B_discrete * ctr;
    updateConstraintVectors(x0, lowerBound, upperBound);
    solver.updateBounds(lowerBound, upperBound);

    std::cout << "Time: " << currentTime << " s, Control: " << ctr
              << ", State: " << x0 << ", Reference: " << xRef(0) << std::endl;
  }

  return 0;
}
