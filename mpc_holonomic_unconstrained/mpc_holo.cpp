#include <ConvexMPC.hpp>
#include <iomanip> // For std::setw
#include <iostream>

double referenceTrajectory(double t) {
  return -0.17 + 0.85 * t - 0.13 * std::pow(t, 2) + 0.008407 * std::pow(t, 3) -
         0.00016 * std::pow(t, 4);
}

// double referenceTrajectory(double t) { return 1; }

// Main function to demonstrate the MPC class usage
int main() {
  const int mpc_horizon = 3;
  const double Ts = 0.1;
  const int simulation_time = 330; // 33
  const int state_dim = 3;
  const int input_dim = 3;

  Eigen::DiagonalMatrix<double, state_dim, state_dim> Q;
  Eigen::DiagonalMatrix<double, input_dim, input_dim> R;

  Eigen::Matrix<double, state_dim, 1> _xMax;
  Eigen::Matrix<double, state_dim, 1> _xMin;
  Eigen::Matrix<double, input_dim, 1> _uMax;
  Eigen::Matrix<double, input_dim, 1> _uMin;

  Eigen::Matrix<double, state_dim, state_dim> A;
  Eigen::Matrix<double, state_dim, input_dim> B;

  Eigen::Matrix<double, state_dim, 1> x0, xRef_const;
  Eigen::MatrixXd xRef;
  xRef.resize(state_dim, mpc_horizon + 1);

  ConvexMPC<state_dim, input_dim> mpc(mpc_horizon, Ts);

  // State variables are [x, y, theta]
  // Control inputs are [vx, vy, w]
  A << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  B << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1;
  Q.diagonal() << 10.0, 10.0, 10.0;
  R.diagonal() << 0.001, 0.001, 0.001;
  _xMax << OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY;
  _xMin << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -OsqpEigen::INFTY;
  _uMax << 1.0, 1.0, 1.0;
  _uMin << -1.0, -1.0, -1.0;
  x0 << 0, 0, 0;
  xRef_const << 1, 1, -1;

  for (int i = 0; i < mpc_horizon + 1; i++) {
    double time = i * Ts;
    xRef(0, i) = 10;
    xRef(1, i) = 10;
    xRef(2, i) = -2;
  }

  mpc.set_dynamics_matrices(A, B);
  mpc.set_state_penalty_matrix(Q);
  mpc.set_control_penalty_matrix(R);
  mpc.set_inequality_constraints(_xMax, _xMin, _uMax, _uMin);
  mpc.set_initial_state(x0);
  mpc.set_reference(xRef);
  mpc.set_reference(xRef_const);

  mpc.cast_mpc_to_qp_hessian();
  mpc.cast_mpc_to_qp_gradient();
  mpc.cast_mpc_to_qp_constraint_matrix();
  mpc.cast_mpc_to_qp_constraint_vectors();

  mpc.initialize_solver();

  for (int step = 0; step < simulation_time; step++) {
    double currentTime = step * Ts;

    // for (int j = 0; j < mpc_horizon + 1; j++) {
    //   double time = currentTime + j * Ts;
    //   xRef(0, j) = 1;
    //   xRef(1, j) = 0;
    //   xRef(2, j) = 0;
    // }
    // mpc.set_reference(xRef_const);
    // mpc.cast_mpc_to_qp_gradient_const();

    mpc.step(x0, true);
    Eigen::VectorXd control = mpc.get_optimal_control();

    int width = 6; // Define the field width for each column

    std::cout << std::left << "Time:" << std::setw(width)
              << currentTime << std::setw(width) << "State:" << std::setw(width)
              << x0.transpose() << std::setw(width)
              << "Reference:" << std::setw(width) << xRef.col(0).transpose()
              << std::setw(width) << "Control:" << std::setw(width)
              << control.transpose() << std::endl;

    // Update state and constraints
    x0 = A * x0 + B * control;
  }

  return 0;
}
