#include <iostream>
#include <ConvexMPC.hpp>

double referenceTrajectory(double t) {
  return -0.17 + 0.85 * t - 0.13 * std::pow(t, 2) + 0.008407 * std::pow(t, 3) -
         0.00016 * std::pow(t, 4);
}

// double referenceTrajectory(double t) { return 1; }

// Main function to demonstrate the MPC class usage
int main() {
  const int mpc_horizon = 10;
  const double Ts = 0.1;
  const int simulation_time = 330; // 33
  const int state_dim = 1;
  const int input_dim = 1;

  Eigen::Matrix<double, state_dim, state_dim> Q;
  Eigen::Matrix<double, input_dim, input_dim> R;

  Eigen::Matrix<double, state_dim, 1> _xMax;
  Eigen::Matrix<double, state_dim, 1> _xMin;
  Eigen::Matrix<double, input_dim, 1> _uMax;
  Eigen::Matrix<double, input_dim, 1> _uMin;

  Eigen::Matrix<double, state_dim, state_dim> A;
  Eigen::Matrix<double, state_dim, input_dim> B;

  Eigen::Matrix<double, state_dim, 1> x0;
  Eigen::MatrixXd xRef;
  xRef.resize(state_dim, mpc_horizon + 1);

  ConvexMPC<1, 1> mpc(mpc_horizon, Ts);

  A << 0.904837;
  B << 0.0951626;
  Q << 10.0;
  R << 0.001;
  _xMax << OsqpEigen::INFTY;
  _xMin << -OsqpEigen::INFTY;
  _uMax << 10.0;
  _uMin << -10.0;
  x0 << 0;

  for (int i = 0; i < mpc_horizon + 1; i++) {
    double time = i * Ts;
    xRef(0,i) = referenceTrajectory(time);
  }

  mpc.set_dynamics_matrices(A, B);
  mpc.set_state_penalty_matrix(Q);
  mpc.set_control_penalty_matrix(R);
  mpc.set_inequality_constraints(_xMax, _xMin, _uMax, _uMin);
  mpc.set_initial_state(x0);
  mpc.set_reference(xRef);

  mpc.cast_mpc_to_qp_hessian();
  mpc.cast_mpc_to_qp_gradient();
  mpc.cast_mpc_to_qp_constraint_matrix();
  mpc.cast_mpc_to_qp_constraint_vectors();

  mpc.initialize_solver();

  for (int step = 0; step < simulation_time; step++) {
    double currentTime = step * Ts;

    for (int j = 0; j < mpc_horizon + 1; j++) {
      double time = currentTime + j * Ts;
      xRef(0,j) = referenceTrajectory(time);
    }
    mpc.set_reference(xRef);
    mpc.cast_mpc_to_qp_gradient();

    Eigen::Matrix<double, input_dim, 1> control = mpc.step(x0, true);

    // Print results
    std::cout << "Time: " << currentTime << " State: " << x0.transpose() << " Reference: " << referenceTrajectory(currentTime)
              << " Control: " << control.transpose() << std::endl;

    // Update state and constraints
    x0 = A * x0 + B * control;
  }

  return 0;
}
