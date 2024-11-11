#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <iostream>

template <int StateDim, int InputDim> class MPCController {
public:
  MPCController(int mpc_horizon, double Ts) {
    this->mpc_horizon = mpc_horizon;
    this->Ts = Ts;
    x0.setZero();
  }

  void
  set_dynamics_matrices(const Eigen::Matrix<double, StateDim, StateDim> &A,
                        const Eigen::Matrix<double, StateDim, InputDim> &B) {
    this->A = A;
    this->B = B;
  }

  void
  set_state_penalty_matrix(const Eigen::Matrix<double, StateDim, StateDim> &Q) {
    this->Q = Q;
  }

  void set_control_penalty_matrix(
      const Eigen::Matrix<double, InputDim, InputDim> &R) {
    this->R = R;
  }

  void set_time_step(double Ts) { this->Ts = Ts; }
  void set_mpc_window(int mpc_horizon) { this->mpc_horizon = mpc_horizon; }

  // Set up constraints
  void set_inequality_constraints(Eigen::Matrix<double, StateDim, 1> _xMax,
                                  Eigen::Matrix<double, StateDim, 1> _xMin,
                                  Eigen::Matrix<double, InputDim, 1> _uMax,
                                  Eigen::Matrix<double, InputDim, 1> _uMin) {
    xMax = _xMax;
    xMin = _xMin;
    uMax = _uMax;
    uMin = _uMin;
  }

  // Set reference state
  void set_reference(const Eigen::MatrixXd &xRef) { this->xRef = xRef; }

  void set_reference(const Eigen::Matrix<double, StateDim, 1> &xRef) {
    this->xRef_const = xRef;
  }

  // Set initial state
  void
  set_initial_state(const Eigen::Matrix<double, StateDim, 1> &initialState) {
    x0 = initialState;
  }

  Eigen::Matrix<double, InputDim, 1>
  step(const Eigen::Matrix<double, StateDim, 1> &state, bool update_gradient) {

    x0 = state;
    update_constraint_vectors();

    // Update gradient for current reference
    if (update_gradient)
      solver.updateGradient(gradient);

    // Solve QP problem
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
      std::cerr << "Solver failed!" << std::endl;
      return Eigen::Matrix<double, InputDim, 1>::Zero();
    }

    // Get control input
    Eigen::VectorXd QPSolution = solver.getSolution();
    Eigen::Matrix<double, InputDim, 1> control =
        QPSolution.block(StateDim * (mpc_horizon + 1), 0, InputDim, 1);

    return control;
  }

  void cast_mpc_to_qp_hessian() {
    hessian.resize(StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon,
                   StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon);

    // populate hessian matrix
    for (int i = 0; i < StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon;
         i++) {
      if (i < StateDim * (mpc_horizon + 1)) {
        int posQ = i % StateDim;
        float value = Q.diagonal()[posQ];
        if (value != 0)
          hessian.insert(i, i) = value;
      } else {
        int posR = i % InputDim;
        float value = R.diagonal()[posR];
        if (value != 0)
          hessian.insert(i, i) = value;
      }
    }
  }

  // Populate gradient vector
  void cast_mpc_to_qp_gradient_const() {
    gradient = Eigen::VectorXd::Zero(StateDim * (mpc_horizon + 1) +
                                     InputDim * mpc_horizon);
    for (int i = 0; i < StateDim * (mpc_horizon + 1); i++) {
      int pos = i % StateDim;
      gradient(i) = -Q(pos, pos) * xRef_const(pos);
    }
  }

  void cast_mpc_to_qp_gradient() {
    gradient = Eigen::VectorXd::Zero(StateDim * (mpc_horizon + 1) +
                                     InputDim * mpc_horizon);

    // Loop over each state dimension and each step in the MPC horizon
    for (int i = 0; i < mpc_horizon + 1; ++i) {
      for (int j = 0; j < StateDim; ++j) {
        // Position in the gradient vector for the current state in the
        // horizon
        int pos = i * StateDim + j;

        // Each entry is the negative product of Q and the reference at each
        // horizon step
        gradient(pos) =
            -Q(j, j) *
            xRef(j, i); // xRef(j, i) corresponds to state j at horizon step i
      }
    }
  }

  void cast_mpc_to_qp_constraint_matrix() {
    constraintMatrix.resize(
        StateDim * (mpc_horizon + 1) + StateDim * (mpc_horizon + 1) +
            InputDim * mpc_horizon,
        StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon);

    // populate linear constraint matrix
    for (int i = 0; i < StateDim * (mpc_horizon + 1); i++) {
      constraintMatrix.insert(i, i) = -1;
    }

    for (int i = 0; i < mpc_horizon; i++)
      for (int j = 0; j < StateDim; j++)
        for (int k = 0; k < StateDim; k++) {
          float value = A(j, k);
          if (value != 0) {
            constraintMatrix.insert(StateDim * (i + 1) + j, StateDim * i + k) =
                value;
          }
        }

    for (int i = 0; i < mpc_horizon; i++)
      for (int j = 0; j < StateDim; j++)
        for (int k = 0; k < InputDim; k++) {
          float value = B(j, k);
          if (value != 0) {
            constraintMatrix.insert(StateDim * (i + 1) + j,
                                    InputDim * i + k +
                                        StateDim * (mpc_horizon + 1)) = value;
          }
        }

    for (int i = 0; i < StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon;
         i++) {
      constraintMatrix.insert(i + (mpc_horizon + 1) * StateDim, i) = 1;
    }
  }

  // Populate constraint bounds
  void cast_mpc_to_qp_constraint_vectors() {
    // evaluate the lower and the upper inequality vectors
    Eigen::VectorXd lowerInequality = Eigen::MatrixXd::Zero(
        StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon, 1);
    Eigen::VectorXd upperInequality = Eigen::MatrixXd::Zero(
        StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon, 1);
    for (int i = 0; i < mpc_horizon + 1; i++) {
      lowerInequality.block(StateDim * i, 0, StateDim, 1) = xMin;
      upperInequality.block(StateDim * i, 0, StateDim, 1) = xMax;
    }
    for (int i = 0; i < mpc_horizon; i++) {
      lowerInequality.block(InputDim * i + StateDim * (mpc_horizon + 1), 0,
                            InputDim, 1) = uMin;
      upperInequality.block(InputDim * i + StateDim * (mpc_horizon + 1), 0,
                            InputDim, 1) = uMax;
    }

    // evaluate the lower and the upper equality vectors
    Eigen::VectorXd lowerEquality =
        Eigen::MatrixXd::Zero(StateDim * (mpc_horizon + 1), 1);
    Eigen::VectorXd upperEquality;
    lowerEquality.block(0, 0, StateDim, 1) = -x0;
    upperEquality = lowerEquality;
    lowerEquality = lowerEquality;

    // merge inequality and equality vectors
    lowerBound = Eigen::MatrixXd::Zero(
        2 * StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon, 1);
    lowerBound << lowerEquality, lowerInequality;

    upperBound = Eigen::MatrixXd::Zero(
        2 * StateDim * (mpc_horizon + 1) + InputDim * mpc_horizon, 1);
    upperBound << upperEquality, upperInequality;
  }

  void initialize_solver() {
    solver.settings()->setWarmStart(true);
    solver.data()->setNumberOfVariables(StateDim * (mpc_horizon + 1) +
                                        InputDim * mpc_horizon);
    solver.data()->setNumberOfConstraints(2 * StateDim * (mpc_horizon + 1) +
                                          InputDim * mpc_horizon);
    solver.data()->setHessianMatrix(hessian);
    solver.data()->setGradient(gradient);
    solver.data()->setLinearConstraintsMatrix(constraintMatrix);
    solver.data()->setLowerBound(lowerBound);
    solver.data()->setUpperBound(upperBound);
    if (!solver.initSolver()) {
      std::cerr << "Solver initialization failed!" << std::endl;
    }
  }

private:
  // Core MPC Parameters
  int mpc_horizon;
  double Ts;

  // State and control matrices
  Eigen::Matrix<double, StateDim, StateDim> Q;
  Eigen::Matrix<double, InputDim, InputDim> R;

  // State and control vectors
  Eigen::Matrix<double, StateDim, 1> x0;
  Eigen::MatrixXd xRef;
  Eigen::Matrix<double, StateDim, 1> xRef_const;
  Eigen::Matrix<double, StateDim, StateDim> A;
  Eigen::Matrix<double, StateDim, InputDim> B;

  // Constraints and weights
  Eigen::Matrix<double, StateDim, 1> xMax, xMin;
  Eigen::Matrix<double, InputDim, 1> uMax, uMin;

  // QP problem components
  Eigen::SparseMatrix<double> hessian;
  Eigen::VectorXd gradient;
  Eigen::SparseMatrix<double> constraintMatrix;
  Eigen::VectorXd lowerBound, upperBound;
  OsqpEigen::Solver solver;

  // Update constraint vectors with the new state
  void update_constraint_vectors() {
    lowerBound(0) = -x0(0);
    upperBound(0) = -x0(0);
    solver.updateBounds(lowerBound, upperBound);
  }
};

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

  MPCController<1, 1> mpc(mpc_horizon, Ts);

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
