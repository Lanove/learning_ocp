#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <iostream>

template <int StateDim, int InputDim, int Horizon> class MPCController {
public:
  // Constructor to initialize MPC settings with Q and R as matrices
  MPCController(
      double Ts,
      const Eigen::Matrix<double, StateDim, StateDim>
          &A, // System state dynamics matrix
      const Eigen::Matrix<double, StateDim, InputDim>
          &B, // System input dynamics matrix
      const Eigen::Matrix<double, StateDim, StateDim>
          &Q, // State penalty matrix
      const Eigen::Matrix<double, InputDim, InputDim>
          &R,                                   // Control penalty matrix
      Eigen::Matrix<double, StateDim, 1> _xMax, // State maximum constraints
      Eigen::Matrix<double, StateDim, 1> _xMin, // State minimum constraints
      Eigen::Matrix<double, InputDim, 1> _uMax, // Control maximum constraints
      Eigen::Matrix<double, InputDim, 1> _uMin, // Control minimum constraints
      const Eigen::Matrix<double, StateDim, 1> &initialState, // Initial state
      const Eigen::Matrix<double, StateDim, 1> &xRef          // Reference state
  ) {
    this->Ts = Ts;
    set_dynamics_matrices(A, B);
    set_state_penalty_matrix(Q);
    set_control_penalty_matrix(R);
    set_inequality_constraints(_xMax, _xMin, _uMax, _uMin);
    set_reference(xRef);
    set_initial_state(initialState);
    x0.setZero();
    xRef.setZero();
  }

  MPCController(double Ts) {
    this->Ts = Ts;
    x0.setZero();
    xRef.setZero();
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
  void set_reference(const Eigen::Matrix<double, StateDim, 1> &xRef) {
    this->xRef = xRef;
  }

  // Set initial state
  void
  set_initial_state(const Eigen::Matrix<double, StateDim, 1> &initialState) {
    x0 = initialState;
  }

  // Run MPC for specified number of steps
  void run(int numSteps) {
    // Initialize QP problem matrices
    for (int step = 0; step < numSteps; step++) {
      // Update gradient for current reference
      cast_mpc_to_qp_gradient();
      solver.updateGradient(gradient);

      // Solve QP problem
      if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
        std::cerr << "Solver failed!" << std::endl;
        return;
      }

      // Get control input
      Eigen::VectorXd QPSolution = solver.getSolution();
      Eigen::Matrix<double, InputDim, 1> control =
          QPSolution.block(StateDim * (Horizon + 1), 0, InputDim, 1);

      // Print results
      std::cout << "Step: " << step << " State: " << x0.transpose()
                << " Control: " << control.transpose() << std::endl;

      // Update state and constraints
      x0 = A * x0 + B * control;
      update_constraint_vectors();
    }
  }

  void cast_mpc_to_qp() {
    cast_mpc_to_qp_hessian();
    cast_mpc_to_qp_constraint_matrix();
    cast_mpc_to_qp_constraint_vectors();
    cast_mpc_to_qp_gradient();
  }

  void initialize_solver() {
    solver.settings()->setWarmStart(true);
    solver.data()->setNumberOfVariables(StateDim * (Horizon + 1) +
                                        InputDim * Horizon);
    solver.data()->setNumberOfConstraints(2 * StateDim * (Horizon + 1) +
                                          InputDim * Horizon);
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
  double Ts;

  // State and control matrices
  Eigen::Matrix<double, StateDim, StateDim> Q;
  Eigen::Matrix<double, InputDim, InputDim> R;

  // State and control vectors
  Eigen::Matrix<double, StateDim, 1> x0;
  Eigen::Matrix<double, StateDim, 1> xRef;
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

  void cast_mpc_to_qp_hessian() {
    hessian.resize(StateDim * (Horizon + 1) + InputDim * Horizon,
                   StateDim * (Horizon + 1) + InputDim * Horizon);

    // populate hessian matrix
    for (int i = 0; i < StateDim * (Horizon + 1) + InputDim * Horizon; i++) {
      if (i < StateDim * (Horizon + 1)) {
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
  void cast_mpc_to_qp_gradient() {
    gradient =
        Eigen::VectorXd::Zero(StateDim * (Horizon + 1) + InputDim * Horizon);
    for (int i = 0; i < StateDim * (Horizon + 1); i++) {
      int pos = i % StateDim;
      gradient(i) = -Q(pos, pos) * xRef(pos);
    }
  }

  void cast_mpc_to_qp_constraint_matrix() {
    constraintMatrix.resize(StateDim * (Horizon + 1) +
                                StateDim * (Horizon + 1) + InputDim * Horizon,
                            StateDim * (Horizon + 1) + InputDim * Horizon);

    // populate linear constraint matrix
    for (int i = 0; i < StateDim * (Horizon + 1); i++) {
      constraintMatrix.insert(i, i) = -1;
    }

    for (int i = 0; i < Horizon; i++)
      for (int j = 0; j < StateDim; j++)
        for (int k = 0; k < StateDim; k++) {
          float value = A(j, k);
          if (value != 0) {
            constraintMatrix.insert(StateDim * (i + 1) + j, StateDim * i + k) =
                value;
          }
        }

    for (int i = 0; i < Horizon; i++)
      for (int j = 0; j < StateDim; j++)
        for (int k = 0; k < InputDim; k++) {
          float value = B(j, k);
          if (value != 0) {
            constraintMatrix.insert(StateDim * (i + 1) + j,
                                    InputDim * i + k +
                                        StateDim * (Horizon + 1)) = value;
          }
        }

    for (int i = 0; i < StateDim * (Horizon + 1) + InputDim * Horizon; i++) {
      constraintMatrix.insert(i + (Horizon + 1) * StateDim, i) = 1;
    }
  }

  // Populate constraint bounds

  void cast_mpc_to_qp_constraint_vectors() {
    // evaluate the lower and the upper inequality vectors
    Eigen::VectorXd lowerInequality =
        Eigen::MatrixXd::Zero(StateDim * (Horizon + 1) + InputDim * Horizon, 1);
    Eigen::VectorXd upperInequality =
        Eigen::MatrixXd::Zero(StateDim * (Horizon + 1) + InputDim * Horizon, 1);
    for (int i = 0; i < Horizon + 1; i++) {
      lowerInequality.block(StateDim * i, 0, StateDim, 1) = xMin;
      upperInequality.block(StateDim * i, 0, StateDim, 1) = xMax;
    }
    for (int i = 0; i < Horizon; i++) {
      lowerInequality.block(InputDim * i + StateDim * (Horizon + 1), 0,
                            InputDim, 1) = uMin;
      upperInequality.block(InputDim * i + StateDim * (Horizon + 1), 0,
                            InputDim, 1) = uMax;
    }

    // evaluate the lower and the upper equality vectors
    Eigen::VectorXd lowerEquality =
        Eigen::MatrixXd::Zero(StateDim * (Horizon + 1), 1);
    Eigen::VectorXd upperEquality;
    lowerEquality.block(0, 0, StateDim, 1) = -x0;
    upperEquality = lowerEquality;
    lowerEquality = lowerEquality;

    // merge inequality and equality vectors
    lowerBound = Eigen::MatrixXd::Zero(
        2 * StateDim * (Horizon + 1) + InputDim * Horizon, 1);
    lowerBound << lowerEquality, lowerInequality;

    upperBound = Eigen::MatrixXd::Zero(
        2 * StateDim * (Horizon + 1) + InputDim * Horizon, 1);
    upperBound << upperEquality, upperInequality;
  }

  // Update constraint vectors with the new state
  void update_constraint_vectors() {
    lowerBound(0) = -x0(0);
    upperBound(0) = -x0(0);
    solver.updateBounds(lowerBound, upperBound);
  }
};

// double referenceTrajectory(double t) {
//     return -0.17 + 0.85 * t - 0.13 * std::pow(t, 2) + 0.008407 * std::pow(t,
//     3) - 0.00016 * std::pow(t, 4);
// }

double referenceTrajectory(double t) { return 1; }

int main() {
  const int state_dim = 1;
  const int input_dim = 1;
  const int mpc_horizon = 10;
  const double Ts = 0.1;
  const int simulation_time = 330; // 33 seconds

  Eigen::Matrix<double, 1, 1> Q;
  Eigen::Matrix<double, 1, 1> R;

  Eigen::Matrix<double, 1, 1> _xMax;
  Eigen::Matrix<double, 1, 1> _xMin;
  Eigen::Matrix<double, 1, 1> _uMax;
  Eigen::Matrix<double, 1, 1> _uMin;

  Eigen::Matrix<double, 1, 1> A, B;

  Eigen::Matrix<double,  1, 1> xRef, x0;

  MPCController<state_dim, input_dim, mpc_horizon> mpc(Ts);

  A << 0.904837;
  B << 0.0951626;
  Q << 10.0;
  R << 0.001;
  _xMax << OsqpEigen::INFTY;
  _xMin << -OsqpEigen::INFTY;
  _uMax << 10.0;
  _uMin << -10.0;

//   for (int j = 0; j < mpc_horizon + 1; j++) {
//     double time = j * Ts;
//     xRef(j) = referenceTrajectory(time);
//   }
    xRef << 7;
  x0 << 0;

  mpc.set_dynamics_matrices(A, B);
  mpc.set_state_penalty_matrix(Q);
  mpc.set_control_penalty_matrix(R);
  mpc.set_inequality_constraints(_xMax, _xMin, _uMax, _uMin);
  mpc.set_reference(xRef);
  mpc.set_initial_state(x0);
  mpc.cast_mpc_to_qp();
  mpc.initialize_solver();

  for (int i = 0; i < simulation_time; i++) {

    for (int j = 0; j < mpc_horizon + 1; j++) {
      double time = (i + j) * Ts;
      xRef(j) = referenceTrajectory(time);
    }
  }

  // Run the MPC
  mpc.run(100);

  return 0;
}
