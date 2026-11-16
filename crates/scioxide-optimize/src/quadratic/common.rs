//! Shared objective arithmetic and original-coordinate result construction.
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

use scioxide_core::backend::{checked_product, try_filled};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, Tolerance};
use scioxide_linalg::{HouseholderDecomposition, HouseholderOrientation, Matrix};

use crate::context::SolveSession;
use crate::problem::{CoefficientMatrix, QpProblem};
use crate::programming::ProgrammingResult;
use crate::result::OptimizeTermination;
use scioxide_core::EvaluationRole;

/// Compare objective-unit residuals after uniform Hessian/objective scaling.
pub(super) fn objective_threshold<S: RealScalar>(
    t: Tolerance<S>,
    reference: S,
    scale: S,
) -> Result<S, SciOxideError> {
    let absolute = t.absolute() / scale;
    let absolute = if absolute.is_finite() {
        absolute
    } else {
        S::MAX
    };
    let relative = t.relative() * reference;
    let threshold = absolute + relative;
    Ok(if threshold.is_finite() {
        threshold
    } else {
        S::MAX
    })
}

pub(super) fn factor_qr<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<HouseholderDecomposition<S>, SciOxideError> {
    let mut factor = HouseholderDecomposition::empty();
    let mut workspace = scioxide_linalg::LinalgWorkspace::new();
    session.invoke(EvaluationRole::Factorization, || {
        HouseholderDecomposition::factor_into(
            matrix,
            HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            &mut factor,
            &mut workspace,
            session.execution()?,
        )
    })?;
    Ok(factor)
}

pub(super) fn stopped_result<S: RealScalar + NumericScalar<Real = S>>(
    problem: &QpProblem<S>,
    hessian: &[S],
    x: Vec<S>,
    iterations: usize,
    termination: OptimizeTermination,
) -> Result<ProgrammingResult<S>, SciOxideError> {
    let model = super::model::Model::new(problem)?;
    let diagnostics = model.diagnostics(problem, &x)?;
    let fun = eval_objective(hessian, problem.linear_objective(), &x, problem.dimension())?;
    Ok(ProgrammingResult {
        evidence: None,
        verification: None,
        x,
        fun,
        iterations,
        termination,
        diagnostics,
        ..ProgrammingResult::empty()
    })
}

pub(super) fn observed_certificate<S: RealScalar + NumericScalar<Real = S>>(
    problem: &QpProblem<S>,
    hessian: &[S],
    x: &[S],
    tolerance: crate::OptimalityTolerances<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<crate::verification::candidate::Assessment<S>, SciOxideError> {
    let gradient = compute_gradient(hessian, problem.linear_objective(), x, problem.dimension())?;
    match crate::verification::candidate::assess_qp(problem, x, &gradient, tolerance, session) {
        Err(_) if session.stopped().is_some() => {
            Ok(crate::verification::candidate::Assessment::unavailable())
        }
        other => other,
    }
}

pub(super) fn materialize_symmetric_hessian<S>(
    hessian: &CoefficientMatrix<S>,
) -> Result<Vec<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let dimension = hessian.rows();
    let len = checked_product(
        &[dimension, dimension],
        scioxide_core::quantity!("qp_dense_hessian_shape"),
    )?;
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let two =
        <S as scioxide_core::AlgebraicScalar>::ONE + <S as scioxide_core::AlgebraicScalar>::ONE;
    let mut dense = try_filled(len, zero, scioxide_core::quantity!("qp_dense_hessian"))?;
    for row in 0..dimension {
        dense[row * dimension + row] = hessian.get(row, row)?;
        for col in (row + 1)..dimension {
            let average = hessian.get(row, col)? / two + hessian.get(col, row)? / two;
            if !average.is_finite() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "symmetric averaging produced a non-finite coefficient",
                ));
            }
            dense[row * dimension + col] = average;
            dense[col * dimension + row] = average;
        }
    }
    Ok(dense)
}

/// Evaluate the QP objective: `0.5 x^T Q x + c^T x`.
pub(super) fn eval_objective<S: RealScalar + NumericScalar<Real = S>>(
    q: &[S],
    c: &[S],
    x: &[S],
    n: usize,
) -> Result<S, SciOxideError> {
    use scioxide_core::{Scaled, ScaledAccumulator};
    let mut value = ScaledAccumulator::ZERO;
    for (row, (&coordinate, &linear)) in x.iter().zip(c).enumerate() {
        if coordinate == <S as scioxide_core::AlgebraicScalar>::ZERO {
            continue;
        }
        value.add_product(Scaled::from_value(linear)?, Scaled::from_value(coordinate)?)?;
        for (&coefficient, &other) in q[row * n..(row + 1) * n].iter().zip(x) {
            if coefficient == <S as scioxide_core::AlgebraicScalar>::ZERO
                || other == <S as scioxide_core::AlgebraicScalar>::ZERO
            {
                continue;
            }
            let term = Scaled::from_value(<S as RealScalar>::HALF)?
                .mul(Scaled::from_value(coordinate)?)?
                .mul(Scaled::from_value(coefficient)?)?
                .mul(Scaled::from_value(other)?)?;
            value.add_scaled(term)?;
        }
    }
    value.total()?.round()
}

/// Compute the gradient of the QP objective: `Q x + c`.
pub(super) fn compute_gradient<S: RealScalar + NumericScalar<Real = S>>(
    q: &[S],
    c: &[S],
    x: &[S],
    n: usize,
) -> Result<Vec<S>, SciOxideError> {
    let mut gradient = try_filled(
        n,
        <S as scioxide_core::AlgebraicScalar>::ZERO,
        scioxide_core::quantity!("qp_gradient"),
    )?;
    compute_gradient_into(q, c, x, &mut gradient)?;
    Ok(gradient)
}

pub(super) fn compute_gradient_into<S: RealScalar + NumericScalar<Real = S>>(
    q: &[S],
    c: &[S],
    x: &[S],
    gradient: &mut [S],
) -> Result<(), SciOxideError> {
    let n = gradient.len();
    gradient.iter_mut().enumerate().try_for_each(|(i, value)| {
        let mut component = scioxide_core::RangeAccumulator::new();
        component.add(c[i])?;
        q[i * n..(i + 1) * n]
            .iter()
            .zip(x)
            .try_for_each(|(&coefficient, &value)| component.add_product(coefficient, value))?;
        *value = component.finish()?;
        Ok(())
    })
}

/// Project a point onto the feasible box defined by bounds.
pub(super) fn project_to_bounds<S: RealScalar>(x: &mut [S], bounds: &[(Option<S>, Option<S>)]) {
    for (xi, &(lo, hi)) in x.iter_mut().zip(bounds.iter()) {
        if let Some(lo_val) = lo
            && *xi < lo_val
        {
            *xi = lo_val;
        }
        if let Some(hi_val) = hi
            && *xi > hi_val
        {
            *xi = hi_val;
        }
    }
}

/// Compute the norm of the projected gradient.
///
/// The projected gradient at a bound is zero if the gradient would push
/// the variable into the bound.
pub(super) fn projected_gradient_norm<S: RealScalar + NumericScalar<Real = S>>(
    x: &[S],
    grad: &[S],
    bounds: &[(Option<S>, Option<S>)],
    tolerance: Tolerance<S>,
) -> Result<S, SciOxideError> {
    let zero_s = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let scale = x.iter().chain(grad).fold(
        <S as scioxide_core::AlgebraicScalar>::ONE,
        |scale, value| scale.max(NumericScalar::abs(*value)),
    );
    let active_tolerance = tolerance.threshold(scale)?;
    let mut norm = scioxide_core::ScaledNorm::new();
    x.iter()
        .zip(grad)
        .zip(bounds)
        .map(|((&xi, &gi), &(lo, hi))| match (lo, hi) {
            (Some(lo_val), _)
                if NumericScalar::abs(xi - lo_val) <= active_tolerance && gi > zero_s =>
            {
                zero_s
            }
            (_, Some(hi_val))
                if NumericScalar::abs(xi - hi_val) <= active_tolerance && gi < zero_s =>
            {
                zero_s
            }
            _ => gi,
        })
        .try_for_each(|value| norm.add(value))?;
    norm.finish()
}

/// Compute an initial step size from the Hessian diagonal.
///
/// Uses `1 / max(diag(Q))` as a conservative initial step.
///
/// # Errors
///
/// - [`scioxide_core::SciOxideErrorKind::NumericalInstability`] — if all diagonal entries are near-zero
///
pub(super) fn initial_step_size<S: RealScalar + NumericScalar<Real = S>>(
    q: &[S],
    n: usize,
) -> Result<S, SciOxideError> {
    let zero_s = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let max_diag = (0..n).fold(zero_s, |acc, i| {
        let d = NumericScalar::abs(q[i * n + i]);
        if d > acc { d } else { acc }
    });

    if max_diag == zero_s {
        Ok(<S as scioxide_core::AlgebraicScalar>::ONE)
    } else {
        scioxide_core::checked_float_quotient(<S as scioxide_core::AlgebraicScalar>::ONE, max_diag)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use crate::QpSolver;
    use crate::problem::{LinearConstraintSense, LinearConstraints};
    use scioxide_core::Bounds;
    use scioxide_core::Budget;

    #[test]
    fn original_quadratic_terms_cancel_before_rounding() {
        macro_rules! check {
            ($scalar:ty) => {{
                let maximum = <$scalar>::MAX;
                assert_eq!(
                    eval_objective(&[maximum], &[-maximum], &[2.0], 1).unwrap(),
                    0.0
                );
                assert_eq!(
                    compute_gradient(&[maximum], &[-maximum], &[2.0], 1).unwrap(),
                    [maximum]
                );
                let tiny = <$scalar>::from_bits(1);
                assert_eq!(
                    eval_objective(&[tiny], &[0.0], &[2.0], 1).unwrap(),
                    tiny * 2.0
                );
            }};
        }
        check!(f32);
        check!(f64);
    }

    fn default_solver() -> QpSolver<f64> {
        QpSolver::default()
    }

    fn problem<S>(hessian: Vec<S>, linear_objective: Vec<S>) -> QpProblem<S>
    where
        S: RealScalar + NumericScalar<Real = S>,
    {
        let dimension = linear_objective.len();
        QpProblem::new(
            Matrix::from_vec(dimension, dimension, hessian).unwrap(),
            linear_objective,
        )
        .unwrap()
    }

    fn constraints(
        coefficients: Vec<f64>,
        rhs: Vec<f64>,
        senses: Vec<LinearConstraintSense>,
    ) -> LinearConstraints<f64> {
        LinearConstraints::new(
            Matrix::from_vec(rhs.len(), 2, coefficients).unwrap(),
            rhs,
            senses,
        )
        .unwrap()
    }

    // ---- Unconstrained QP ----

    #[test]
    fn test_qp_unconstrained() {
        // Minimize 0.5 * x^T [[2,0],[0,2]] x + [-2,-6]^T x
        // Solution: x = [1, 3]
        let problem = problem(vec![2.0, 0.0, 0.0, 2.0], vec![-2.0, -6.0]);

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert_eq!(result.termination, OptimizeTermination::Converged);
        assert!((result.x[0] - 1.0).abs() < 1e-10);
        assert!((result.x[1] - 3.0).abs() < 1e-10);
        // f(1,3) = 0.5*(2+18) + (-2-18) = 10 - 20 = -10
        assert!((result.fun - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn indefinite_qp_is_rejected_before_claiming_convergence() {
        let hessian = Matrix::from_vec(1, 1, vec![-1.0_f64]).unwrap();
        let problem = QpProblem::new(hessian, vec![0.0]).unwrap();
        let result = QpSolver::default().solve(&problem);

        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
    }

    #[test]
    fn redundant_equalities_are_rank_reduced_before_kkt_solves() {
        let hessian = Matrix::identity(2).unwrap();
        let constraints = LinearConstraints::new(
            Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 2.0, 2.0]).unwrap(),
            vec![1.0, 2.0],
            vec![LinearConstraintSense::Equal; 2],
        )
        .unwrap();
        let problem = QpProblem::new(hessian, vec![0.0, 0.0])
            .unwrap()
            .with_constraints(constraints)
            .unwrap();

        let result = QpSolver::default().solve(&problem).unwrap();

        assert!(result.is_success(), "{}", result.message());
        assert!((result.x()[0] - 0.5).abs() < 1e-7);
        assert!((result.x()[1] - 0.5).abs() < 1e-7);
        assert!(result.stationarity().is_some_and(|value| value < 1e-7));
    }

    #[test]
    fn contradictory_equalities_are_not_reported_as_success() {
        let hessian = Matrix::identity(2).unwrap();
        let constraints = LinearConstraints::new(
            Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 1.0, 1.0]).unwrap(),
            vec![1.0, 2.0],
            vec![LinearConstraintSense::Equal; 2],
        )
        .unwrap();
        let problem = QpProblem::new(hessian, vec![0.0, 0.0])
            .unwrap()
            .with_constraints(constraints)
            .unwrap();

        let result = QpSolver::default().solve(&problem).unwrap();

        assert!(!result.is_success());
        assert!(result.primal_feasibility().is_some_and(|value| value > 0.0));
        let certificate = result.certificate().unwrap();
        let view = crate::QpVerificationProblem::from_problem(&problem);
        let verified = crate::ProgrammingVerifier::default()
            .verify(&view, certificate)
            .unwrap()
            .is_accepted();
        assert_eq!(
            result.verification_report().unwrap().is_accepted(),
            verified
        );
        assert_eq!(
            result.termination(),
            if verified {
                OptimizeTermination::Infeasible
            } else {
                OptimizeTermination::SuspectedInfeasible
            }
        );
        // Independently check the original Farkas equations, not only the
        // solver's cached verification: y^T A = 0 and y^T b < 0.
        if let crate::ProgrammingCertificate::Infeasibility(dual) = certificate {
            assert_eq!(dual.rows.len(), 2);
            if verified {
                assert!((dual.rows[0] + dual.rows[1]).abs() < 1e-12);
                assert!(dual.rows[0] + 2.0 * dual.rows[1] < -0.1);
                assert!(dual.lower.iter().chain(dual.upper).all(|v| v.abs() < 1e-12));
            }
        } else {
            assert!(matches!(
                certificate,
                crate::ProgrammingCertificate::Infeasibility(_)
            ));
        }
        // Proof construction may be inconclusive, but the exact original
        // contradiction (x+y=1 and x+y=2) has the dyadic witness (1,-1).
        for (rows, accepted) in [([1.0, -1.0], true), ([1.0, 1.0], false)] {
            let proof = crate::ProgrammingCertificate::Infeasibility(crate::DualCertificate {
                rows: &rows,
                lower: &[0.0; 2],
                upper: &[0.0; 2],
            });
            assert_eq!(
                crate::ProgrammingVerifier::default()
                    .verify(&view, proof)
                    .unwrap()
                    .is_accepted(),
                accepted
            );
        }
    }

    #[test]
    fn qp_phase_one_budget_exhaustion_is_not_hidden() {
        let hessian = Matrix::identity(2).unwrap();
        let constraints = LinearConstraints::new(
            Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 1.0, -1.0]).unwrap(),
            vec![1.0, 0.0],
            vec![LinearConstraintSense::Equal; 2],
        )
        .unwrap();
        let problem = QpProblem::new(hessian, vec![0.0, 0.0])
            .unwrap()
            .with_constraints(constraints)
            .unwrap();

        let result = QpSolver::default()
            .with_max_iterations(Budget::ONE)
            .solve(&problem)
            .unwrap();

        assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
        assert!(!result.is_success());
        assert!(result.message().contains("Phase I"));
    }

    #[test]
    fn test_qp_unconstrained_identity_hessian() {
        // Minimize 0.5 * x^T I x + [-3, -5]^T x
        // Solution: x = [3, 5]
        let problem = problem(vec![1.0, 0.0, 0.0, 1.0], vec![-3.0, -5.0]);

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.x[0] - 3.0).abs() < 1e-10);
        assert!((result.x[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_qp_unconstrained_3d() {
        // Minimize 0.5 * x^T diag(2,4,6) x + [-2,-8,-12]^T x
        // Solution: x_i = c_i / q_ii => x = [1, 2, 2]
        let problem = problem(
            vec![2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0],
            vec![-2.0, -8.0, -12.0],
        );

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.x[0] - 1.0).abs() < 1e-10);
        assert!((result.x[1] - 2.0).abs() < 1e-10);
        assert!((result.x[2] - 2.0).abs() < 1e-10);
    }

    // ---- Bound-constrained QP ----

    #[test]
    fn test_qp_with_bounds() {
        // Minimize 0.5 * x^T [[2,0],[0,2]] x + [-2,-6]^T x
        // Unconstrained solution: [1, 3]
        // With bounds 0 <= x <= 2: solution is [1, 2]
        let problem = problem(vec![2.0, 0.0, 0.0, 2.0], vec![-2.0, -6.0])
            .with_bounds(Bounds::new(vec![(Some(0.0), Some(2.0)), (Some(0.0), Some(2.0))]).unwrap())
            .unwrap();

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_qp_with_bounds_at_boundary() {
        // Minimize 0.5 * x^T [[2,0],[0,2]] x + [-10,-10]^T x
        // Unconstrained solution: [5, 5]
        // With bounds 0 <= x <= 3: solution is [3, 3]
        let problem = problem(vec![2.0, 0.0, 0.0, 2.0], vec![-10.0, -10.0])
            .with_bounds(Bounds::new(vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))]).unwrap())
            .unwrap();

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.x[0] - 3.0).abs() < 1e-6);
        assert!((result.x[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_qp_with_lower_bounds_only() {
        // Minimize 0.5 * x^T [[2,0],[0,2]] x + [2,2]^T x
        // Unconstrained solution: [-1, -1]
        // With x >= 0: solution is [0, 0]
        let problem = problem(vec![2.0, 0.0, 0.0, 2.0], vec![2.0, 2.0])
            .with_bounds(Bounds::new(vec![(Some(0.0), None), (Some(0.0), None)]).unwrap())
            .unwrap();

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!(result.x[0].abs() < 1e-6);
        assert!(result.x[1].abs() < 1e-6);
    }

    // ---- Inequality-constrained QP ----

    #[test]
    fn test_qp_with_inequality_constraints() {
        // Minimize 0.5 * x^T [[2,0],[0,2]] x + [-2,-6]^T x
        // Subject to: x0 + x1 <= 2, -x0 <= 0, -x1 <= 0
        // Unconstrained solution: [1, 3] violates x0 + x1 <= 2
        // Using KKT: x = [0, 2]
        let problem = problem(vec![2.0, 0.0, 0.0, 2.0], vec![-2.0, -6.0])
            .with_constraints(constraints(
                vec![1.0, 1.0, -1.0, 0.0, 0.0, -1.0],
                vec![2.0, 0.0, 0.0],
                vec![LinearConstraintSense::LessEqual; 3],
            ))
            .unwrap();

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!(
            result.x[0].abs() < 1e-4,
            "x[0] = {} (expected ~0)",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 1e-4,
            "x[1] = {} (expected ~2)",
            result.x[1]
        );
    }

    // ---- Equality-constrained QP ----

    #[test]
    fn test_qp_with_equality_constraint() {
        // Minimize 0.5 * x^T [[2,0],[0,2]] x
        // Subject to: x0 + x1 = 1
        // Solution: x = [0.5, 0.5]
        let problem = problem(vec![2.0, 0.0, 0.0, 2.0], vec![0.0, 0.0])
            .with_constraints(constraints(
                vec![1.0, 1.0],
                vec![1.0],
                vec![LinearConstraintSense::Equal],
            ))
            .unwrap();

        let solver = default_solver();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!(
            (result.x[0] - 0.5).abs() < 1e-6,
            "x[0] = {} (expected 0.5)",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.5).abs() < 1e-6,
            "x[1] = {} (expected 0.5)",
            result.x[1]
        );
    }

    // ---- Objective evaluation ----

    #[test]
    fn test_eval_objective_identity() {
        // 0.5 * x^T I x + 0^T x = 0.5 * (1 + 4) = 2.5
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let c = vec![0.0, 0.0];
        let x = vec![1.0, 2.0];
        let val = eval_objective(&q, &c, &x, 2).unwrap();
        assert!((val - 2.5_f64).abs() < 1e-12);
    }

    #[test]
    fn test_eval_objective_with_linear() {
        // 0.5 * [1,2]^T [[2,0],[0,2]] [1,2] + [-1,-1]^T [1,2]
        // = 0.5 * (2 + 8) + (-1 - 2) = 5 - 3 = 2
        let q = vec![2.0, 0.0, 0.0, 2.0];
        let c = vec![-1.0, -1.0];
        let x = vec![1.0, 2.0];
        let val = eval_objective(&q, &c, &x, 2).unwrap();
        assert!((val - 2.0_f64).abs() < 1e-12);
    }

    // ---- Gradient computation ----

    #[test]
    fn test_compute_gradient() {
        // grad = Qx + c = [[2,0],[0,2]][1,2] + [-2,-6] = [2,4] + [-2,-6] = [0,-2]
        let q = vec![2.0, 0.0, 0.0, 2.0];
        let c = vec![-2.0, -6.0];
        let x = vec![1.0, 2.0];
        let grad = compute_gradient(&q, &c, &x, 2).unwrap();
        assert!(f64::abs(grad[0]) < 1e-12);
        assert!((grad[1] - (-2.0_f64)).abs() < 1e-12);
    }

    // ---- f32 support ----

    #[test]
    fn test_qp_unconstrained_f32() {
        let problem = problem(vec![2.0_f32, 0.0, 0.0, 2.0], vec![-2.0_f32, -6.0]);

        let solver = QpSolver::<f32>::default();
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 3.0).abs() < 1e-4);
    }
}
