//! Powell's hybrid trust-region method for nonlinear systems.

#[cfg(test)]
use alloc::vec;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{Matrix, Vector};

use super::driver::{Done, Start, drive};
use super::options::SystemRootOptions;
use crate::context::Session;
use crate::nonlinear_math::{StepRounding, affine_into, norm};
use crate::residual::DenseJacobian;
use crate::result::{RootTermination, SystemRootResult};
use crate::system::{SystemWorkspace, accept_trial, evaluate_jacobian_into, residual_ok};

use super::dense::{gauss_newton_into, orthogonal_product_into, triangular_product_into};
use super::secant::{factor_secant_into, secant_update_into};

/// Numerical kernel of [`crate::TrustRegionSystem`]; storage lives in the workspace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrustRegionSystemKernel {
    derivative_strategy: DerivativeStrategy,
}

impl TrustRegionSystemKernel {
    /// Creates the kernel with the given Jacobian strategy.
    #[must_use]
    pub(crate) const fn new(derivative_strategy: DerivativeStrategy) -> Self {
        Self {
            derivative_strategy,
        }
    }
}

/// `0.1`, computed as `(1/5) · ½`.
fn tenth<S: RealScalar>() -> S {
    S::ONE / (S::TWO * S::TWO + S::ONE) * S::HALF
}

/// Trust-region state carried between trials.
struct Region<S> {
    /// Accepted residual norm.
    norm: S,
    /// Radius `Δ`.
    delta: S,
    /// Scaled accepted point norm `‖Dx‖`.
    point_norm: S,
    /// Whether any trial has been accepted.
    succeeded: bool,
    successes: usize,
    failures: usize,
    slow_trials: usize,
    slow_jacobians: usize,
    /// Whether the next iteration evaluates a fresh Jacobian.
    fresh: bool,
}

impl<S: RealScalar + NumericScalar<Real = S>> Region<S> {
    const fn new(norm: S) -> Self {
        Self {
            norm,
            delta: S::ZERO,
            point_norm: S::ZERO,
            succeeded: false,
            successes: 0,
            failures: 0,
            slow_trials: 0,
            slow_jacobians: 0,
            fresh: true,
        }
    }

    /// Update the radius from the reduction `ratio` of a trial of scaled length `step_norm`.
    fn resize(&mut self, ratio: S, step_norm: S) {
        let tenth = tenth::<S>();
        if ratio < tenth {
            self.successes = 0;
            self.failures += 1;
            self.delta *= S::HALF;
        } else {
            self.failures = 0;
            self.successes += 1;
            if ratio >= S::HALF || self.successes > 1 {
                self.delta = self.delta.max(step_norm * S::TWO);
            }
            if (ratio - S::ONE).abs() <= tenth {
                self.delta = step_norm * S::TWO;
            }
        }
    }

    /// Count trials without a 0.1% reduction of `‖F‖²` and fresh Jacobians without a 10% one.
    fn track(&mut self, actual: S) {
        let tenth = tenth::<S>();
        self.slow_trials = if actual >= tenth * tenth * tenth {
            0
        } else {
            self.slow_trials + 1
        };
        self.slow_jacobians = if actual >= tenth {
            0
        } else {
            self.slow_jacobians + usize::from(self.fresh)
        };
    }

    /// Residual convergence, or stagnation of the radius or of the reduction.
    fn termination(
        &self,
        opts: &SystemRootOptions<'_, S>,
        start: &Start<S>,
        step_norm: S,
    ) -> Result<Option<RootTermination>, SciOxideError> {
        let tenth = tenth::<S>();
        Ok(if residual_ok(&start.tolerance, self.norm, start.scale)? {
            Some(RootTermination::ResidualTolerance)
        } else if opts
            .step_tolerance
            .is_satisfied(self.delta, self.point_norm)?
            || tenth * (tenth * self.delta).max(step_norm) <= S::EPSILON * self.point_norm
            || self.slow_jacobians == 5
            || self.slow_trials == 10
        {
            Some(RootTermination::Stagnation)
        } else {
            None
        })
    }
}

/// Actual and predicted relative reductions of `‖F‖²` and their ratio.
fn reduction<S: RealScalar>(norm: S, trial_norm: S, model_norm: S) -> (S, S) {
    let actual = if trial_norm < norm {
        S::ONE - (trial_norm / norm) * (trial_norm / norm)
    } else {
        -S::ONE
    };
    let predicted = if model_norm < norm {
        S::ONE - (model_norm / norm) * (model_norm / norm)
    } else {
        S::ZERO
    };
    let ratio = if predicted > S::ZERO {
        actual / predicted
    } else {
        S::ZERO
    };
    (actual, ratio)
}

/// An evaluated dogleg trial, or the termination that ended the search.
enum Trial<S> {
    Evaluated { step_norm: S, trial_norm: S },
    Stop(RootTermination),
}

impl TrustRegionSystemKernel {
    pub(crate) fn solve_with_context<S, F>(
        &self,
        f: &F,
        initial: &[S],
        opts: &SystemRootOptions<'_, S>,
        result: &mut SystemRootResult<S>,
        workspace: &mut SystemWorkspace<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: DenseJacobian<S> + ?Sized,
    {
        drive(
            f,
            initial,
            opts,
            result,
            workspace,
            context,
            |workspace, session, start| self.iterate(f, opts, workspace, session, start),
        )
    }

    /// Dogleg iterations after an unconverged initial residual.
    fn iterate<S, F>(
        &self,
        f: &F,
        opts: &SystemRootOptions<'_, S>,
        workspace: &mut SystemWorkspace<S>,
        session: &Session<'_, '_, '_>,
        start: Start<S>,
    ) -> Result<Done<S>, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: DenseJacobian<S> + ?Sized,
    {
        let mut region = Region::new(start.norm);
        for _ in 0..opts.max_iterations.get() {
            session.iteration()?;
            if region.fresh {
                self.fresh_jacobian(f, opts, workspace, session, &mut region)?;
            }
            gauss_newton(workspace, session)?;
            let (step_norm, trial_norm) =
                match dogleg_trial(f, opts, workspace, session, &mut region)? {
                    Trial::Evaluated {
                        step_norm,
                        trial_norm,
                    } => (step_norm, trial_norm),
                    Trial::Stop(termination) => return Ok(Done::new(region.norm, termination)),
                };
            let model_norm = model_residual(workspace, session, step_norm)?;
            let (actual, ratio) = reduction(region.norm, trial_norm, model_norm);
            region.resize(ratio, step_norm);
            let tenth = tenth::<S>();
            if ratio >= tenth * tenth * tenth * tenth {
                accept_trial(&mut workspace.state, session)?;
                region.norm = trial_norm;
                region.point_norm = scaled_norm(
                    &workspace.diag,
                    &workspace.state.current_point,
                    &mut workspace.scaled,
                )?;
                region.succeeded = true;
            } else {
                session.reject()?;
            }
            region.track(actual);
            if let Some(termination) = region.termination(opts, &start, step_norm)? {
                return Ok(Done::new(region.norm, termination));
            }
            region.fresh = region.failures == 2;
            if !region.fresh {
                secant_update(opts, workspace, session, step_norm)?;
            }
        }
        Ok(Done::new(region.norm, RootTermination::IterationLimit))
    }

    /// Evaluate and factor a fresh Jacobian, updating the scaling; before the
    /// first success also reset the radius to `100‖Dx‖` (or `100‖F‖` at `Dx = 0`).
    fn fresh_jacobian<S, F>(
        &self,
        f: &F,
        opts: &SystemRootOptions<'_, S>,
        workspace: &mut SystemWorkspace<S>,
        session: &Session<'_, '_, '_>,
        region: &mut Region<S>,
    ) -> Result<(), SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: DenseJacobian<S> + ?Sized,
    {
        evaluate_jacobian_into(
            f,
            &workspace.state.current_point,
            &workspace.state.current_residual,
            self.derivative_strategy,
            &mut workspace.jacobian,
            &mut workspace.candidate_jacobian,
            &mut workspace.state.scratch,
            workspace.finite_difference.as_mut(),
            session,
        )?;
        update_scaling(&workspace.jacobian, &mut workspace.diag, !region.succeeded)?;
        if !region.succeeded {
            region.point_norm = scaled_norm(
                &workspace.diag,
                &workspace.state.current_point,
                &mut workspace.scaled,
            )?;
            let hundred = (S::ONE / tenth::<S>()) * (S::ONE / tenth::<S>());
            region.delta = hundred
                * if region.point_norm == S::ZERO {
                    region.norm
                } else {
                    region.point_norm
                };
        }
        session.admit(scioxide_core::EvaluationRole::Factorization)?;
        factor_secant_into(
            &workspace.jacobian,
            &mut workspace.secant,
            &mut workspace.secant_workspace,
            opts.policy,
            session.execution()?,
        )
    }
}

/// `QᵀF` and the Gauss–Newton point of the factored model.
fn gauss_newton<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
    orthogonal_product_into(
        workspace.secant.q_matrix(),
        true,
        &workspace.state.current_residual,
        &mut workspace.qtf,
    );
    session.admit(scioxide_core::EvaluationRole::LinearSolve)?;
    gauss_newton_into(
        workspace.secant.r_matrix(),
        &workspace.qtf,
        &mut workspace.gauss_newton,
    );
    Ok(())
}

/// Evaluate dogleg trials, halving the radius after each rejection by the
/// residual's domain, until a finite residual is obtained.
fn dogleg_trial<S, F>(
    f: &F,
    opts: &SystemRootOptions<'_, S>,
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
    region: &mut Region<S>,
) -> Result<Trial<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    let mut domain_rejected = false;
    loop {
        dogleg_into(
            workspace.secant.r_matrix(),
            &workspace.qtf,
            &workspace.diag,
            region.delta,
            &workspace.gauss_newton,
            &mut workspace.gradient,
            &mut workspace.product,
            &mut workspace.scaled,
            &mut workspace.step,
        )?;
        let step_norm = scaled_norm(&workspace.diag, &workspace.step, &mut workspace.scaled)?;
        if !region.succeeded {
            region.delta = region.delta.min(step_norm);
        }
        affine_into(
            workspace.state.trial_point.as_mut_slice(),
            workspace.state.current_point.as_slice(),
            S::ONE,
            workspace.step.as_slice(),
            StepRounding::Fused,
        )?;
        if workspace.state.trial_point.as_slice() == workspace.state.current_point.as_slice() {
            return Ok(Trial::Stop(if domain_rejected {
                RootTermination::GlobalizationFailure
            } else {
                RootTermination::Stagnation
            }));
        }
        let status = super::evaluate_trial_into(
            f,
            &workspace.state.trial_point,
            &mut workspace.state.trial_residual,
            &mut workspace.line_search_residual,
            &mut workspace.state.scratch,
            session,
        )?;
        if status == crate::residual::TrialStatus::Finite {
            return Ok(Trial::Evaluated {
                step_norm,
                trial_norm: norm(workspace.state.trial_residual.as_slice())?,
            });
        }
        session.reject()?;
        domain_rejected = true;
        region.delta *= S::HALF;
        if region.delta == S::ZERO
            || opts
                .step_tolerance
                .is_satisfied(region.delta, region.point_norm)?
        {
            return Ok(Trial::Stop(RootTermination::GlobalizationFailure));
        }
        session.proposal()?;
    }
}

/// Rotated linear model `Qᵀ(F + Jp) = QᵀF + Rp`, its norm, and the scaled secant
/// correction `(ΔF − Jp)/‖Dp‖ = (F(x + p) − Q·Qᵀ(F + Jp))/‖Dp‖`.
fn model_residual<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
    step_norm: S,
) -> Result<S, SciOxideError> {
    session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
    triangular_product_into(
        workspace.secant.r_matrix(),
        &workspace.step,
        &mut workspace.model,
    );
    for (model, &qtf) in workspace
        .model
        .as_mut_slice()
        .iter_mut()
        .zip(workspace.qtf.as_slice())
    {
        *model += qtf;
    }
    let model_norm = norm(workspace.model.as_slice())?;
    session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
    orthogonal_product_into(
        workspace.secant.q_matrix(),
        false,
        &workspace.model,
        &mut workspace.product,
    );
    for ((difference, &trial), &model) in workspace
        .difference
        .as_mut_slice()
        .iter_mut()
        .zip(workspace.state.trial_residual.as_slice())
        .zip(workspace.product.as_slice())
    {
        *difference = (trial - model) / step_norm;
    }
    Ok(model_norm)
}

/// Scaled Broyden update `J ← J + (ΔF − Jp)(D²p)ᵀ/‖Dp‖²` of the secant factors.
fn secant_update<S: RealScalar + NumericScalar<Real = S>>(
    opts: &SystemRootOptions<'_, S>,
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
    step_norm: S,
) -> Result<(), SciOxideError> {
    for ((target, &d), &p) in workspace
        .gradient
        .as_mut_slice()
        .iter_mut()
        .zip(workspace.diag.as_slice())
        .zip(workspace.step.as_slice())
    {
        *target = d * ((d * p) / step_norm);
    }
    secant_update_into(
        &mut workspace.secant,
        &mut workspace.secant_workspace,
        &workspace.difference,
        &workspace.gradient,
        opts.policy,
        session.execution()?,
    )
}

/// `‖D v‖`, using `scratch` for the scaled vector.
fn scaled_norm<S: RealScalar + NumericScalar<Real = S>>(
    diag: &Vector<S>,
    vector: &Vector<S>,
    scratch: &mut Vector<S>,
) -> Result<S, SciOxideError> {
    for ((target, &d), &v) in scratch
        .as_mut_slice()
        .iter_mut()
        .zip(diag.as_slice())
        .zip(vector.as_slice())
    {
        *target = d * v;
    }
    norm(scratch.as_slice())
}

/// Variable scaling `D`: the column norms of the first Jacobian (one for a zero
/// column), then the componentwise maximum with every later Jacobian.
fn update_scaling<S: RealScalar + NumericScalar<Real = S>>(
    jacobian: &Matrix<S>,
    diag: &mut Vector<S>,
    reset: bool,
) -> Result<(), SciOxideError> {
    for (column, target) in diag.as_mut_slice().iter_mut().enumerate() {
        let column_norm = crate::nonlinear_math::norm_iter(
            jacobian
                .as_slice()
                .chunks_exact(jacobian.cols())
                .map(|row| row[column]),
        )?;
        *target = if reset {
            if column_norm == <S as scioxide_core::AlgebraicScalar>::ZERO {
                <S as scioxide_core::AlgebraicScalar>::ONE
            } else {
                column_norm
            }
        } else {
            target.max(column_norm)
        };
    }
    Ok(())
}

/// Dogleg step: the step `p` minimizing `‖QᵀF + Rp‖` on the dogleg path
/// between the scaled steepest-descent minimizer and the Gauss–Newton step `−x`,
/// subject to `‖Dp‖ ≤ Δ`. A Gauss–Newton step that overflows leaves the scaled
/// steepest-descent step.
#[expect(
    clippy::too_many_arguments,
    reason = "Numerical execution borrows separate caller-owned buffers and solve state."
)]
fn dogleg_into<S: RealScalar + NumericScalar<Real = S>>(
    r: &Matrix<S>,
    qtf: &Vector<S>,
    diag: &Vector<S>,
    delta: S,
    gauss_newton: &Vector<S>,
    gradient: &mut Vector<S>,
    product: &mut Vector<S>,
    scratch: &mut Vector<S>,
    step: &mut Vector<S>,
) -> Result<(), SciOxideError> {
    let one: S = <S as scioxide_core::AlgebraicScalar>::ONE;
    let zero: S = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let newton_norm = scaled_norm(diag, gauss_newton, scratch)?;
    if newton_norm <= delta {
        for (target, &value) in step.as_mut_slice().iter_mut().zip(gauss_newton.as_slice()) {
            *target = -value;
        }
        return Ok(());
    }
    // Scaled gradient `D⁻¹RᵀQᵀF = D⁻¹JᵀF`.
    super::dense::upper_transpose_into(
        r.as_slice(),
        qtf.as_slice(),
        Some(diag.as_slice()),
        gradient.as_mut_slice(),
    );
    let gradient_norm = norm(gradient.as_slice())?;
    let mut gradient_length = zero;
    let mut alpha = if newton_norm.is_finite() {
        delta / newton_norm
    } else {
        zero
    };
    if gradient_norm != zero {
        for (value, &d) in gradient.as_mut_slice().iter_mut().zip(diag.as_slice()) {
            *value = (*value / gradient_norm) / d;
        }
        triangular_product_into(r, gradient, product);
        let curvature = norm(product.as_slice())?;
        if curvature == zero {
            return Err(model_error("nonzero gradient has zero model curvature"));
        }
        gradient_length = (gradient_norm / curvature) / curvature;
        alpha = zero;
        if gradient_length < delta && newton_norm.is_finite() {
            let residual_norm = norm(qtf.as_slice())?;
            let quotient = delta / newton_norm;
            let fraction = gradient_length / delta;
            let mut temp =
                (residual_norm / gradient_norm) * (residual_norm / newton_norm) * fraction;
            temp = temp - quotient * fraction * fraction
                + ((temp - quotient) * (temp - quotient)
                    + (one - quotient * quotient) * (one - fraction * fraction))
                    .sqrt();
            alpha = (quotient * (one - fraction * fraction)) / temp;
        }
    }
    let gradient_weight = (one - alpha) * gradient_length.min(delta);
    for ((target, &g), &x) in step
        .as_mut_slice()
        .iter_mut()
        .zip(gradient.as_slice())
        .zip(gauss_newton.as_slice())
    {
        *target = if alpha == zero {
            -(gradient_weight * g)
        } else {
            -(gradient_weight * g + alpha * x)
        };
    }
    if step.as_slice().iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(model_error("dogleg step is not finite"))
    }
}

/// Inconsistent trust-region model.
fn model_error(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::IllConditioned, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SystemRootSolver, TrustRegionSystem};

    struct TwoDSystem;
    impl scioxide_core::ResidualOperator<f64> for TwoDSystem {
        fn input_dimension(&self) -> usize {
            2
        }
        fn output_dimension(&self) -> usize {
            2
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<(), SciOxideError> {
            let x0 = x[0];
            let x1 = x[1];
            out[0] = x0.mul_add(x0, x1 - 1.0);
            out[1] = x1.mul_add(x1, x0 - 1.0);
            Ok(())
        }
    }
    impl crate::DenseJacobian<f64> for TwoDSystem {}

    /// Dogleg step for an upper-triangular `R` with `Q = I`.
    fn dogleg(r: [f64; 4], residual: [f64; 2], diag: [f64; 2], delta: f64) -> Vector<f64> {
        let r = Matrix::from_vec(2, 2, r.to_vec()).unwrap();
        let qtf = Vector::from_vec(residual.to_vec());
        let diag = Vector::from_vec(diag.to_vec());
        let mut gauss_newton = Vector::zeros(2).unwrap();
        gauss_newton_into(&r, &qtf, &mut gauss_newton);
        let mut scratch = [
            Vector::zeros(2).unwrap(),
            Vector::zeros(2).unwrap(),
            Vector::zeros(2).unwrap(),
        ];
        let [gradient, product, scaled] = &mut scratch;
        let mut step = Vector::zeros(2).unwrap();
        dogleg_into(
            &r,
            &qtf,
            &diag,
            delta,
            &gauss_newton,
            gradient,
            product,
            scaled,
            &mut step,
        )
        .unwrap();
        step
    }

    #[test]
    fn gauss_newton_replaces_zero_pivots_by_scaled_epsilon() {
        let mut solution = Vector::zeros(2).unwrap();
        gauss_newton_into(
            &Matrix::from_vec(2, 2, vec![2.0, 4.0, 0.0, 0.0]).unwrap(),
            &Vector::from_vec(vec![2.0, 1.0]),
            &mut solution,
        );
        let expected = 1.0 / (4.0 * f64::EPSILON);
        assert_eq!(solution.as_slice()[1], expected);
        assert_eq!(solution.as_slice()[0], (2.0 - 4.0 * expected) / 2.0);
    }

    #[test]
    fn dogleg_takes_gauss_newton_inside_the_radius() {
        let step = dogleg([2.0, 0.0, 0.0, 4.0], [2.0, 4.0], [1.0, 1.0], 10.0);
        assert_eq!(step.as_slice(), &[-1.0, -1.0]);
    }

    #[test]
    fn dogleg_steps_reach_the_scaled_radius_on_both_legs() {
        for scale in [1e-150, 1.0, 1e150] {
            let jacobian = [2.0 * scale, 0.0, 0.0, 4.0 * scale];
            let diag = [2.0 * scale, 4.0 * scale];
            // Steepest-descent leg: the radius is shorter than the Cauchy point.
            for delta in [0.01, 0.5, 1.2] {
                let step = dogleg(jacobian, [2.0, 4.0], diag, delta);
                let scaled = (step.as_slice()[0] * diag[0]).hypot(step.as_slice()[1] * diag[1]);
                assert!(
                    (scaled / delta - 1.0).abs() < 1e-12,
                    "{scale} {delta} {scaled}"
                );
                assert!(step.as_slice().iter().all(|&v| v < 0.0));
            }
        }
    }

    #[test]
    fn scaling_starts_from_jacobian_columns_and_only_grows() {
        let mut diag = Vector::zeros(2).unwrap();
        update_scaling(
            &Matrix::from_vec(2, 2, vec![3.0, 0.0, 4.0, 0.0]).unwrap(),
            &mut diag,
            true,
        )
        .unwrap();
        assert_eq!(diag.as_slice(), &[5.0, 1.0]);
        update_scaling(
            &Matrix::from_vec(2, 2, vec![1.0, 2.0, 0.0, 0.0]).unwrap(),
            &mut diag,
            false,
        )
        .unwrap();
        assert_eq!(diag.as_slice(), &[5.0, 2.0]);
    }

    fn himmelblau(scale: [f64; 2]) -> SystemRootResult<f64> {
        let function = crate::ResidualFn::new(2, move |y: &[f64], out: &mut [f64]| {
            let (a, b) = (scale[0] * y[0], scale[1] * y[1]);
            out.copy_from_slice(&[a * a + b - 11.0, a + b * b - 7.0]);
            Ok(())
        })
        .with_jacobian(move |y: &[f64], out: &mut [f64]| {
            let (a, b) = (scale[0] * y[0], scale[1] * y[1]);
            out.copy_from_slice(&[2.0 * a * scale[0], scale[1], scale[0], 2.0 * b * scale[1]]);
            Ok(())
        });
        TrustRegionSystem::new(
            (crate::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap()
        .solve(&function, &[-1.0 / scale[0], -1.0 / scale[1]])
        .unwrap()
    }

    #[test]
    fn hybrid_steps_are_invariant_under_variable_scaling() {
        let reference = himmelblau([1.0, 1.0]);
        let scale = [2.0_f64.powi(-20), 2.0_f64.powi(20)];
        let scaled = himmelblau(scale);
        assert!(reference.is_success() && scaled.is_success());
        assert_eq!(reference.iterations(), scaled.iterations());
        assert_eq!(reference.counts(), scaled.counts());
        for ((&x, &y), &c) in reference.point().iter().zip(scaled.point()).zip(&scale) {
            assert_eq!(x, y * c);
        }
        // Secant corrections replace most Jacobian evaluations.
        assert!(
            reference
                .counts()
                .provider(scioxide_core::EvaluationRole::Jacobian)
                < reference.iterations()
        );
    }

    #[test]
    fn test_trust_region_2d() {
        let solver = TrustRegionSystem::new(
            (crate::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
        )
        .unwrap();

        let result = solver.solve(&TwoDSystem, &[0.5, 0.5]).unwrap();
        assert!(result.is_success());
        assert!(result.residual_norm().unwrap() < 1e-6);
    }

    #[test]
    fn test_trust_region_name() {
        let solver = TrustRegionSystem::new(
            (crate::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
        )
        .unwrap();
        assert_eq!(SystemRootSolver::<f64>::name(&solver), "TrustRegionSystem");
    }

    #[test]
    fn test_trust_region_far_start() {
        let solver = TrustRegionSystem::new(
            (crate::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
        )
        .unwrap();

        let result = solver.solve(&TwoDSystem, &[5.0, 5.0]).unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_trust_region_rejects_dimension_mismatch() {
        let solver = TrustRegionSystem::new(crate::TrustRegionSystemOptions::default()).unwrap();
        let result = solver.solve(&TwoDSystem, &[0.5]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn test_trust_region_rejects_trials_for_inconsistent_singular_model() {
        struct SingularSystem;
        impl scioxide_core::ResidualOperator<f64> for SingularSystem {
            fn input_dimension(&self) -> usize {
                2
            }
            fn output_dimension(&self) -> usize {
                2
            }
            fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn residual_into(
                &self,
                _x: &[f64],
                out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<(), SciOxideError> {
                out.fill(1.0);
                Ok(())
            }
        }
        impl crate::DenseJacobian<f64> for SingularSystem {
            fn jacobian_into(
                &self,
                _x: &[f64],
                out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<bool, SciOxideError> {
                out.copy_from_slice(&[1.0, 2.0, 2.0, 4.0]);
                Ok(true)
            }
        }

        let solver = TrustRegionSystem::new(
            (crate::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();

        let result = solver.solve(&SingularSystem, &[0.0, 0.0]);
        assert_eq!(result.unwrap().termination(), RootTermination::Stagnation);
    }
}
