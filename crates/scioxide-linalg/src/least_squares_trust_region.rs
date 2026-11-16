//! Moré's safeguarded search for the Levenberg–Marquardt parameter of a
//! least-squares trust-region step, from a QR or SVD factorization.
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

/// Result of a least-squares trust-region solve: status, damping `λ` and step norm.
///
/// Returned by the trust-region solves of
/// [`HouseholderDecomposition`](crate::HouseholderDecomposition::solve_trust_region_into)
/// (norm `‖D p‖₂`) and
/// [`ThinSvdDecomposition`](crate::ThinSvdDecomposition::solve_trust_region_into)
/// (norm `‖p‖₂`). The search stops when the step norm is within 10% of the
/// radius, when it cannot improve further, or after 10 iterations. A step with
/// `λ = 0` has norm at most 1.1 times the radius.
#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresTrustRegionReport<S: RealScalar> {
    solve: crate::DirectSolveReport<S>,
    step: Option<TrustRegionStep<S>>,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrustRegionStep<S: RealScalar> {
    pub(crate) damping: S,
    pub(crate) norm: S,
}
impl<S: RealScalar> LeastSquaresTrustRegionReport<S> {
    pub(crate) const fn new(
        solve: crate::DirectSolveReport<S>,
        step: Option<TrustRegionStep<S>>,
    ) -> Self {
        Self { solve, step }
    }
    /// Returns whether the solve was accepted, or why it was rejected.
    pub const fn termination(self) -> crate::DirectSolveTermination {
        self.solve.termination()
    }
    /// Returns the status with its rank, condition and residual diagnostics.
    pub const fn solve_report(self) -> crate::DirectSolveReport<S> {
        self.solve
    }
    /// Returns the Levenberg–Marquardt parameter `λ ≥ 0` of the step, or `None`
    /// if the solve was rejected.
    pub const fn damping(self) -> Option<S> {
        match self.step {
            Some(step) => Some(step.damping),
            None => None,
        }
    }
    /// Returns the norm of the step in the trust-region norm, or `None` if the
    /// solve was rejected.
    pub const fn step_norm(self) -> Option<S> {
        match self.step {
            Some(step) => Some(step.norm),
            None => None,
        }
    }
}

pub(crate) fn validate<S: RealScalar>(radius: S, hint: S) -> Result<(), SciOxideError> {
    if !radius.is_finite() || radius <= S::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("radius"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "radius must be finite and positive",
        ));
    }
    if !hint.is_finite() || hint < S::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("hint"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "damping hint must be finite and nonnegative",
        ));
    }
    Ok(())
}

/// A query from [`select`] to the solver that forms the damped steps `p(λ)`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StepQuery<S> {
    /// Form `p(λ)` for this `λ` and return `‖D p(λ)‖₂`.
    Norm(S),
    /// Return `q = ‖R_λ⁻ᵀ Dᵀ D p / ‖D p‖₂‖₂` for the step of the last `Norm`
    /// query, given its `λ` and `‖D p‖₂`, where `R_λᵀ R_λ = AᵀA + λ DᵀD`. Then
    /// `d‖D p‖₂/dλ = −‖D p‖₂ q²`.
    NewtonScale { damping: S, norm: S },
}

fn without_range<S: RealScalar>(
    result: Result<S, SciOxideError>,
) -> Result<Option<S>, SciOxideError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error)
            if matches!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::NumericalInstability
                    | scioxide_core::SciOxideErrorKind::NonFinite
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Returns the Newton correction `((φ/Δ)/q)/q` at `λ = damping`, or `None` if
/// `q` or the correction is outside the scalar range.
fn newton_correction<S: RealScalar>(
    evaluate: &mut impl FnMut(StepQuery<S>) -> Result<S, SciOxideError>,
    radius: S,
    damping: S,
    norm: S,
) -> Result<Option<S>, SciOxideError> {
    let scale = without_range(evaluate(StepQuery::NewtonScale { damping, norm }))?;
    Ok(scale.and_then(|scale| {
        let correction = (((norm - radius) / radius) / scale) / scale;
        (scale.is_finite() && correction.is_finite()).then_some(correction)
    }))
}

/// Chooses `λ` by Moré’s safeguarded iteration.
///
/// With `φ(λ) = ‖D p(λ)‖₂ − Δ`, the Newton correction for `1/‖D p‖₂` is
/// `((φ/Δ)/q)/q`. The Gauss–Newton step gives the lower bound `l` by that
/// correction, `u = ‖D⁻¹ Aᵀ b‖₂ / Δ` is an upper bound, and each iterate
/// replaces `l` or `u` and moves to `max(l, λ + correction)`. A `q` outside the
/// scalar range gives no correction; the iterate then moves to Moré's safeguard
/// `max(u/1000, √(l u))`.
pub(crate) fn select<S: RealScalar>(
    radius: S,
    hint: S,
    gradient: impl FnOnce() -> Result<S, SciOxideError>,
    mut evaluate: impl FnMut(StepQuery<S>) -> Result<S, SciOxideError>,
) -> Result<TrustRegionStep<S>, SciOxideError> {
    validate(radius, hint)?;
    let tolerance = radius / S::TEN;
    let thousandth = S::ONE / (S::TEN * S::TEN * S::TEN);
    // A Gauss–Newton step outside the range has no Newton bound.
    let gauss_newton = without_range(evaluate(StepQuery::Norm(S::ZERO)))?;
    let norm = gauss_newton.unwrap_or(S::MAX);
    let mut difference = norm - radius;
    if difference <= tolerance {
        return Ok(TrustRegionStep {
            damping: S::ZERO,
            norm,
        });
    }
    let mut lower = match gauss_newton {
        Some(norm) => newton_correction(&mut evaluate, radius, S::ZERO, norm)?
            .map_or(S::ZERO, |value| value.max(S::ZERO)),
        None => S::ZERO,
    };
    let gradient = gradient()?;
    let mut upper = (gradient / radius).min(S::MAX);
    if upper == S::ZERO {
        upper = S::MIN_POSITIVE / radius.min(S::ONE / S::TEN);
    }
    let mut damping = hint.max(lower).min(upper);
    if damping == S::ZERO {
        damping = gradient / norm;
    }
    let mut iteration = 0;
    loop {
        iteration += 1;
        if damping == S::ZERO {
            damping = S::MIN_POSITIVE.max(thousandth * upper);
        }
        // A step norm outside the range is an error once λ > 0.
        let norm = evaluate(StepQuery::Norm(damping))?;
        let previous = difference;
        difference = norm - radius;
        if difference.abs() <= tolerance
            || (lower == S::ZERO && difference <= previous && previous < S::ZERO)
            || iteration == 10
        {
            return Ok(TrustRegionStep { damping, norm });
        }
        let correction = newton_correction(&mut evaluate, radius, damping, norm)?;
        if difference > S::ZERO {
            lower = lower.max(damping);
        } else {
            upper = upper.min(damping);
        }
        damping = match correction.map(|correction| damping + correction) {
            Some(candidate) if candidate.is_finite() => lower.max(candidate),
            _ => (thousandth * upper).max(lower.sqrt() * upper.sqrt()),
        };
    }
}

/// Use ordinary spectral filters until an intermediate loses scalar range.
fn coefficient<S: RealScalar + NumericScalar<Real = S>>(
    value: S,
    singular: S,
    damping: S,
) -> Result<S, SciOxideError> {
    let square = singular * singular;
    let numerator = value * singular;
    let denominator = square + damping;
    let result = numerator / denominator;
    if !scioxide_core::backend::standard::product_needs_retry(singular, singular, square)
        && !scioxide_core::backend::standard::product_needs_retry(value, singular, numerator)
        && denominator.is_finite()
        && result.is_finite()
        && (value == S::ZERO || result.abs() >= S::MIN_POSITIVE)
    {
        return Ok(result);
    }
    crate::decompose::filtered_svd_coefficient(value, 0, singular, damping)
}

impl<S: RealScalar + NumericScalar<Real = S>> crate::ThinSvdDecomposition<S> {
    /// Solves the trust-region subproblem `min ‖A p − b‖₂` subject to
    /// `‖p‖₂ ≤ radius`, restricted to the singular vectors kept by `rank`, into `output`.
    ///
    /// Singular values at or below the rank threshold are dropped. With
    /// `A = U Σ Vᵀ`, the step is `p(λ) = Σ_i σ_i (u_iᵀ b) / (σ_i² + λ) v_i`, and
    /// `λ` is chosen by the safeguarded iteration described on the
    /// [QR trust-region solve](crate::HouseholderDecomposition::solve_trust_region_into).
    /// The solve is always accepted; the report carries the rank but no
    /// condition estimate or residual. The scratch is that of
    /// [`SvdWorkspaceRequest::solve`](crate::SvdWorkspaceRequest::solve).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `radius` is not finite and positive,
    ///   or `hint` is not finite and nonnegative.
    /// - `DimensionMismatch` if `rhs` does not have length `m` or `output` does
    ///   not have length `n`.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if an entry of the step overflows.
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_trust_region_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        radius: S,
        hint: S,
        rank: crate::RankPolicy<S>,
        options: &crate::LinalgExecutionOptions,
        mut output: crate::VectorViewMut<'_, S>,
        workspace: &mut crate::LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<LeastSquaresTrustRegionReport<S>, SciOxideError> {
        workspace.begin_attempt();
        validate(radius, hint)?;
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let rhs = rhs.into();
        crate::checked::expect_shape([self.rows()], [rhs.len()])?;
        crate::checked::expect_shape([self.cols()], [output.len()])?;
        let rhs = rhs.try_finite_dispatched(context.kernel().dispatch())?;
        let rank = self.analyze_rank(rank)?;
        let threshold = rank.threshold();
        workspace.try_reserve(&Self::workspace_requirements(
            &crate::SvdWorkspaceRequest::solve(self.rows(), self.cols(), options),
        )?)?;
        workspace.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "svd_trust_region", 1),
                self.rows().max(self.cols()) as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, kernels, solve| {
                crate::solve_support::admit_iteration(
                    Some(kernels.execution()),
                    Some(solve),
                    "SVD trust-region budget exhausted",
                )?;
                workspace.with_direct(|storage, _, _, _| {
                    let k = self.singular_values().len();
                    let (coefficients, candidate) = crate::workspace::split_mut(storage, k)?;
                    let candidate = &mut candidate[..self.cols()];
                    for (i, (&singular, coefficient)) in self
                        .singular_values()
                        .iter()
                        .zip(coefficients.iter_mut())
                        .enumerate()
                    {
                        let column =
                            crate::FiniteVectorView::from_workspace(self.u_matrix().view().col(i)?);
                        let (dot, exponent) =
                            crate::reduction::compensated_dot(column, rhs, S::ZERO, S::ONE)?;
                        *coefficient = if singular > threshold {
                            crate::kernel::standard::restore(dot, exponent)?
                        } else {
                            S::ZERO
                        };
                    }
                    let step = select(
                        radius,
                        hint,
                        || {
                            let mut gradient = scioxide_core::ScaledNorm::new();
                            for (&singular, &coefficient) in
                                self.singular_values().iter().zip(coefficients.iter())
                            {
                                gradient.add(singular * coefficient)?;
                            }
                            gradient.finish()
                        },
                        |query| {
                            kernels.execution().check()?;
                            let (damping, step_norm) = match query {
                                StepQuery::Norm(damping) => (damping, None),
                                StepQuery::NewtonScale { damping, norm } => (damping, Some(norm)),
                            };
                            // ‖p‖₂, or q = ‖(AᵀA + λI)^{-1/2} p‖₂ / ‖p‖₂ with
                            // components (vᵢ/‖p‖₂)/√(σᵢ² + λ) for p = Σ vᵢ xᵢ.
                            let root = damping.sqrt();
                            let mut norm = scioxide_core::ScaledNorm::new();
                            for (&singular, &coefficient) in
                                self.singular_values().iter().zip(coefficients.iter())
                            {
                                if singular <= threshold {
                                    continue;
                                }
                                let value = self::coefficient(coefficient, singular, damping)?;
                                norm.add(match step_norm {
                                    None => value,
                                    Some(step_norm) => (value / step_norm) / singular.hypot(root),
                                })?;
                            }
                            norm.finish()
                        },
                    )?;
                    for (column, value) in candidate.iter_mut().enumerate() {
                        kernels.execution().check()?;
                        let mut sum = scioxide_core::RangeAccumulator::new();
                        for (i, (&singular, &coefficient)) in self
                            .singular_values()
                            .iter()
                            .zip(coefficients.iter())
                            .enumerate()
                        {
                            if singular > threshold {
                                let factor =
                                    self::coefficient(coefficient, singular, step.damping)?;
                                sum.add_product(
                                    self.vt_matrix().as_slice()[i * self.cols() + column],
                                    factor,
                                )?;
                            }
                        }
                        *value = sum.finish()?;
                    }
                    kernels.execution().check()?;
                    output
                        .iter_mut()
                        .zip(candidate.iter())
                        .for_each(|(out, &value)| *out = value);
                    Ok(LeastSquaresTrustRegionReport::new(
                        crate::DirectSolveReport::new(
                            crate::DirectSolveTermination::Accepted,
                            rank,
                            None,
                            None,
                        ),
                        Some(step),
                    ))
                })
            },
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::*;
    use scioxide_core::ExecutionContext;

    #[test]
    fn qr_diagonal_trust_region_satisfies_kkt_and_undamped_limit() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let options = LinalgExecutionOptions::new();
        let factor = HouseholderDecomposition::factor(
            &a,
            HouseholderOrientation::Qr,
            HouseholderPivoting::Column,
        )
        .unwrap();
        let request = HouseholderWorkspaceRequest::damped_solve(2, 2, &options);
        let mut workspace = HouseholderDecomposition::workspace(&request).unwrap();
        for (diagonal, radius) in [([1.0, 1.0], 1.0), ([2.0, 0.5], 1.0), ([1.0, 1.0], 10.0)] {
            let mut output = [0.0; 2];
            let report = factor
                .solve_trust_region_into(
                    &[3.0, 4.0][..],
                    &diagonal,
                    radius,
                    1e300,
                    &DirectSolveOptions::new(RankPolicy::Exact).with_residual(Estimation::Estimate),
                    VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.solve_report().residual().is_some());
            let norm = (diagonal[0] * output[0]).hypot(diagonal[1] * output[1]);
            assert!((norm - report.step_norm().unwrap()).abs() < 1e-14);
            assert!(norm <= 1.1 * radius);
            if report.damping().unwrap() > 0.0 {
                assert!((norm - radius).abs() <= 0.1 * radius);
            }
            for i in 0..2 {
                assert!(
                    (output[i] - [3.0, 4.0][i]
                        + report.damping().unwrap() * diagonal[i] * diagonal[i] * output[i])
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    /// `A = diag(1, σ)`, `b = (1/2, 10⁻³)`, `D = I`, `Δ = 1`: the
    /// Gauss–Newton step has norm `10⁻³/σ`, and the root is
    /// `λ* ≈ 10⁻³ σ / √(3/4)` because `σ² ≪ λ*`.
    fn ill_conditioned_trust_region<S>(sigma: S)
    where
        S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S> + core::fmt::Debug,
    {
        let half = S::HALF;
        let small = S::try_from_f64_lossy(1e-3).unwrap();
        let a = Matrix::from_vec(2, 2, alloc::vec![S::ONE, S::ZERO, S::ZERO, sigma]).unwrap();
        let rhs = [half, small];
        let root = small * sigma / (S::try_from_f64_lossy(0.75).unwrap()).sqrt();
        let check = |damping: S, norm: S, output: [S; 2]| {
            let ratio = damping / root;
            assert!(
                ratio > S::try_from_f64_lossy(0.8).unwrap()
                    && ratio < S::try_from_f64_lossy(1.25).unwrap(),
                "{damping:?} vs {root:?}"
            );
            assert!((norm - S::ONE).abs() <= S::ONE / S::TEN, "{norm:?}");
            let expected = sigma * small / (sigma * sigma + damping);
            let tolerance = S::try_from_f64_lossy(1e-4).unwrap();
            assert!(((output[1] - expected) / expected).abs() < tolerance);
            assert!((output[0] - half / (S::ONE + damping)).abs() < tolerance);
        };
        let qr = HouseholderDecomposition::factor(
            &a,
            HouseholderOrientation::Qr,
            HouseholderPivoting::None,
        )
        .unwrap();
        let svd = ThinSvdDecomposition::compute(&a).unwrap();
        for hint in [S::ZERO, S::ONE] {
            let mut output = [S::ZERO; 2];
            let report = qr
                .solve_trust_region_into(
                    &rhs[..],
                    &[S::ONE, S::ONE],
                    S::ONE,
                    hint,
                    &DirectSolveOptions::new(RankPolicy::Exact),
                    VectorViewMut::from_slice(&mut output),
                    &mut LinalgWorkspace::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            check(
                report.damping().unwrap(),
                report.step_norm().unwrap(),
                output,
            );
            let report = svd
                .solve_trust_region_into(
                    &rhs[..],
                    S::ONE,
                    hint,
                    RankPolicy::Exact,
                    &LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(&mut output),
                    &mut LinalgWorkspace::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            check(
                report.damping().unwrap(),
                report.step_norm().unwrap(),
                output,
            );
        }
    }

    #[test]
    fn newton_bounds_hold_when_the_gauss_newton_slope_overflows() {
        ill_conditioned_trust_region(1e-15_f32);
        ill_conditioned_trust_region(1e-15_f64);
        ill_conditioned_trust_region(1e-150_f64);
    }

    #[test]
    fn svd_wide_trust_region_is_minimum_norm() {
        let a = Matrix::from_vec(1, 2, alloc::vec![1.0_f64, 1.0]).unwrap();
        let options = LinalgExecutionOptions::new();
        let factor = ThinSvdDecomposition::compute(&a).unwrap();
        let mut workspace =
            ThinSvdDecomposition::<f64>::workspace(&SvdWorkspaceRequest::solve(1, 2, &options))
                .unwrap();
        let mut output = [0.0; 2];
        let report = factor
            .solve_trust_region_into(
                &[2.0][..],
                0.5,
                1e-300,
                RankPolicy::Exact,
                &options,
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((output[0] - output[1]).abs() < 1e-14);
        assert!((output[0].hypot(output[1]) - 0.5).abs() < 0.05);
        assert!(report.damping().unwrap() > 0.0);
        let before = output;
        assert!(
            factor
                .solve_trust_region_into(
                    &[2.0][..],
                    0.0,
                    0.0,
                    RankPolicy::Exact,
                    &options,
                    VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(output, before);
    }

    #[test]
    fn qr_trust_region_rejects_lq_and_preserves_rank_rejection() {
        let matrix = Matrix::from_vec(2, 2, alloc::vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();
        let mut output = [7.0; 2];
        let mut workspace = LinalgWorkspace::new();
        for orientation in [HouseholderOrientation::Qr, HouseholderOrientation::Lq] {
            let factor =
                HouseholderDecomposition::factor(&matrix, orientation, HouseholderPivoting::Column)
                    .unwrap();
            let result = factor.solve_trust_region_into(
                &[1.0, 2.0][..],
                &[1.0, 1.0],
                1.0,
                0.0,
                &DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon),
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            );
            if orientation == HouseholderOrientation::Lq {
                assert!(result.is_err());
            } else {
                let report = result.unwrap();
                assert_eq!(report.termination(), DirectSolveTermination::RankDeficient);
                assert!(report.damping().is_none() && report.step_norm().is_none());
            }
            assert_eq!(output, [7.0; 2]);
        }
    }

    #[test]
    fn unbounded_steps_do_not_require_a_representable_normal_gradient() {
        let options = LinalgExecutionOptions::new();
        for scale in [1e-200_f64, 1e200] {
            let matrix = Matrix::from_vec(1, 1, alloc::vec![scale]).unwrap();
            let qr = HouseholderDecomposition::factor(
                &matrix,
                HouseholderOrientation::Qr,
                HouseholderPivoting::Column,
            )
            .unwrap();
            let svd = ThinSvdDecomposition::compute(&matrix).unwrap();
            let mut point = [0.0];
            let report = qr
                .solve_trust_region_into(
                    &[scale][..],
                    &[1.0],
                    2.0,
                    0.0,
                    &DirectSolveOptions::new(RankPolicy::Exact),
                    VectorViewMut::from_slice(&mut point),
                    &mut LinalgWorkspace::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            assert!((point[0] - 1.0).abs() < 1e-14);
            let report = svd
                .solve_trust_region_into(
                    &[scale][..],
                    2.0,
                    0.0,
                    RankPolicy::Exact,
                    &options,
                    VectorViewMut::from_slice(&mut point),
                    &mut LinalgWorkspace::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            assert!((point[0] - 1.0).abs() < 1e-14);
            assert_eq!(report.damping(), Some(0.0));
        }
    }
}
