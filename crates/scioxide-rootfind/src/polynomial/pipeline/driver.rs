//! Input checks and method selection of polynomial root solves.
use super::*;

/// Validate the coefficient `input` and return its degree.
pub(crate) fn validate_coefficients<C: NumericScalar>(
    coefficients: &[C],
    input: Argument,
) -> Result<usize, SciOxideError> {
    if coefficients.len() < 2 || coefficients.last() == Some(&C::ZERO) {
        return Err(invalid(
            input,
            scioxide_core::InvalidInputReason::OutOfRange,
            "positive degree and nonzero leading coefficient required",
        ));
    }
    if !coefficients.iter().all(|c| c.has_finite_components()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "coefficients must have finite components",
        ));
    }
    Ok(coefficients.len() - 1)
}
/// Reject a `method` unsupported by `basis`; errors name `options.method`.
pub(crate) fn validate_method<S: RealScalar>(
    basis: PolynomialRootBasisRef<'_, S>,
    method: PolynomialRootMethod,
) -> Result<(), SciOxideError> {
    if !basis.is_monomial()
        && matches!(
            method,
            PolynomialRootMethod::Laguerre | PolynomialRootMethod::DurandKerner
        )
    {
        return Err(invalid(
            scioxide_core::argument!("options.method"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Laguerre and Durand-Kerner require monomial coefficients",
        ));
    }
    Ok(())
}

/// Polynomial after removal of exact zero roots, with its power-of-two scale.
#[derive(Clone, Copy)]
pub(crate) struct Prepared<'a, S: RealScalar> {
    pub(crate) zeros: usize,
    pub(crate) degree: usize,
    pub(crate) scale: S,
    pub(crate) basis: PolynomialRootBasisRef<'a, S>,
    pub(crate) real: bool,
    pub(crate) companion_safe: bool,
}
/// Solves one polynomial into `output` within `context` with the method of
/// `options` and returns the report; `input` names the coefficient argument
/// in errors.
pub(crate) fn solve_into<S>(
    coefficients: &[impl NumericScalar<Real = S>],
    basis: PolynomialRootBasisRef<'_, S>,
    output: &mut [Complex<S>],
    workspace: &mut PolynomialRootWorkspace<S>,
    options: &PolynomialRootOptions<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
    input: Argument,
) -> Result<PolynomialRootReport<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let method = options.method;
    let degree = validate_coefficients(coefficients, input)?;
    validate_method(basis, method)?;
    if output.len() != degree {
        return Err(SciOxideError::dimension_mismatch([degree], [output.len()]));
    }
    if workspace.prepared != Some((degree, method)) {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "reserve workspace for exact degree and method",
        ));
    }
    let session = &crate::context::Session::new(context);
    // Initialize a finite, explicitly unassessed payload before admission can stop.
    workspace.candidates.fill(Complex::new(S::ZERO, S::ZERO));
    workspace.evaluations.fill(None);
    workspace
        .scratch_diagnostics
        .fill(PolynomialRootDiagnostic::unevaluated());
    let coefficient_scale = coefficients.iter().map(|c| c.abs()).fold(S::ZERO, S::max);
    let mut stats = PolynomialRootReport::new(
        method,
        options.backward_error_tolerance.tolerance(),
        coefficient_scale,
        degree,
        session.algorithm()?,
    );
    if !basis.is_monomial() {
        stats.error_model = scioxide_poly::PolynomialBackwardErrorModel::Normwise;
    }
    if coefficient_scale.is_finite() {
        let cutoff = options
            .rank_threshold
            .map_or(Ok(S::ZERO), |t| t.threshold(coefficient_scale))?;
        stats.effective_degree = coefficients
            .iter()
            .rposition(|c| c.abs() > cutoff)
            .unwrap_or(0);
        stats.effective_rank = coefficients.iter().filter(|c| c.abs() > cutoff).count();
    } else {
        // Finite complex components can have an unrepresentable modulus. Keep
        // the classification scale extended even though the report stores +∞.
        let magnitude = |c| scioxide_core::ScaledValue::from_value(c)?.magnitude();
        let scale = coefficients.iter().try_fold(Scaled::ZERO, |largest, &c| {
            let value = magnitude(c)?;
            Ok::<_, SciOxideError>(if value.abs_cmp(largest).is_gt() {
                value
            } else {
                largest
            })
        })?;
        stats.effective_degree = 0;
        stats.effective_rank = 0;
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let significant = match options.rank_threshold {
                Some(threshold) => !scioxide_core::backend::scaled::scaled_magnitude_le_tolerance(
                    magnitude(coefficient)?,
                    &threshold.tolerance(),
                    scale,
                )?,
                None => !magnitude(coefficient)?.is_zero(),
            };
            if significant {
                stats.effective_degree = index;
                stats.effective_rank += 1;
            }
        }
    }
    let result: Result<(), PolynomialFailure> = (|| {
        if !session.check_stop()? {
            return Err(PolynomialFailure::Stopped);
        }
        let prepared = prepare(coefficients, basis, &mut workspace.coefficients, input)?;
        let solved = run(prepared, workspace, method, options, &mut stats, session)?;
        finalize(
            coefficients,
            prepared,
            workspace,
            options,
            &mut stats,
            session,
        )?;
        stats.status = if !solved {
            RootTermination::IterationLimit
        } else if workspace.scratch_diagnostics.iter().all(|d| d.validated) {
            RootTermination::Complete
        } else {
            RootTermination::Unverified
        };
        Ok(())
    })();
    if let Err(error) = result {
        if let Some(stop) = session.stopped.get() {
            stats.status = stop;
        } else {
            let PolynomialFailure::Numerical(error) = error else {
                return Err(invalid(
                    scioxide_core::argument!("scope"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "missing polynomial stop reason",
                ));
            };
            match error.view() {
                scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } => {
                    stats.status = match reason {
                        scioxide_core::ExecutionStopReason::DeadlineReached => {
                            crate::RootTermination::TimeLimit
                        }
                        _ => crate::RootTermination::Cancelled,
                    };
                }
                scioxide_core::SciOxideErrorView::NumericalInstability { issue, .. } => {
                    stats.status = RootTermination::NumericalFailure(issue);
                }
                _ => return Err(error),
            }
        }
    }
    if matches!(stats.status, RootTermination::NumericalFailure(_)) {
        // Assess only real candidates, never the finite placeholders used for
        // ungenerated slots. Keep completed diagnostics for these exact points.
        for (root, diagnostic) in workspace
            .candidates
            .iter()
            .zip(&mut workspace.scratch_diagnostics)
        {
            if !diagnostic.available || diagnostic.backward_error.is_some() {
                continue;
            }
            match assess(coefficients, basis, *root, options, session) {
                Ok(assessment) => *diagnostic = assessment,
                Err(PolynomialFailure::Stopped) => break,
                Err(PolynomialFailure::Numerical(error))
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
                    ) => {}
                Err(PolynomialFailure::Numerical(error)) => return Err(error),
            }
        }
        if let Some(stop) = session.stopped.get() {
            stats.status = stop;
        }
    }
    if !matches!(
        stats.status,
        RootTermination::WorkLimit(_)
            | RootTermination::IterationLimit
            | RootTermination::Cancelled
            | RootTermination::TimeLimit
    ) && session.stopped.get().is_none()
    {
        let _ = session.check_stop()?;
        if let Some(stop) = session.stopped.get() {
            stats.status = stop;
        }
    }
    stats.counts = session.accounting()?;
    stats.max_residual = workspace
        .scratch_diagnostics
        .iter()
        .try_fold(S::ZERO, |largest, d| {
            d.backward_error.map(|e| largest.max(e))
        });
    stats.max_root_condition = workspace
        .scratch_diagnostics
        .iter()
        .map(|d| d.condition)
        .reduce(|a, b| a.maximum(b))
        .unwrap_or(scioxide_poly::RootEstimate::Unavailable);
    sort_with_diagnostics(
        &mut workspace.candidates,
        &mut workspace.scratch_diagnostics,
    );
    if stats.status.is_success() {
        session.assess()?;
    }
    output.copy_from_slice(&workspace.candidates);
    core::mem::swap(
        &mut workspace.diagnostics,
        &mut workspace.scratch_diagnostics,
    );
    Ok(stats)
}
pub(crate) fn run<S>(
    prepared: Prepared<'_, S>,
    w: &mut PolynomialRootWorkspace<S>,
    requested: PolynomialRootMethod,
    options: &PolynomialRootOptions<S>,
    stats: &mut PolynomialRootReport<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<bool, PolynomialFailure>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let n = prepared.degree;
    for diagnostic in &mut w.scratch_diagnostics[..prepared.zeros] {
        diagnostic.available = true;
    }
    if n <= 2 && prepared.basis.is_monomial() {
        stats.effective_method = Some(PolynomialRootAlgorithm::Analytic);
        if n > 0 {
            analytic(&w.coefficients[..=n], &mut w.current[..n])?;
            publish_candidates(prepared, w)?;
        }
        return Ok(true);
    }
    seed_radii(
        &w.coefficients[..=n],
        prepared.basis,
        &mut w.seed_radii[..n],
        &mut w.hull,
    )?;
    initialize(&mut w.current[..n], &w.seed_radii[..n], 0)?;
    publish_candidates(prepared, w)?;
    // Reservation and selection use the same declared degree, even after deflation.
    let mut primary = primary_method(w.candidates.len(), requested);
    if primary == PolynomialRootMethod::Companion
        && !prepared.companion_safe
        && prepared.basis.is_monomial()
    {
        primary = PolynomialRootMethod::Aberth;
        stats.fallback_used = true;
    }
    stats.effective_method = PolynomialRootAlgorithm::from_method(primary);
    let mut seeded = false;
    if primary == PolynomialRootMethod::Companion {
        match companion(prepared, w, options, session) {
            Ok(()) => {
                // A nonzero constant excludes zero roots. Eigenvalue deflation
                // can erase small roots even when coefficients survived scaling.
                if prepared.basis.is_monomial()
                    && w.current[..n]
                        .iter()
                        .any(|z| *z == Complex::new(S::ZERO, S::ZERO))
                {
                    stats.fallback_used = true;
                    stats.effective_method = Some(PolynomialRootAlgorithm::Aberth);
                    initialize(&mut w.current[..n], &w.seed_radii[..n], 0)?;
                } else {
                    seeded = true;
                    publish_candidates(prepared, w)?;
                }
            }
            Err(error)
                if session.stopped.get().is_none()
                    && matches!(&error, PolynomialFailure::Numerical(error) if is_fallback_error(error)) =>
            {
                stats.fallback_used = true;
                stats.effective_method = Some(PolynomialRootAlgorithm::Aberth);
            }
            Err(error) => return Err(error),
        }
    } else if primary == PolynomialRootMethod::Laguerre {
        seeded = laguerre_candidates(prepared, w, options, stats, session)?;
        if !seeded {
            stats.fallback_used = true;
            stats.effective_method = Some(PolynomialRootAlgorithm::Aberth);
        }
    }
    let method = if primary == PolynomialRootMethod::DurandKerner {
        primary
    } else {
        PolynomialRootMethod::Aberth
    };
    // Unconverged simultaneous iteration publishes its unsuccessful report;
    // `Auto` above degree 64 reserves no companion storage to fall back to.
    simultaneous(prepared, w, method, options, stats, session, seeded)
}
pub(crate) fn primary_method(
    degree: usize,
    requested: PolynomialRootMethod,
) -> PolynomialRootMethod {
    match requested {
        PolynomialRootMethod::Auto if degree <= 64 => PolynomialRootMethod::Companion,
        PolynomialRootMethod::Auto => PolynomialRootMethod::Aberth,
        method => method,
    }
}
pub(crate) fn is_fallback_error(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            | scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    )
}

// Admission stops are inline; constructing a structured convergence error would allocate.
#[derive(Debug)]
pub(crate) enum PolynomialFailure {
    Stopped,
    Numerical(SciOxideError),
}
impl From<SciOxideError> for PolynomialFailure {
    fn from(error: SciOxideError) -> Self {
        Self::Numerical(error)
    }
}
pub(crate) fn admitted(granted: bool) -> Result<(), PolynomialFailure> {
    if granted {
        Ok(())
    } else {
        Err(PolynomialFailure::Stopped)
    }
}
pub(crate) fn iteration(
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), PolynomialFailure> {
    admitted(session.try_iteration()?)
}
pub(crate) fn admit(
    session: &crate::context::Session<'_, '_, '_>,
    role: scioxide_core::EvaluationRole,
) -> Result<(), PolynomialFailure> {
    admitted(session.try_admit(role)?)
}
pub(crate) fn invoke<T>(
    session: &crate::context::Session<'_, '_, '_>,
    role: scioxide_core::EvaluationRole,
    provider: impl FnOnce() -> Result<T, SciOxideError>,
) -> Result<T, PolynomialFailure> {
    admit(session, role)?;
    Ok(provider()?)
}
