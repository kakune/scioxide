//! `solve_system`.
use scioxide_core::{
    Argument, ConvergenceReason, InvalidInputReason, NonFiniteOrigin, NumericScalar, RealScalar,
    SciOxideError, SciOxideErrorView, SolveCounts, argument,
};
use scioxide_rootfind::{
    ResidualFn, RootTermination, SystemRootResult, SystemRootSolver, TrustRegionSystem,
    TrustRegionSystemOptions, solve_system,
};

fn invalid(error: &SciOxideError) -> Option<(Argument, InvalidInputReason)> {
    match error.view() {
        SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    }
}

fn non_finite(error: &SciOxideError) -> Option<NonFiniteOrigin> {
    match error.view() {
        SciOxideErrorView::NonFinite { origin, .. } => Some(origin),
        _ => None,
    }
}

fn convergence_failure(error: &SciOxideError) -> Option<(ConvergenceReason, SolveCounts)> {
    match error.view() {
        SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            ..
        } => Some((reason, diagnostics.counts())),
        _ => None,
    }
}

fn c<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

/// `x² + y − 1 = 0`, `x + y² − 1 = 0`; from `(0.5, 0.5)` the solve reaches
/// `x = y = (√5 − 1)/2`.
fn parabolas<S: RealScalar>(x: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
    out[0] = x[0] * x[0] + x[1] - S::ONE;
    out[1] = x[0] + x[1] * x[1] - S::ONE;
    Ok(())
}

/// `Fᵢ(x) = xᵢ³ − (i + 2)`, a diagonal cubic with root `xᵢ = ∛(i + 2)`.
fn cubes<S: RealScalar>(x: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
    for (index, (slot, &value)) in out.iter_mut().zip(x).enumerate() {
        *slot = value * value * value - c::<S>((index + 2) as f64);
    }
    Ok(())
}

type Callback<S> = fn(&[S], &mut [S]) -> Result<(), SciOxideError>;

fn configured<S>(f: Callback<S>, initial: &[S]) -> SystemRootResult<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    TrustRegionSystem::new(TrustRegionSystemOptions::default())
        .unwrap()
        .solve(&ResidualFn::new(initial.len(), f), initial)
        .unwrap()
}

fn matches_trust_region<S>(f: Callback<S>, initial: &[f64])
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let initial: Vec<S> = initial.iter().map(|&x| c(x)).collect();
    let configured = configured(f, &initial);
    assert!(configured.is_success(), "{configured:?}");
    let one_shot = solve_system(f, &initial).unwrap();
    assert_eq!(one_shot, configured.point());
    for (x, y) in one_shot.iter().zip(configured.point()) {
        assert_eq!(x.is_sign_negative(), y.is_sign_negative());
    }
}

#[test]
fn solve_system_equals_the_trust_region_with_default_options() {
    matches_trust_region::<f64>(parabolas, &[0.5, 0.5]);
    matches_trust_region::<f32>(parabolas, &[0.5, 0.5]);
    matches_trust_region::<f64>(parabolas, &[-3.0, 4.0]);
    matches_trust_region::<f64>(cubes, &[1.0, 1.0, 1.0]);
    matches_trust_region::<f32>(cubes, &[3.0, -2.0]);
}

#[test]
fn solve_system_reaches_reference_values_without_a_jacobian() {
    let root = (5.0_f64.sqrt() - 1.0) / 2.0;
    let x = solve_system(parabolas::<f64>, &[0.5, 0.5]).unwrap();
    assert!(x.iter().all(|x| (x - root).abs() < 1e-7), "{x:?}");
    let x = solve_system(parabolas::<f32>, &[0.5, 0.5]).unwrap();
    assert!(
        x.iter().all(|&x| (f64::from(x) - root).abs() < 1e-3),
        "{x:?}"
    );

    let x = solve_system(cubes::<f64>, &[1.0; 3]).unwrap();
    for (index, x) in x.iter().enumerate() {
        assert!((x - ((index + 2) as f64).cbrt()).abs() < 1e-7, "{x}");
    }
}

#[test]
fn solve_system_reports_its_own_arguments() {
    let empty = solve_system(parabolas::<f64>, &[]).unwrap_err();
    assert_eq!(
        invalid(&empty),
        Some((argument!("initial"), InvalidInputReason::OutOfRange))
    );
    let nan = solve_system(parabolas::<f64>, &[f64::NAN, 0.5]).unwrap_err();
    assert_eq!(non_finite(&nan), Some(NonFiniteOrigin::Input));
    let infinite = solve_system(parabolas::<f32>, &[0.5, f32::INFINITY]).unwrap_err();
    assert_eq!(non_finite(&infinite), Some(NonFiniteOrigin::Input));
}

#[test]
fn solve_system_returns_callback_errors_unchanged() {
    let failure = SciOxideError::external("callback failed");
    let error = solve_system(
        |_: &[f64], _: &mut [f64]| Err(SciOxideError::external("callback failed")),
        &[0.5, 0.5],
    );
    assert_eq!(error.unwrap_err(), failure);

    let error = solve_system(
        |_: &[f64], out: &mut [f64]| {
            out[0] = f64::NAN;
            Ok(())
        },
        &[0.0],
    );
    assert_eq!(
        non_finite(&error.unwrap_err()),
        Some(NonFiniteOrigin::Callback)
    );
}

/// `x² + 1 = 0` has no real root: `‖F(x)‖ ≥ 1`, so the residual test cannot
/// pass and the configured form stops with `Stagnation`.
#[test]
fn solve_system_reports_an_unsuccessful_termination_as_an_error() {
    let no_root: Callback<f64> = |x, out| {
        out[0] = x[0] * x[0] + 1.0;
        Ok(())
    };
    let result = configured(no_root, &[1.0]);
    assert_eq!(result.termination(), RootTermination::Stagnation);
    assert!(!result.is_success());
    let error = solve_system(no_root, &[1.0]).unwrap_err();
    assert_eq!(
        convergence_failure(&error),
        Some((ConvergenceReason::Stagnation, result.accounting()))
    );
}

#[test]
fn solve_system_returns_a_root_at_the_start_and_one_unknown() {
    let line: Callback<f64> = |x, out| {
        out[0] = x[0] - 0.25;
        Ok(())
    };
    assert_eq!(solve_system(line, &[0.25]).unwrap(), [0.25]);
    assert!((solve_system(line, &[-7.0]).unwrap()[0] - 0.25).abs() < 1e-12);
    let exact = [1.0, 0.0];
    assert_eq!(solve_system(parabolas::<f64>, &exact).unwrap(), exact);
    assert_eq!(
        solve_system(parabolas::<f32>, &[0.0, 1.0]).unwrap(),
        [0.0, 1.0]
    );
}
