//! `find_root` and `find_bracket`.
use scioxide_core::{
    Argument, Interval, InvalidInputReason, NonFiniteOrigin, RealScalar, SciOxideError,
    SciOxideErrorView, argument,
};
use scioxide_rootfind::{
    BracketSearchDirection, BracketSearchInput, BracketSearchOptions, Brent, ScalarFn,
    ScalarRootOptions, ScalarRootSolver, find_bracket, find_bracket_with_options, find_root,
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

fn c<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

fn wallis<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    Ok(x * x * x - S::TWO * x - c::<S>(5.0))
}

fn kepler<S: RealScalar>(e: S) -> Result<S, SciOxideError> {
    Ok(e - c::<S>(0.5) * e.sin() - S::ONE)
}

fn root_matches_brent<S: RealScalar>(f: fn(S) -> Result<S, SciOxideError>, lower: f64, upper: f64) {
    let (lower, upper) = (c::<S>(lower), c::<S>(upper));
    let configured = Brent
        .solve(
            &ScalarFn::new(f),
            Interval::new(lower, upper).unwrap(),
            &ScalarRootOptions::default(),
        )
        .unwrap();
    assert!(configured.is_success());
    let one_shot = find_root(f, lower, upper).unwrap();
    assert_eq!(one_shot, configured.point());
    assert_eq!(
        one_shot.is_sign_negative(),
        configured.point().is_sign_negative()
    );
}

fn bracket_matches_configured<S: RealScalar>(f: fn(S) -> Result<S, SciOxideError>, guess: f64) {
    let guess = c::<S>(guess);
    let configured = find_bracket_with_options(
        &ScalarFn::new(f),
        BracketSearchInput::Guess(guess),
        BracketSearchDirection::Automatic,
        &BracketSearchOptions::default(),
    )
    .unwrap();
    let one_shot = find_bracket(f, guess).unwrap();
    assert_eq!(one_shot.lower(), configured.lower());
    assert_eq!(one_shot.upper(), configured.upper());
    assert_eq!(one_shot.lower_value(), configured.lower_value());
    assert_eq!(one_shot.upper_value(), configured.upper_value());
}

#[test]
fn find_root_equals_brent_with_default_options() {
    root_matches_brent::<f64>(wallis, 2.0, 3.0);
    root_matches_brent::<f32>(wallis, 2.0, 3.0);
    root_matches_brent::<f64>(kepler, 0.0, 4.0);
    root_matches_brent::<f32>(kepler, -1.0, 4.0);
    root_matches_brent::<f64>(|x| Ok(x.cos() - x), 0.0, 1.0);
}

#[test]
fn find_bracket_equals_the_configured_search() {
    bracket_matches_configured::<f64>(kepler, 0.0);
    bracket_matches_configured::<f32>(kepler, 0.0);
    bracket_matches_configured::<f64>(wallis, -40.0);
    bracket_matches_configured::<f32>(wallis, 1.0e3);
}

#[test]
fn find_root_reaches_reference_values() {
    let x = find_root(wallis::<f64>, 2.0, 3.0).unwrap();
    assert!((x - 2.094_551_481_542_326_5).abs() <= 4.0 * f64::EPSILON * 3.0);
    let x = find_root(wallis::<f32>, 2.0, 3.0).unwrap();
    assert!((x - 2.094_551_5_f32).abs() <= 4.0 * f32::EPSILON * 3.0);

    let bracket = find_bracket(kepler::<f64>, 0.0).unwrap();
    let e = find_root(kepler::<f64>, bracket.lower(), bracket.upper()).unwrap();
    assert!(kepler(e).unwrap().abs() < 1e-14);
}

#[test]
fn find_root_reports_its_own_arguments() {
    let unordered = find_root(wallis::<f64>, 3.0, 2.0).unwrap_err();
    assert_eq!(
        invalid(&unordered),
        Some((argument!("lower"), InvalidInputReason::Inconsistent))
    );
    let same_sign = find_root(wallis::<f64>, 3.0, 4.0).unwrap_err();
    assert_eq!(
        invalid(&same_sign),
        Some((argument!("lower"), InvalidInputReason::Inconsistent))
    );
    let nan = find_root(wallis::<f64>, f64::NAN, 3.0).unwrap_err();
    assert_eq!(non_finite(&nan), Some(NonFiniteOrigin::Input));
    let infinite = find_root(wallis::<f32>, 2.0, f32::INFINITY).unwrap_err();
    assert_eq!(non_finite(&infinite), Some(NonFiniteOrigin::Input));
}

#[test]
fn find_root_returns_callback_errors_unchanged() {
    let failure = SciOxideError::external("callback failed");
    let error = find_root(
        |_: f64| Err(SciOxideError::external("callback failed")),
        0.0,
        1.0,
    );
    assert_eq!(error.unwrap_err(), failure);

    let error = find_root(|x: f64| Ok(if x > 0.5 { f64::NAN } else { -1.0 }), 0.0, 1.0);
    assert_eq!(
        non_finite(&error.unwrap_err()),
        Some(NonFiniteOrigin::Callback)
    );
}

#[test]
fn find_bracket_rejects_a_non_finite_guess() {
    let error = find_bracket(kepler::<f64>, f64::NAN).unwrap_err();
    assert_eq!(non_finite(&error), Some(NonFiniteOrigin::Input));
    let error = find_bracket(kepler::<f32>, f32::NEG_INFINITY).unwrap_err();
    assert_eq!(non_finite(&error), Some(NonFiniteOrigin::Input));
}

#[test]
fn find_root_accepts_roots_at_the_endpoints() {
    let line = |x: f64| Ok(x - 1.0);
    assert_eq!(find_root(line, 1.0, 2.0).unwrap(), 1.0);
    assert_eq!(find_root(line, 0.0, 1.0).unwrap(), 1.0);
    assert_eq!(find_root(line, 1.0, 1.0).unwrap(), 1.0);
    let error = find_root(line, 2.0, 2.0).unwrap_err();
    assert_eq!(
        invalid(&error),
        Some((argument!("lower"), InvalidInputReason::Inconsistent))
    );
}
