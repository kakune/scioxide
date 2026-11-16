//! Acceleration evidence, best-estimate publication and range safety against
//! mpmath references (`mp.dps = 30`).
#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, ErrorTolerance, ExecutionContext, NormSelection, SciOxideError};
use scioxide_integrate::{
    SeriesError, SeriesMethod, SeriesOptions, SeriesProblem, SeriesReport, SeriesSummation,
    SeriesTermination,
};

use core::f64::consts::{E, FRAC_PI_4 as QUARTER_PI, LN_2};

/// mpmath `zeta(2)`, `zeta(1.1)`, `(1 - sqrt(2)) zeta(1/2)`, `-log(1 - 0.99)` and
/// `-log(1 - 0.999)`,
/// rounded to binary64.
const ZETA_2: f64 = 1.644_934_066_848_226_4;
const ZETA_1_1: f64 = 10.584_448_464_950_81;
const ALTERNATING_ROOT: f64 = 0.604_898_643_421_630_4;
const LOG_HUNDRED: f64 = 4.605_170_185_988_092;
const LOG_THOUSAND: f64 = 6.907_755_278_982_137;

struct Run {
    report: SeriesReport,
    values: Vec<f64>,
    errors: Vec<SeriesError<f64>>,
}

fn run(
    method: SeriesMethod,
    first: usize,
    order: usize,
    max_terms: usize,
    (absolute, relative): (f64, f64),
    width: usize,
    terms: impl Fn(usize, &mut [f64]) + Sync,
) -> Run {
    let plan = SeriesSummation::<f64>::new(
        method,
        first,
        SeriesOptions::default()
            .with_order(Budget::try_from(order).unwrap())
            .with_max_terms(Budget::try_from(max_terms).unwrap())
            .with_terms_per_advance(Budget::try_from(max_terms).unwrap())
            .with_tolerance(
                ErrorTolerance::scalar(absolute, relative, NormSelection::Infinity).unwrap(),
            ),
    )
    .unwrap();
    let mut workspace = plan.workspace(width).unwrap();
    let mut values = vec![f64::NAN; width];
    let provider = |n: usize, row: &mut [f64]| {
        terms(n, row);
        Ok::<(), SciOxideError>(())
    };
    let report = plan
        .sum_into(
            SeriesProblem::Terms(&provider),
            &mut values,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    Run {
        report,
        values,
        errors: workspace.component_errors().to_vec(),
    }
}

fn alternating(n: usize) -> f64 {
    let magnitude = 1.0 / (n as f64 + 1.0);
    if n.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

const DEFAULT: (f64, f64) = (1.490_116_119_384_765_6e-8, 1.490_116_119_384_765_6e-8);

#[test]
fn zero_and_stationary_components_never_block_the_others() {
    for method in [SeriesMethod::LevinU, SeriesMethod::WynnEpsilon] {
        for tolerance in [DEFAULT, (1e-12, 1e-12)] {
            let zero = run(method, 0, 8, 2000, tolerance, 2, |n, row| {
                row.copy_from_slice(&[alternating(n), 0.0]);
            });
            assert_eq!(
                zero.report.termination(),
                SeriesTermination::HeuristicToleranceReached,
                "{method:?} {tolerance:?}"
            );
            assert!(zero.report.terms() <= 60, "{:?}", zero.report);
            assert!((zero.values[0] - LN_2).abs() <= 1e3 * tolerance.0);
            assert_eq!(zero.values[1], 0.0);
            assert_eq!(zero.errors[1].acceleration(), Some(0.0));
            let factorial = run(method, 0, 8, 2000, tolerance, 2, |n, row| {
                let inverse = (1..=n).fold(1.0, |f, k| f / k as f64);
                row.copy_from_slice(&[alternating(n), inverse]);
            });
            assert_eq!(
                factorial.report.termination(),
                SeriesTermination::HeuristicToleranceReached
            );
            assert!(factorial.report.terms() <= 60);
            assert!((factorial.values[0] - LN_2).abs() <= 1e3 * tolerance.0);
            assert!((factorial.values[1] - E).abs() <= 4.0 * f64::EPSILON * E);
        }
    }
}

#[test]
fn terminating_series_stop_at_their_exact_partial_sum() {
    for method in [SeriesMethod::LevinU, SeriesMethod::WynnEpsilon] {
        for order in [2, 8, 12] {
            let result = run(method, 0, order, 1000, (1e-13, 1e-13), 1, |n, row| {
                row[0] = if n < 5 { 1.0 / (n as f64 + 1.0) } else { 0.0 };
            });
            assert_eq!(
                result.report.termination(),
                SeriesTermination::HeuristicToleranceReached
            );
            // A full window of zero terms, then three consecutive stationary steps.
            assert!(
                result.report.terms() <= 5 + order + 1 + 4,
                "{:?}",
                result.report
            );
            assert!((result.values[0] - 137.0 / 60.0).abs() <= 2.0 * f64::EPSILON);
            assert_eq!(result.errors[0].acceleration(), Some(0.0));
        }
    }
}

#[test]
fn zero_interspersed_series_are_transformed_on_their_informative_terms() {
    // sum_(n >= 1) sin(n pi / 2) / n = pi / 4: every other term is zero.
    let terms = |n: usize, row: &mut [f64]| {
        row[0] = match n % 4 {
            1 => 1.0 / n as f64,
            3 => -1.0 / n as f64,
            _ => 0.0,
        };
    };
    let levin = run(SeriesMethod::LevinU, 1, 8, 256, DEFAULT, 1, terms);
    assert_eq!(
        levin.report.termination(),
        SeriesTermination::HeuristicToleranceReached
    );
    assert!(levin.report.terms() <= 40);
    assert!((levin.values[0] - QUARTER_PI).abs() <= 1e-8);
    let tight = run(SeriesMethod::LevinU, 1, 12, 256, (1e-13, 1e-13), 1, terms);
    assert_eq!(
        tight.report.termination(),
        SeriesTermination::HeuristicToleranceReached
    );
    assert!((tight.values[0] - QUARTER_PI).abs() <= 1e-14);
    // Wynn accelerates the same rows far beyond the partial sums' 1/n accuracy.
    let wynn = run(SeriesMethod::WynnEpsilon, 1, 8, 256, DEFAULT, 1, terms);
    assert!(
        !wynn.report.termination().is_success(),
        "{:?}",
        wynn.report.termination()
    );
    assert!((wynn.values[0] - QUARTER_PI).abs() <= 1e-10);
}

#[test]
fn log_convergent_levin_succeeds_at_the_default_tolerance_within_its_reported_error() {
    let zeta = |n: usize, row: &mut [f64]| row[0] = 1.0 / (n as f64 * n as f64);
    let slow = |n: usize, row: &mut [f64]| row[0] = (n as f64).powf(-1.1);
    for (terms, reference, budget) in [
        (&zeta as &(dyn Fn(usize, &mut [f64]) + Sync), ZETA_2, 12),
        (&slow, ZETA_1_1, 16),
    ] {
        let result = run(SeriesMethod::LevinU, 1, 8, 256, DEFAULT, 1, terms);
        assert_eq!(
            result.report.termination(),
            SeriesTermination::HeuristicToleranceReached
        );
        assert!(result.report.terms() <= budget, "{:?}", result.report);
        let error = (result.values[0] - reference).abs();
        assert!(error <= DEFAULT.1 * reference, "{error:e}");
        let reported = result.errors[0]
            .acceleration()
            .unwrap()
            .max(result.errors[0].roundoff());
        assert!(error <= reported, "{error:e} > {reported:e}");
    }
}

#[test]
fn tighter_tolerances_publish_the_best_estimate_seen() {
    let zeta = |n: usize, row: &mut [f64]| row[0] = 1.0 / (n as f64 * n as f64);
    let default = run(SeriesMethod::LevinU, 1, 8, 256, DEFAULT, 1, zeta);
    let error = (default.values[0] - ZETA_2).abs();
    for max_terms in [60, 1000, 10_000] {
        let tight = run(SeriesMethod::LevinU, 1, 8, max_terms, (0.0, 1e-14), 1, zeta);
        assert_eq!(tight.report.termination(), SeriesTermination::TermLimit);
        // The sliding window loses accuracy after about 16 terms; the best is retained.
        let tight_error = (tight.values[0] - ZETA_2).abs();
        assert!(
            tight_error <= error && tight_error <= 2e-11 * ZETA_2,
            "{tight_error:e}"
        );
        assert!(tight.errors[0].acceleration().unwrap() < 1e-9);
    }
    let slow = |n: usize, row: &mut [f64]| row[0] = (n as f64).powf(-1.1);
    for order in [8, 12] {
        let tight = run(SeriesMethod::LevinU, 1, order, 1000, (0.0, 1e-13), 1, slow);
        assert!(
            !tight.report.termination().is_success(),
            "order={order} {:?}",
            tight.report.termination()
        );
        assert!((tight.values[0] - ZETA_1_1).abs() <= 1e-8 * ZETA_1_1);
    }
}

#[test]
fn unreliable_evidence_does_not_hold_back_improving_estimates() {
    // Wynn epsilon cannot accelerate these logarithmic or nearly linear sequences;
    // its estimates still improve, so the newest ones are published.
    let zeta = |n: usize, row: &mut [f64]| row[0] = 1.0 / (n as f64 * n as f64);
    let wynn = run(SeriesMethod::WynnEpsilon, 1, 8, 256, DEFAULT, 1, zeta);
    assert_eq!(wynn.report.termination(), SeriesTermination::TermLimit);
    assert!((wynn.values[0] - ZETA_2).abs() <= 7e-4 * ZETA_2);
    let log = |n: usize, row: &mut [f64]| row[0] = 0.999_f64.powi(n as i32) / n as f64;
    let reference = LOG_THOUSAND;
    let wynn = run(SeriesMethod::WynnEpsilon, 1, 12, 1000, (0.0, 1e-13), 1, log);
    assert!((wynn.values[0] - reference).abs() <= 2e-3 * reference);
    // Early Levin changes shrink slowly; their geometric tail makes them less credible
    // than the partial sums' remainder evidence.
    let levin = run(SeriesMethod::LevinU, 1, 12, 1000, (0.0, 1e-13), 1, log);
    assert!((levin.values[0] - reference).abs() <= 0.04 * reference);
}

#[test]
fn unaccelerated_partial_sums_compete_by_their_remainder_evidence() {
    // Levin u cannot condition sum 0.99^n / n at order 8; the partial sums converge.
    let reference = LOG_HUNDRED;
    let terms = |n: usize, row: &mut [f64]| row[0] = 0.99_f64.powi(n as i32) / n as f64;
    let short = run(SeriesMethod::LevinU, 1, 12, 1000, (0.0, 1e-13), 1, terms);
    assert_eq!(short.report.termination(), SeriesTermination::TermLimit);
    assert!((short.values[0] - reference).abs() <= 1e-5);
    assert!(short.errors[0].acceleration().is_none());
    assert!(short.errors[0].remainder().unwrap() < 1e-4);
    // A full window of terms below the partial sum's rounding is stationary evidence.
    let long = run(SeriesMethod::LevinU, 1, 8, 10_000, (0.0, 1e-10), 1, terms);
    assert_eq!(
        long.report.termination(),
        SeriesTermination::HeuristicToleranceReached
    );
    assert!((long.values[0] - reference).abs() <= 1e-12);
}

#[test]
fn conditionally_convergent_floors_follow_the_partial_sums_not_the_term_mass() {
    // sum (-1)^n / sqrt(n + 1): the absolute term mass grows like 2 sqrt(n).
    let terms = |n: usize, row: &mut [f64]| {
        let magnitude = 1.0 / (n as f64 + 1.0).sqrt();
        row[0] = if n.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        };
    };
    for method in [SeriesMethod::LevinU, SeriesMethod::WynnEpsilon] {
        let result = run(method, 0, 12, 1000, (0.0, 1e-13), 1, terms);
        assert_eq!(
            result.report.termination(),
            SeriesTermination::HeuristicToleranceReached,
            "{method:?}"
        );
        assert!((result.values[0] - ALTERNATING_ROOT).abs() <= 1e-15);
        assert!(result.errors[0].roundoff() <= 1e-13);
    }
}

#[test]
fn near_overflow_alternating_terms_accelerate_without_a_mass_overflow() {
    // Partial sums stay finite although the absolute term mass overflows.
    for method in [SeriesMethod::LevinU, SeriesMethod::WynnEpsilon] {
        let result = run(method, 0, 8, 256, DEFAULT, 1, |n, row| {
            row[0] = 1.5e308 * alternating(n);
        });
        assert_eq!(
            result.report.termination(),
            SeriesTermination::HeuristicToleranceReached
        );
        assert!(
            (result.values[0] / 1.5e308 - LN_2).abs() <= 1e-8,
            "{method:?}"
        );
    }
}
