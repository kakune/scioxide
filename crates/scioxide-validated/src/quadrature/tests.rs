use super::*;
use crate::point_enclosure;
use alloc::vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{AlgebraicScalar, Complex};
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, NormSelection, SciOxideErrorKind, SolveBudgets, SolveState,
};

const fn opts() -> ValidatedOptions {
    ValidatedOptions::new()
}
const fn seq() -> ExecutionContext<'static> {
    ExecutionContext::sequential()
}

/// Fixture oracle argument.
const REQUEST: Argument = scioxide_core::argument!("request");

#[test]
fn panel_sum_retains_exact_cancellation_and_subnormal_endpoints() {
    let eta = f64::from_bits(1);
    let point = |x: f64| point_enclosure(Complex::new(x, -x)).unwrap();
    for values in [
        [f64::MAX, -f64::MAX, eta],
        [-f64::MAX, f64::MAX, eta],
        [eta, eta, -eta],
    ] {
        let bounds = values.map(point);
        let mut total = [sentinel()];
        assert_eq!(proof::sum(&bounds, &mut total, REQUEST).unwrap(), None);
        for (component, expected) in [(total[0].re, eta), (total[0].im, -eta)] {
            let (lo, hi) = component.bounds().unwrap();
            assert_eq!(lo.unwrap().to_bits(), expected.to_bits());
            assert_eq!(hi.unwrap().to_bits(), expected.to_bits());
        }
    }
}

fn r<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn tolerance<R: RealScalar>(x: f64) -> ErrorTolerance<R> {
    ErrorTolerance::scalar(r(x), R::ZERO, NormSelection::Infinity).unwrap()
}

fn smooth<S: NumericScalar>(
    name: &str,
    q: QuadratureRequest<S::Real>,
    out: &mut [IntervalBox<S::Real>],
) -> Result<QuadratureObligation> {
    let x = D::from(q.domain);
    let value = match (name, q.query) {
        ("exp", QuadratureQuery::Value | QuadratureQuery::EighthDerivative) => x.exp()?.hull(),
        ("cauchy", QuadratureQuery::Value) => {
            D::point(r(1.)).unwrap().add(&x.square()?)?.reciprocal()?
        }
        // Partial fractions at the poles ±i give
        // |f^(8)(x)| ≤ 8! / |x+i|^9 ≤ 40320 for real x.
        ("cauchy", QuadratureQuery::EighthDerivative) => D::new(r(-40320.), r(40320.)).unwrap(),
        _ => return Ok(QuadratureObligation::Unavailable),
    };
    out.fill(Complex::new(
        value.set(),
        if S::IS_REAL {
            D::point(r(0.)).unwrap().set()
        } else {
            value.neg()?.set()
        },
    ));
    Ok(QuadratureObligation::Established)
}

fn high_order_fixtures<S: NumericScalar>(width: &str, goal: f64) {
    let mut checked = 0;
    for line in include_str!("../../tests/data/quadrature-gauss.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[1] != width || !matches!(fields[0], "exp" | "cauchy") {
            continue;
        }
        checked += 1;
        let name = fields[0];
        let decode = |text| {
            let bits = u64::from_str_radix(text, 16).unwrap();
            r::<S::Real>(if width == "32" {
                f32::from_bits(bits as u32) as f64
            } else {
                f64::from_bits(bits)
            })
        };
        let expected = D::new(decode(fields[2]), decode(fields[3])).unwrap();
        let pieces = [piece(
            0.,
            1.,
            QuadratureAssumption::EightTimesContinuouslyDifferentiable,
        )];
        let missing = AtomicUsize::new(0);
        let oracle = |q: QuadratureRequest<S::Real>, out: &mut [IntervalBox<S::Real>]| {
            if q.query == QuadratureQuery::EighthDerivative && missing.load(Ordering::Relaxed) != 0
            {
                return Ok(QuadratureObligation::Unavailable);
            }
            smooth::<S>(name, q, out)
        };
        let plan = QuadratureVerifier::<S>::new(&pieces, 1, &oracle, opts()).unwrap();
        let (report, certificate) = plan.generate(&tolerance(goal), 32).unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::ToleranceReached,
            "{name}"
        );
        assert!(report.panels() <= 32);
        let certificate = certificate.unwrap();
        assert!(certificate.total[0].re.contains_interval(&expected.set()));
        assert!(adaptive::radius_scale(certificate.total[0]).unwrap().0 <= r(goal));
        assert_eq!(
            plan.verify(certificate.view()).unwrap(),
            QuadratureVerificationStatus::Verified
        );
        if !S::IS_REAL {
            assert!(
                certificate.total[0]
                    .im
                    .contains_interval(&expected.neg().unwrap().set())
            );
        }
        let mut bad = certificate.clone();
        bad.panels[0].domain = D::point(r(0.)).unwrap().set();
        assert!(plan.verify(bad.view()).is_err());
        let mut bad = certificate.clone();
        bad.local_bounds[0] = zero().unwrap();
        assert_eq!(
            plan.verify(bad.view()).unwrap(),
            QuadratureVerificationStatus::Rejected(QuadratureRejection::LocalBound)
        );
        let mut bad = certificate.clone();
        bad.total[0] = zero().unwrap();
        assert_eq!(
            plan.verify(bad.view()).unwrap(),
            QuadratureVerificationStatus::Rejected(QuadratureRejection::TotalBound)
        );
        missing.store(1, Ordering::Relaxed);
        assert_eq!(
            plan.verify(certificate.view()).unwrap(),
            QuadratureVerificationStatus::Inconclusive(QuadratureInconclusive::Reason(
                QuadratureProofFailure::MissingObligation
            ))
        );
        assert!(
            !plan
                .generate(&tolerance(goal), 32)
                .unwrap()
                .0
                .has_enclosure()
        );
    }
    assert_eq!(checked, 2);
}

#[test]
fn gauss4_flint_integrals() {
    high_order_fixtures::<f64>("64", 1e-12);
    high_order_fixtures::<Complex<f64>>("64", 1e-12);
    high_order_fixtures::<f32>("32", 2e-6);
    high_order_fixtures::<Complex<f32>>("32", 2e-6);
}

#[test]
fn gauss4_polynomial_exactness_and_remainder_sign() {
    let pieces = [piece(
        0.,
        1.,
        QuadratureAssumption::EightTimesContinuouslyDifferentiable,
    )];
    for degree in 0..=8 {
        let oracle = |q: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
            let value = match q.query {
                QuadratureQuery::Value => (0..degree)
                    .try_fold(D::point(r(1.)).unwrap(), |value, _| {
                        value.mul(&D::from(q.domain))
                    })?,
                QuadratureQuery::EighthDerivative => {
                    if degree == 8 {
                        D::point(r(40320.)).unwrap()
                    } else {
                        D::point(r(0.)).unwrap()
                    }
                }
                _ => return Ok(QuadratureObligation::Unavailable),
            };
            out.fill(Complex::new(value.set(), D::point(r(0.)).unwrap().set()));
            Ok(QuadratureObligation::Established)
        };
        let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
        let (report, certificate) = plan.generate(&tolerance(1e-14), 1).unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::ToleranceReached,
            "degree {degree}"
        );
        let certificate = certificate.unwrap();
        assert!(certificate.total[0].re.contains(1. / (degree + 1) as f64));
        assert_eq!(
            plan.verify(certificate.view()).unwrap(),
            QuadratureVerificationStatus::Verified
        );
    }
}

#[test]
fn gauss4_interrupted_queries_preserve_complete_cover_and_errors_rollback() {
    let pieces = [piece(
        0.,
        1.,
        QuadratureAssumption::EightTimesContinuouslyDifferentiable,
    )];
    let calls = AtomicUsize::new(0);
    let fail_at = AtomicUsize::new(usize::MAX);
    let oracle = |q, out: &mut [IntervalBox<f64>]| {
        if calls.fetch_add(1, Ordering::Relaxed) == fail_at.load(Ordering::Relaxed) {
            return Err(invalid(REQUEST, "injected C8 provider error"));
        }
        smooth::<f64>("exp", q, out)
    };
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut workspace = plan.workspace(ProofRequest::refinement(3)).unwrap();
    let mut certificate = QuadratureCertificate::new(3, 1).unwrap();
    for budget in 0..=15 {
        calls.store(0, Ordering::Relaxed);
        let saved = certificate.clone();
        let report = limited(budget, usize::MAX, |context| {
            plan.generate_into(&tolerance(0.), &mut certificate, &mut workspace, context)
        });
        assert_eq!(
            report.panels(),
            if budget < 5 { 0 } else { 1 + (budget - 5) / 10 }
        );
        if report.has_enclosure() {
            assert_eq!(
                plan.verify(certificate.view()).unwrap(),
                QuadratureVerificationStatus::Verified
            );
        } else {
            assert_eq!(certificate, saved);
        }
    }
    let saved = certificate.clone();
    calls.store(0, Ordering::Relaxed);
    fail_at.store(12, Ordering::Relaxed);
    assert!(
        plan.generate_into(&tolerance(0.), &mut certificate, &mut workspace, seq())
            .is_err()
    );
    assert_eq!(certificate, saved);
}

#[test]
fn incremental_totals_equal_fresh_exact_sums() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let plan = QuadratureVerifier::<Complex<f64>>::new(&pieces, 3, &square::<Complex<f64>>, opts())
        .unwrap();
    let mut workspace = plan.workspace(ProofRequest::refinement(65)).unwrap();
    for capacity in 1..=65 {
        let mut certificate = QuadratureCertificate::new(capacity, 3).unwrap();
        let report = plan
            .generate_into(&tolerance(0.), &mut certificate, &mut workspace, seq())
            .unwrap();
        assert_eq!(report.panels(), capacity);
        assert_eq!(
            report.termination(),
            QuadratureTermination::Limit(crate::RefinementLimit::Capacity)
        );
        let view = certificate.view();
        let mut fresh = [zero().unwrap(); 3];
        assert_eq!(
            proof::sum(view.local_bounds, &mut fresh, REQUEST).unwrap(),
            None
        );
        assert_eq!(view.total, fresh);
        assert_eq!(
            plan.verify(view).unwrap(),
            QuadratureVerificationStatus::Verified
        );
    }
}
fn piece<R: RealScalar>(a: f64, b: f64, assumption: QuadratureAssumption) -> QuadraturePiece<R> {
    QuadraturePiece {
        domain: D::<R>::new(r::<R>(a), r::<R>(b)).unwrap().set(),
        assumption,
    }
}
fn square<S: NumericScalar>(
    q: QuadratureRequest<S::Real>,
    out: &mut [IntervalBox<S::Real>],
) -> Result<QuadratureObligation> {
    let value = match q.query {
        QuadratureQuery::Value => D::from(q.domain).square()?,
        QuadratureQuery::SecondDerivative => D::point(r(2.)).unwrap(),
        QuadratureQuery::EighthDerivative => D::point(r(0.)).unwrap(),
        QuadratureQuery::Integral => return Ok(QuadratureObligation::Unavailable),
    };
    for (index, result) in out.iter_mut().enumerate() {
        let v = value.mul(&D::point(S::Real::try_from_usize_exact(index + 1)?)?)?;
        *result = Complex::new(
            v.set(),
            if !S::IS_REAL {
                v.neg()?.set()
            } else {
                D::point(S::Real::ZERO)?.set()
            },
        );
    }
    Ok(QuadratureObligation::Established)
}
fn basics<S: NumericScalar>() {
    for assumption in [
        QuadratureAssumption::IntegrableRange,
        QuadratureAssumption::TwiceContinuouslyDifferentiable,
        QuadratureAssumption::EightTimesContinuouslyDifferentiable,
    ] {
        let pieces = [piece(0., 1., assumption)];
        let plan = QuadratureVerifier::<S>::new(&pieces, 3, &square::<S>, opts()).unwrap();
        let max = 128;
        let request = ProofRequest::refinement(max);
        let bytes = plan
            .workspace_requirements(request)
            .unwrap()
            .required_bytes();
        let limited = QuadratureVerifier::<S>::new(
            &pieces,
            3,
            &square::<S>,
            opts().with_byte_limit(bytes - 1),
        )
        .unwrap();
        assert_eq!(
            limited.workspace(request).err().unwrap().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let mut work = plan.workspace(request).unwrap();
        let mut output = QuadratureCertificate::new(max, 3).unwrap();
        assert!(plan.verify_scoped(output.view(), &mut work, seq()).is_err());
        let report = plan
            .generate_into(&tolerance(0.025), &mut output, &mut work, seq())
            .unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::ToleranceReached
        );
        let view = output.view();
        assert_eq!(report.panels(), view.panels.len());
        for (index, value) in view.total.iter().enumerate() {
            let expected = D::point(S::Real::try_from_usize_exact(index + 1).unwrap())
                .unwrap()
                .div(&D::point(r(3.)).unwrap())
                .unwrap();
            assert!(value.re.contains_interval(&expected.set()));
            if !S::IS_REAL {
                assert!(value.im.contains_interval(&expected.neg().unwrap().set()));
            }
        }
        assert_eq!(
            plan.verify_scoped(view, &mut work, seq()).unwrap(),
            QuadratureVerificationStatus::Verified
        );
        if assumption == QuadratureAssumption::TwiceContinuouslyDifferentiable {
            assert_eq!(report.panels(), 1);
        }
        assert_eq!(
            plan.generate_into(&tolerance(0.025), &mut output, &mut work, seq())
                .unwrap(),
            report
        );
    }
}
#[test]
fn real32() {
    basics::<f32>();
}
#[test]
fn real64() {
    basics::<f64>();
}
#[test]
fn complex32() {
    basics::<Complex<f32>>();
}
#[test]
fn complex64() {
    basics::<Complex<f64>>();
}

#[test]
fn corruption_and_original_problem_replay() {
    let pieces = [
        piece(0., 0.5, QuadratureAssumption::IntegrableRange),
        piece(
            0.5,
            1.,
            QuadratureAssumption::TwiceContinuouslyDifferentiable,
        ),
    ];
    let calls = AtomicUsize::new(0);
    let oracle = |q, out: &mut [IntervalBox<f64>]| {
        calls.fetch_add(1, Ordering::Relaxed);
        square::<f64>(q, out)
    };
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut w = plan.workspace(ProofRequest::refinement(8)).unwrap();
    let mut c = QuadratureCertificate::new(8, 1).unwrap();
    assert!(
        plan.generate_into(&tolerance(1.), &mut c, &mut w, seq())
            .unwrap()
            .has_enclosure()
    );
    calls.store(0, Ordering::Relaxed);
    assert_eq!(
        plan.verify_scoped(c.view(), &mut w, seq()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
    assert_eq!(calls.load(Ordering::Relaxed), 4);
    let mut bad = c.clone();
    bad.local_bounds[0] = zero().unwrap();
    assert_eq!(
        plan.verify_scoped(bad.view(), &mut w, seq()).unwrap(),
        QuadratureVerificationStatus::Rejected(QuadratureRejection::LocalBound)
    );
    let mut bad = c.clone();
    bad.total[0] = zero().unwrap();
    assert_eq!(
        plan.verify_scoped(bad.view(), &mut w, seq()).unwrap(),
        QuadratureVerificationStatus::Rejected(QuadratureRejection::TotalBound)
    );
    for domain in [
        D::new(r(0.), r(0.25)).unwrap().set(),
        D::new(r(0.), r(0.75)).unwrap().set(),
        ExtendedInterval::entire(),
        ExtendedInterval::empty(),
    ] {
        let mut bad = c.clone();
        bad.panels[0].domain = domain;
        assert!(plan.verify_scoped(bad.view(), &mut w, seq()).is_err());
    }
    let mut bad = c.clone();
    bad.panels[0].piece = 1;
    assert!(plan.verify_scoped(bad.view(), &mut w, seq()).is_err());
    let changed = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        out.fill(crate::point_enclosure(100.).unwrap());
        Ok(QuadratureObligation::Established)
    };
    let other = QuadratureVerifier::<f64>::new(&pieces, 1, &changed, opts()).unwrap();
    assert_eq!(
        other.verify_scoped(c.view(), &mut w, seq()).unwrap(),
        QuadratureVerificationStatus::Rejected(QuadratureRejection::LocalBound)
    );
}

fn limited<T>(
    evaluations: usize,
    iterations: usize,
    f: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T>,
) -> T {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded()
        .with_limit(
            BudgetKind::Evaluation,
            Budget::try_from(evaluations.max(1)).unwrap(),
        )
        .with_limit(
            BudgetKind::Iteration,
            Budget::try_from(iterations.max(1)).unwrap(),
        );
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("quadrature", "test", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    if evaluations == 0 {
        assert!(
            ctx.try_consume(BudgetKind::Evaluation, 1)
                .unwrap()
                .is_granted()
        );
    }
    if iterations == 0 {
        assert!(
            ctx.try_consume(BudgetKind::Iteration, 1)
                .unwrap()
                .is_granted()
        );
    }
    f(&mut ctx).unwrap()
}
#[test]
fn interrupted_children_retain_whole_domain_and_errors_rollback() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let calls = AtomicUsize::new(0);
    let fail_at = AtomicUsize::new(usize::MAX);
    let oracle = |q, out: &mut [IntervalBox<f64>]| {
        let call = calls.fetch_add(1, Ordering::Relaxed);
        if call == fail_at.load(Ordering::Relaxed) {
            return Err(invalid(REQUEST, "injected provider error"));
        }
        square::<f64>(q, out)
    };
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut w = plan.workspace(ProofRequest::refinement(16)).unwrap();
    let mut c = QuadratureCertificate::new(16, 1).unwrap();
    for (budget, has, count) in [
        (0, false, 0),
        (1, true, 1),
        (2, true, 1),
        (3, true, 2),
        (4, true, 2),
    ] {
        calls.store(0, Ordering::Relaxed);
        let report = limited(budget, usize::MAX, |ctx| {
            plan.generate_into(&tolerance(0.), &mut c, &mut w, ctx)
        });
        assert_eq!(report.has_enclosure(), has);
        assert_eq!(report.panels(), count);
        assert_eq!(calls.load(Ordering::Relaxed), budget);
        if has {
            assert_eq!(
                plan.verify_scoped(c.view(), &mut w, seq()).unwrap(),
                QuadratureVerificationStatus::Verified
            );
        }
    }
    let saved = c.clone();
    calls.store(0, Ordering::Relaxed);
    fail_at.store(2, Ordering::Relaxed);
    assert!(
        plan.generate_into(&tolerance(0.), &mut c, &mut w, seq())
            .is_err()
    );
    assert_eq!(c.panels, saved.panels);
    assert_eq!(c.total, saved.total);
    assert_eq!(c.local_bounds, saved.local_bounds);
    assert_eq!(c.count, saved.count);
    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    // Iteration allowances are local to the quadrature epoch and set by options;
    // one iteration bounds the initial partition and stops the first refinement.
    fail_at.store(usize::MAX, Ordering::Relaxed);
    let budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let limited =
        QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts().with_budgets(budgets)).unwrap();
    let report = limited
        .generate_into(&tolerance(0.), &mut c, &mut w, seq())
        .unwrap();
    assert_eq!(report.panels(), 1);
    assert_eq!(
        report.termination(),
        QuadratureTermination::Inconclusive(QuadratureInconclusive::IterationLimit)
    );
    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Stopped);
    // Output that cannot hold the initial partition is rejected before any query.
    let saved = c.clone();
    let mut empty = QuadratureCertificate::new(0, 1).unwrap();
    calls.store(0, Ordering::Relaxed);
    assert_eq!(
        plan.generate_into(&tolerance(0.), &mut empty, &mut w, seq())
            .err()
            .unwrap()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(c, saved);
    // So is output with another component count, which stays unchanged.
    let mut wide = QuadratureCertificate::new(c.capacity(), 2).unwrap();
    let fresh = wide.clone();
    assert_eq!(
        plan.generate_into(&tolerance(0.), &mut wide, &mut w, seq())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(wide, fresh);
}

#[test]
fn missing_and_malformed_obligations_do_not_certify() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    for mode in 0..4 {
        let oracle = move |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
            match mode {
                0 => return Ok(QuadratureObligation::Unavailable),
                1 => {}
                2 => out.fill(Complex::new(
                    ExtendedInterval::entire(),
                    D::point(0.)?.set(),
                )),
                _ => out.fill(Complex::new(D::point(0.)?.set(), D::point(1.)?.set())),
            }
            Ok(QuadratureObligation::Established)
        };
        let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
        let mut w = plan.workspace(ProofRequest::refinement(1)).unwrap();
        let mut c = QuadratureCertificate::new(1, 1).unwrap();
        let result = plan.generate_into(&tolerance(1.), &mut c, &mut w, seq());
        if mode == 0 || mode == 2 {
            assert!(!result.unwrap().has_enclosure());
        } else {
            assert!(result.is_err());
        }
        assert_eq!(c.count, 0);
    }
}

#[test]
fn zero_domains_and_empty_vectors_make_no_callbacks() {
    let oracle = |_: QuadratureRequest<f64>, _: &mut [IntervalBox<f64>]| {
        Err(invalid(REQUEST, "must not call"))
    };
    for (a, b, components) in [(2., 2., 3), (0., 1., 0)] {
        let pieces = [piece(a, b, QuadratureAssumption::IntegrableRange)];
        let plan = QuadratureVerifier::<f64>::new(&pieces, components, &oracle, opts()).unwrap();
        let mut w = plan.workspace(ProofRequest::refinement(1)).unwrap();
        let mut c = QuadratureCertificate::new(1, components).unwrap();
        assert_eq!(
            plan.generate_into(&tolerance(0.), &mut c, &mut w, seq())
                .unwrap()
                .termination(),
            QuadratureTermination::ToleranceReached
        );
        assert_eq!(
            plan.verify_scoped(c.view(), &mut w, seq()).unwrap(),
            QuadratureVerificationStatus::Verified
        );
        assert!(c.total.iter().all(|x| *x == zero().unwrap()));
    }
    assert_eq!(
        QuadratureCertificate::<f64>::new(usize::MAX, 2)
            .err()
            .unwrap()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
}

#[test]
fn all_norms_and_cancellation() {
    for norm in [
        NormSelection::Infinity,
        NormSelection::L1,
        NormSelection::L2,
        NormSelection::RootMeanSquare,
    ] {
        let pieces = [
            piece(
                -1.,
                0.,
                QuadratureAssumption::TwiceContinuouslyDifferentiable,
            ),
            piece(
                0.,
                1.,
                QuadratureAssumption::TwiceContinuouslyDifferentiable,
            ),
        ];
        let oracle = |q: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
            let v = if q.query == QuadratureQuery::SecondDerivative {
                D::point(r(0.)).unwrap()
            } else {
                D::from(q.domain)
            };
            out.fill(Complex::new(v.set(), D::point(r(0.)).unwrap().set()));
            Ok(QuadratureObligation::Established)
        };
        let plan = QuadratureVerifier::<f64>::new(&pieces, 2, &oracle, opts()).unwrap();
        let mut w = plan.workspace(ProofRequest::refinement(2)).unwrap();
        let mut c = QuadratureCertificate::new(2, 2).unwrap();
        let t = ErrorTolerance::try_vector(&[0., 0.], 0., norm).unwrap();
        assert_eq!(
            plan.generate_into(&t, &mut c, &mut w, seq())
                .unwrap()
                .termination(),
            QuadratureTermination::ToleranceReached
        );
        assert_eq!(c.total, vec![zero().unwrap(); 2]);
        assert_eq!(
            plan.verify_scoped(c.view(), &mut w, seq()).unwrap(),
            QuadratureVerificationStatus::Verified
        );
    }
}

fn singular_and_tails<S: NumericScalar>() {
    let pieces = [
        piece(0., 0.25, QuadratureAssumption::IntegrableSingularity),
        piece(
            0.25,
            1.,
            QuadratureAssumption::TwiceContinuouslyDifferentiable,
        ),
    ];
    let oracle = |q: QuadratureRequest<S::Real>, out: &mut [IntervalBox<S::Real>]| {
        let x = D::from(q.domain);
        let v = match q.query {
            QuadratureQuery::EighthDerivative => return Ok(QuadratureObligation::Unavailable),
            QuadratureQuery::Value => x.sqrt()?.reciprocal()?,
            QuadratureQuery::SecondDerivative => D::point(r(0.75))
                .unwrap()
                .div(&x.square()?.mul(&x.sqrt()?)?)?,
            QuadratureQuery::Integral => {
                let (a, b) = bounded(x, REQUEST)?;
                D::point(b)?
                    .sqrt()?
                    .sub(&D::point(a)?.sqrt()?)?
                    .mul(&D::point(r(2.)).unwrap())?
            }
        };
        out.fill(Complex::new(
            v.set(),
            if !S::IS_REAL {
                v.set()
            } else {
                D::point(r(0.)).unwrap().set()
            },
        ));
        Ok(QuadratureObligation::Established)
    };
    let plan = QuadratureVerifier::<S>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(64)).unwrap();
    let mut c = QuadratureCertificate::new(64, 1).unwrap();
    let result = plan
        .generate_into(&tolerance(0.001), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        result.termination(),
        QuadratureTermination::ToleranceReached
    );
    assert!(c.total[0].re.contains(r(2.)));
    assert_eq!(c.panels[0].domain, pieces[0].domain);
    assert_eq!(
        plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
        QuadratureVerificationStatus::Verified
    );

    // Integral exp(-abs(x)) dx over the real line is exactly 2. Each tail
    // independently converges; no cancellation of divergent pieces is assumed.
    let pieces = [
        QuadraturePiece {
            domain: ExtendedInterval::new(None, Some(r(-4.))).unwrap(),
            assumption: QuadratureAssumption::ConvergentTail,
        },
        piece(
            -4.,
            0.,
            QuadratureAssumption::TwiceContinuouslyDifferentiable,
        ),
        piece(
            0.,
            4.,
            QuadratureAssumption::TwiceContinuouslyDifferentiable,
        ),
        QuadraturePiece {
            domain: ExtendedInterval::new(Some(r(4.)), None).unwrap(),
            assumption: QuadratureAssumption::ConvergentTail,
        },
    ];
    let oracle = |q: QuadratureRequest<S::Real>, out: &mut [IntervalBox<S::Real>]| {
        let x = if q.query == QuadratureQuery::Integral {
            D::point(r(-4.)).unwrap()
        } else if q.piece == 1 {
            D::from(q.domain)
        } else {
            D::from(q.domain).neg()?
        };
        let v = x.exp()?.hull();
        out.fill(Complex::new(
            v.set(),
            if !S::IS_REAL {
                v.neg()?.set()
            } else {
                D::point(r(0.)).unwrap().set()
            },
        ));
        Ok(QuadratureObligation::Established)
    };
    let plan = QuadratureVerifier::<S>::new(&pieces, 1, &oracle, opts()).unwrap();
    let result = plan
        .generate_into(&tolerance(0.003), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        result.termination(),
        QuadratureTermination::ToleranceReached
    );
    assert!(c.total[0].re.contains(r(2.)));
    if !S::IS_REAL {
        assert!(c.total[0].im.contains(r(-2.)));
    }
    assert_eq!(
        plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
    // Removing a tail or subdividing a fixed tail is structural corruption.
    let v = c.view();
    assert!(
        plan.verify_scoped(
            QuadratureCertificateView {
                panels: &v.panels[1..],
                local_bounds: &v.local_bounds[1..],
                total: v.total
            },
            &mut work,
            seq()
        )
        .is_err()
    );
}
#[test]
fn singular_tail_real32() {
    singular_and_tails::<f32>();
}
#[test]
fn singular_tail_real64() {
    singular_and_tails::<f64>();
}
#[test]
fn singular_tail_complex32() {
    singular_and_tails::<Complex<f32>>();
}
#[test]
fn singular_tail_complex64() {
    singular_and_tails::<Complex<f64>>();
}

#[test]
fn unseen_peak_cannot_be_certified_from_endpoint_samples() {
    // A continuous triangular peak of area one, supported far from 0, 1/2, 1.
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let oracle = |q: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        if q.query != QuadratureQuery::Value {
            return Ok(QuadratureObligation::Unavailable);
        }
        let v = D::point(r(1.))
            .unwrap()
            .sub(
                &D::from(q.domain)
                    .sub(&D::point(r(0.3125)).unwrap())?
                    .abs()?
                    .mul(&D::point(r(1024.)).unwrap())?,
            )?
            .mul(&D::point(r(1024.)).unwrap())?;
        let (lo, hi) = bounded(v, REQUEST)?;
        out.fill(Complex::new(
            D::new(lo.max(0.), hi.max(0.))?.set(),
            D::point(r(0.)).unwrap().set(),
        ));
        Ok(QuadratureObligation::Established)
    };
    for x in [0., 0.5, 1.] {
        let mut out = [sentinel()];
        crate::shell::run_local(seq().into(), identity(), SolveBudgets::unbounded(), |_| {
            oracle(
                QuadratureRequest {
                    piece: 0,
                    domain: D::point(r(x)).unwrap().set(),
                    query: QuadratureQuery::Value,
                },
                &mut out,
            )
        })
        .unwrap();
        assert_eq!(out[0], zero().unwrap());
    }
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(128)).unwrap();
    let mut c = QuadratureCertificate::new(128, 1).unwrap();
    let report = plan
        .generate_into(&tolerance(0.025), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        report.termination(),
        QuadratureTermination::ToleranceReached
    );
    assert!(c.total[0].re.contains(1.));
    assert!(!c.total[0].re.contains(0.));
    assert_eq!(
        plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
    let derivative_pieces = [piece(
        0.,
        1.,
        QuadratureAssumption::TwiceContinuouslyDifferentiable,
    )];
    let derivative =
        QuadratureVerifier::<f64>::new(&derivative_pieces, 1, &oracle, opts()).unwrap();
    let report = derivative
        .generate_into(&tolerance(0.1), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        report.termination(),
        QuadratureTermination::Inconclusive(QuadratureInconclusive::Reason(
            QuadratureProofFailure::MissingObligation
        ))
    );
    assert!(!report.has_enclosure());
}

#[test]
fn fixed_bounds_resolution_and_partition_validation() {
    let oracle = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        out.fill(Complex::new(
            D::new(r(0.), r(1.)).unwrap().set(),
            D::point(r(0.)).unwrap().set(),
        ));
        Ok(QuadratureObligation::Established)
    };
    for (pieces, expected) in [
        (
            [piece(0., 1., QuadratureAssumption::IntegrableSingularity)],
            QuadratureTermination::FixedBounds,
        ),
        (
            [piece(
                1.,
                f64::from_bits(1_f64.to_bits() + 1),
                QuadratureAssumption::IntegrableRange,
            )],
            QuadratureTermination::Limit(crate::RefinementLimit::Resolution),
        ),
    ] {
        let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
        let mut work = plan.workspace(ProofRequest::refinement(8)).unwrap();
        let mut c = QuadratureCertificate::new(8, 1).unwrap();
        let report = plan
            .generate_into(&tolerance(0.), &mut c, &mut work, seq())
            .unwrap();
        assert!(report.has_enclosure());
        assert_eq!(report.termination(), expected);
        assert_eq!(
            plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
            QuadratureVerificationStatus::Verified
        );
    }
    for pieces in [
        vec![],
        vec![
            piece(2., 3., QuadratureAssumption::IntegrableRange),
            piece(0., 1., QuadratureAssumption::IntegrableRange),
        ],
        vec![
            piece(0., 0.75, QuadratureAssumption::IntegrableRange),
            piece(0.5, 1., QuadratureAssumption::IntegrableRange),
        ],
        vec![
            piece(0., 0.25, QuadratureAssumption::IntegrableRange),
            piece(0.5, 1., QuadratureAssumption::IntegrableRange),
        ],
        vec![piece(0., 1., QuadratureAssumption::ConvergentTail)],
        vec![
            piece(0., 0., QuadratureAssumption::IntegrableRange),
            piece(0., 1., QuadratureAssumption::IntegrableRange),
        ],
    ] {
        let error = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts())
            .err()
            .unwrap();
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, .. } if argument == scioxide_core::argument!("pieces"))
        );
    }
    let pieces = [QuadraturePiece {
        domain: ExtendedInterval::entire(),
        assumption: QuadratureAssumption::ConvergentTail,
    }];
    assert!(QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).is_err());
}

#[test]
fn subnormal_and_range_limits_are_explicit() {
    let oracle = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        out.fill(crate::point_enclosure(0.5)?);
        Ok(QuadratureObligation::Established)
    };
    let tiny = f64::from_bits(1);
    let pieces = [piece(0., tiny, QuadratureAssumption::IntegrableRange)];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(1)).unwrap();
    let mut c = QuadratureCertificate::new(1, 1).unwrap();
    let report = plan
        .generate_into(&tolerance(tiny), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        report.termination(),
        QuadratureTermination::ToleranceReached
    );
    assert_eq!(c.total[0].re, D::new(r(0.), r(tiny)).unwrap().set());
    assert_eq!(
        plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
    // The width 2·MAX overflows, so the whole line has no finite bound; its halves
    // do, and bisection certifies the representable integral MAX.
    let pieces = [piece(
        -f64::MAX,
        f64::MAX,
        QuadratureAssumption::IntegrableRange,
    )];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(2)).unwrap();
    let mut c = QuadratureCertificate::new(2, 1).unwrap();
    let report = plan
        .generate_into(&tolerance(1.), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        report.termination(),
        QuadratureTermination::ToleranceReached
    );
    assert_eq!(report.panels(), 2);
    assert_eq!(c.total[0].re, D::point(r(f64::MAX)).unwrap().set());
    assert_eq!(
        plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
    // Twice that integral exceeds the range: finite halves, unbounded total.
    let unit = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        out.fill(crate::point_enclosure(1.)?);
        Ok(QuadratureObligation::Established)
    };
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &unit, opts()).unwrap();
    let saved = c.clone();
    let report = plan
        .generate_into(&tolerance(1.), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        report.termination(),
        QuadratureTermination::Inconclusive(QuadratureInconclusive::Unbounded)
    );
    assert!(!report.has_enclosure());
    assert_eq!(c, saved);
}

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(3),
        scioxide_core::algorithm_id!("quadrature", "tests", 1),
        0,
        0,
    )
}

fn independent_fixtures<S: NumericScalar>(width: &str) {
    let mut checked = 0;
    for line in include_str!("../../tests/data/quadrature.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[1] != width {
            continue;
        }
        let name = fields[0];
        let decode = |text: &str| {
            let bits = u64::from_str_radix(text, 16).unwrap();
            r::<S::Real>(if width == "32" {
                f32::from_bits(bits as u32) as f64
            } else {
                f64::from_bits(bits)
            })
        };
        let expected = D::new(decode(fields[2]), decode(fields[3])).unwrap();
        let regular = QuadratureAssumption::TwiceContinuouslyDifferentiable;
        let pieces = match name {
            "inverse_sqrt" => vec![
                piece(0., 0.25, QuadratureAssumption::IntegrableSingularity),
                piece(0.25, 1., regular),
            ],
            "exp_tail" => vec![QuadraturePiece {
                domain: ExtendedInterval::new(Some(r(4.)), None).unwrap(),
                assumption: QuadratureAssumption::ConvergentTail,
            }],
            "log_singular" => vec![piece(0., 1., QuadratureAssumption::IntegrableSingularity)],
            _ => vec![piece(0., 1., regular)],
        };
        let oracle = |q: QuadratureRequest<S::Real>, out: &mut [IntervalBox<S::Real>]| {
            let x = D::from(q.domain);
            let v = match (name, q.query) {
                ("square", QuadratureQuery::Value) => x.square()?,
                ("square", QuadratureQuery::SecondDerivative) => D::point(r(2.)).unwrap(),
                ("exp", _) => x.exp()?.hull(),
                ("sin", QuadratureQuery::Value) => x.sin()?.hull(),
                ("sin", QuadratureQuery::SecondDerivative) => x.sin()?.hull().neg()?,
                ("rational", QuadratureQuery::Value) => {
                    x.add(&D::point(r(1.)).unwrap())?.reciprocal()?
                }
                ("rational", QuadratureQuery::SecondDerivative) => {
                    let y = x.add(&D::point(r(1.)).unwrap())?;
                    D::point(r(2.)).unwrap().div(&y.square()?.mul(&y)?)?
                }
                ("inverse_sqrt", QuadratureQuery::Value) => x.sqrt()?.reciprocal()?,
                ("inverse_sqrt", QuadratureQuery::SecondDerivative) => {
                    D::point(r(0.75))
                        .unwrap()
                        .div(&x.square()?.mul(&x.sqrt()?)?)?
                }
                ("inverse_sqrt", QuadratureQuery::Integral) => {
                    let (a, b) = bounded(x, REQUEST)?;
                    D::point(b)?
                        .sqrt()?
                        .sub(&D::point(a)?.sqrt()?)?
                        .mul(&D::point(r(2.)).unwrap())?
                }
                ("exp_tail", QuadratureQuery::Integral) => D::point(r(-4.)).unwrap().exp()?.hull(),
                ("log_singular", QuadratureQuery::Integral) => {
                    let (a, b) = bounded(x, REQUEST)?;
                    let antiderivative = |t| -> Result<D<S::Real>> {
                        if t == S::Real::ZERO {
                            D::point(t)
                        } else {
                            D::point(t)?
                                .mul(&D::point(t)?.ln()?.hull())?
                                .sub(&D::point(t)?)
                        }
                    };
                    antiderivative(b)?.sub(&antiderivative(a)?)?
                }
                _ => return Ok(QuadratureObligation::Unavailable),
            };
            out.fill(Complex::new(
                v.set(),
                if !S::IS_REAL {
                    v.neg()?.set()
                } else {
                    D::point(r(0.)).unwrap().set()
                },
            ));
            Ok(QuadratureObligation::Established)
        };
        let plan = QuadratureVerifier::<S>::new(&pieces, 1, &oracle, opts()).unwrap();
        let mut work = plan.workspace(ProofRequest::refinement(128)).unwrap();
        let mut c = QuadratureCertificate::new(128, 1).unwrap();
        let report = plan
            .generate_into(&tolerance(0.002), &mut c, &mut work, seq())
            .unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::ToleranceReached,
            "{name}"
        );
        assert!(
            c.total[0].re.contains_interval(&expected.set()),
            "{name}: {:?} / {expected:?}",
            c.total[0]
        );
        if !S::IS_REAL {
            assert!(
                c.total[0]
                    .im
                    .contains_interval(&expected.neg().unwrap().set())
            );
        }
        assert_eq!(
            plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
            QuadratureVerificationStatus::Verified,
            "{name}"
        );
        checked += 1;
    }
    assert_eq!(checked, 7);
}
#[test]
fn independent_real32() {
    independent_fixtures::<f32>("32");
}
#[test]
fn independent_real64() {
    independent_fixtures::<f64>("64");
}
#[test]
fn independent_complex32() {
    independent_fixtures::<Complex<f32>>("32");
}
#[test]
fn independent_complex64() {
    independent_fixtures::<Complex<f64>>("64");
}

#[derive(Debug)]
struct TickClock(AtomicUsize);
impl scioxide_core::Clock for TickClock {
    fn now(
        &self,
    ) -> core::result::Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_nanos(self.0.fetch_add(1, Ordering::Relaxed) as u64),
        ))
    }
}
#[test]
fn deadlines_and_nested_work_share_caller_limits() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    struct NestedSquare;
    impl QuadratureOracle<f64> for NestedSquare {
        fn enclose(
            &self,
            q: QuadratureRequest<f64>,
            out: &mut [IntervalBox<f64>],
        ) -> Result<QuadratureObligation> {
            square::<f64>(q, out)
        }
        fn enclose_scoped(
            &self,
            q: QuadratureRequest<f64>,
            out: &mut [IntervalBox<f64>],
            ctx: &mut SolveContext<'_, '_>,
        ) -> Result<QuadratureObligation> {
            if !ctx.record_provider(EvaluationRole::Function)?.is_granted() {
                return Ok(QuadratureObligation::Unavailable);
            }
            self.enclose(q, out)
        }
    }
    let oracle = NestedSquare;
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(16)).unwrap();
    let mut c = QuadratureCertificate::new(16, 1).unwrap();
    let report = limited(1, usize::MAX, |ctx| {
        let report = plan.generate_into(&tolerance(1.), &mut c, &mut work, &mut *ctx)?;
        // The quadrature runs in a child epoch: its admitted query and the oracle's
        // nested charge both reach the caller's shared limit.
        assert_eq!(ctx.counts().own().total_providers().unwrap(), 0);
        assert_eq!(ctx.counts().total()?.total_providers()?, 1);
        Ok(report)
    });
    assert!(!report.has_enclosure());
    for ticks in [0, 2, 10] {
        let clock = TickClock(AtomicUsize::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let deadline = scioxide_core::Deadline::at(
            &clock_binding,
            scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_nanos(ticks),
            ),
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut ctx = state
            .begin_solve(
                ExecutionContext::sequential()
                    .with_deadline(deadline)
                    .unwrap(),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let report = plan
            .generate_into(&tolerance(0.), &mut c, &mut work, &mut ctx)
            .unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::Inconclusive(QuadratureInconclusive::Stopped(
                ExecutionStopReason::DeadlineReached
            ))
        );
        if report.has_enclosure() {
            assert_eq!(
                plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
                QuadratureVerificationStatus::Verified
            );
        }
    }
}
#[test]
fn cancellation_and_verifier_work_limits() {
    let pieces = [piece(
        0.,
        1.,
        QuadratureAssumption::TwiceContinuouslyDifferentiable,
    )];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &square::<f64>, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(4)).unwrap();
    let mut c = QuadratureCertificate::new(4, 1).unwrap();
    let _ = plan
        .generate_into(&tolerance(0.01), &mut c, &mut work, seq())
        .unwrap();
    for evaluations in 0..3 {
        let status = limited(evaluations, usize::MAX, |ctx| {
            plan.verify_scoped(c.view(), &mut work, ctx)
        });
        assert_eq!(
            status,
            QuadratureVerificationStatus::Inconclusive(QuadratureInconclusive::EvaluationLimit)
        );
    }
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity(),
            &mut budgets,
        )
        .unwrap();
    assert!(matches!(
        plan.verify_scoped(c.view(), &mut work, &mut ctx).unwrap(),
        QuadratureVerificationStatus::Inconclusive(QuadratureInconclusive::Stopped(_))
    ));
    assert_eq!(ctx.counts().total().unwrap().total_providers().unwrap(), 0);
}

struct FaultyExecutor(bool);
impl scioxide_core::Executor for FaultyExecutor {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::MIN
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<()> {
        if self.0 {
            scioxide_core::SequentialExecutor.execute(context, plan, operation)?;
            scioxide_core::SequentialExecutor.execute(context, plan, operation)
        } else {
            scioxide_core::SequentialExecutor.execute(context, plan, &|_, (), _| Ok(()))
        }
    }
}
#[test]
fn malformed_executor_dispatch_is_rejected() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &square::<f64>, opts()).unwrap();
    let mut w = plan.workspace(ProofRequest::refinement(1)).unwrap();
    let mut c = QuadratureCertificate::new(1, 1).unwrap();
    let _ = plan
        .generate_into(&tolerance(1.), &mut c, &mut w, seq())
        .unwrap();
    let blank = QuadratureVerificationStatus::NotRun;
    for duplicate in [false, true] {
        let executor = FaultyExecutor(duplicate);
        let mut batch = plan.workspace(ProofRequest::proofs(2)).unwrap();
        let mut statuses = [blank; 2];
        assert!(
            plan.verify_batch_into(
                &[c.view(); 2],
                &mut statuses,
                &mut batch,
                ExecutionContext::new(&executor),
            )
            .is_err()
        );
        assert_eq!(statuses, [blank; 2]);
        assert_eq!(
            batch.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
    }
    let error = plan
        .verify_batch_into(
            &[c.view(); 2],
            &mut [blank; 1],
            &mut QuadratureWorkspace::new(),
            seq(),
        )
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
}
#[cfg(feature = "std")]
#[test]
fn executor_equivalence_and_stable_budget_partitioning() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    let pool = ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
    let pieces = [piece(
        0.,
        1.,
        QuadratureAssumption::TwiceContinuouslyDifferentiable,
    )];
    let plan = QuadratureVerifier::<Complex<f64>>::new(&pieces, 2, &square::<Complex<f64>>, opts())
        .unwrap();
    let mut w = plan.workspace(ProofRequest::refinement(2)).unwrap();
    let mut c = QuadratureCertificate::new(2, 2).unwrap();
    let _ = plan
        .generate_into(&tolerance(0.01), &mut c, &mut w, seq())
        .unwrap();
    for limit in [1, 2, 3, 8, 9, 100] {
        let mut reference = None;
        for workers in [1, 2, 3] {
            let execution = if workers == 1 {
                ExecutionContext::sequential()
            } else {
                ExecutionContext::new(&pool)
            };
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
            let mut ctx = state
                .begin_solve(execution, identity(), &mut budgets)
                .unwrap();
            let policy = if workers == 1 {
                ExecutionPolicy::Sequential
            } else {
                ExecutionPolicy::Parallel(NonZeroUsize::new(workers).unwrap())
            };
            let plan = QuadratureVerifier::<Complex<f64>>::new(
                &pieces,
                2,
                &square::<Complex<f64>>,
                opts().with_execution_policy(policy),
            )
            .unwrap();
            let mut batch = plan.workspace(ProofRequest::proofs(3)).unwrap();
            let mut reports = [QuadratureVerificationStatus::NotRun; 3];
            plan.verify_batch_into(&[c.view(); 3], &mut reports, &mut batch, &mut ctx)
                .unwrap();
            let counts = ctx.counts();
            if let Some((old, old_counts)) = &reference {
                assert_eq!(&reports, old);
                assert_eq!(&counts, old_counts);
            } else {
                reference = Some((reports, counts));
            }
        }
    }
}

#[test]
fn unallocatable_capacity_fails_cleanly() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &square::<f64>, opts()).unwrap();
    let huge = 1 << 55;
    assert_eq!(
        plan.workspace(ProofRequest::refinement(huge))
            .err()
            .unwrap()
            .kind(),
        SciOxideErrorKind::AllocationFailure
    );
    assert_eq!(
        QuadratureCertificate::<f64>::new(huge, 1)
            .err()
            .unwrap()
            .kind(),
        SciOxideErrorKind::AllocationFailure
    );
    let mut workspace = QuadratureWorkspace::<f64>::new();
    let requirements = plan
        .workspace_requirements(ProofRequest::proofs(1 << 50))
        .unwrap();
    assert_eq!(
        workspace.try_reserve(&requirements).err().unwrap().kind(),
        SciOxideErrorKind::AllocationFailure
    );
}

#[test]
fn reused_workspace_grows_only_requested_component_lanes() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let oracle = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        out.fill(point_enclosure(1.)?);
        Ok(QuadratureObligation::Established)
    };
    let small = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let large = QuadratureVerifier::<f64>::new(&pieces, 64, &oracle, opts()).unwrap();
    let limit = small
        .workspace_requirements(ProofRequest::proofs(64))
        .unwrap()
        .required_bytes()
        + large
            .workspace_requirements(ProofRequest::proofs(1))
            .unwrap()
            .required_bytes();
    let limited =
        QuadratureVerifier::<f64>::new(&pieces, 64, &oracle, opts().with_byte_limit(limit))
            .unwrap();
    let mut workspace = small.workspace(ProofRequest::proofs(64)).unwrap();
    let mut certificate = QuadratureCertificate::new(1, 64).unwrap();
    let generated = large
        .generate_into(
            &tolerance(0.),
            &mut certificate,
            &mut QuadratureWorkspace::new(),
            seq(),
        )
        .unwrap();
    assert!(generated.has_enclosure());
    assert_eq!(
        limited
            .verify_scoped(certificate.view(), &mut workspace, seq())
            .unwrap(),
        QuadratureVerificationStatus::Verified
    );
    {
        assert_eq!(
            limited
                .verify_scoped(certificate.view(), &mut workspace, seq())
                .unwrap(),
            QuadratureVerificationStatus::Verified
        );
    };
}

#[test]
fn requirement_overflow_replaces_last_attempt_in_every_workspace_form() {
    let calls = AtomicUsize::new(0);
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let oracle = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        calls.fetch_add(1, Ordering::Relaxed);
        out.fill(point_enclosure(1.)?);
        Ok(QuadratureObligation::Established)
    };
    let warm = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let huge = QuadratureVerifier::<f64>::new(&pieces, usize::MAX / 8, &oracle, opts()).unwrap();
    let mut workspace = QuadratureWorkspace::new();
    let mut certificate = QuadratureCertificate::new(1, 1).unwrap();
    let mut statuses = [QuadratureVerificationStatus::NotRun];
    for form in 0..3 {
        let generated = warm
            .generate_into(&tolerance(0.), &mut certificate, &mut workspace, seq())
            .unwrap();
        assert_eq!(
            generated.termination(),
            QuadratureTermination::ToleranceReached
        );
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Succeeded
        );
        let previous = certificate.clone();
        calls.store(0, Ordering::Relaxed);
        let error = match form {
            0 => huge
                .generate_into(&tolerance(0.), &mut certificate, &mut workspace, seq())
                .map(|_| ()),
            1 => huge
                .verify_scoped(certificate.view(), &mut workspace, seq())
                .map(|_| ()),
            _ => {
                huge.verify_batch_into(&[certificate.view()], &mut statuses, &mut workspace, seq())
            }
        }
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::SizeOverflow);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(statuses, [QuadratureVerificationStatus::NotRun]);
        assert_eq!(certificate.panels, previous.panels);
        assert_eq!(certificate.local_bounds, previous.local_bounds);
        assert_eq!(certificate.total, previous.total);
        assert_eq!(certificate.count, previous.count);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
        assert!(attempt.epoch().is_none());
        assert_eq!(
            attempt.counts().total().unwrap().total_providers().unwrap(),
            0
        );
    }
}

/// The natural interval extension of `1 / (x·x − x + 1)`. Dependency overestimation
/// puts zero in the denominator enclosure on `[0, 1]`, `[1, 2]` and their hull.
fn rational(
    q: QuadratureRequest<f64>,
    out: &mut [IntervalBox<f64>],
) -> Result<QuadratureObligation> {
    if q.query != QuadratureQuery::Value {
        return Ok(QuadratureObligation::Unavailable);
    }
    let x = D::from(q.domain);
    let value = x
        .mul(&x)?
        .sub(&x)?
        .add(&D::point(r(1.)).unwrap())?
        .reciprocal()?;
    out.fill(Complex::new(value.set(), D::point(r(0.)).unwrap().set()));
    Ok(QuadratureObligation::Established)
}

#[test]
fn panels_without_finite_bounds_are_bisected_before_certification() {
    // ∫₀² dx / (x² − x + 1) = π/√3.
    const EXACT: f64 = 1.813_799_364_234_217_8;
    let regular = QuadratureAssumption::IntegrableRange;
    let mut reference = None;
    for edges in [
        &[0., 2.][..],
        &[0., 1., 2.][..],
        &[0., 0.5, 1., 1.5, 2.][..],
    ] {
        let pieces: Vec<_> = edges
            .windows(2)
            .map(|e| piece(e[0], e[1], regular))
            .collect();
        let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &rational, opts()).unwrap();
        let (report, certificate) = plan.generate(&tolerance(1e-2), 4096).unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::ToleranceReached
        );
        let certificate = certificate.unwrap();
        let total = certificate.view().total[0];
        assert!(total.re.contains(EXACT), "{total:?}");
        assert!(adaptive::radius_scale(total).unwrap().0 <= 1e-2);
        assert_eq!(
            plan.verify(certificate.view()).unwrap(),
            QuadratureVerificationStatus::Verified
        );
        // Bisection reaches the four-piece cover, so refinement then coincides.
        let summary = (report.panels(), total);
        assert_eq!(*reference.get_or_insert(summary), summary);
    }
}

#[test]
fn pending_panels_publish_nothing_when_refinement_stops() {
    let pieces = [piece(0., 2., QuadratureAssumption::IntegrableRange)];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &rational, opts()).unwrap();
    let previous = QuadratureVerifier::<f64>::new(&pieces, 1, &square::<f64>, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(3)).unwrap();
    let mut c = QuadratureCertificate::new(3, 1).unwrap();
    assert_eq!(
        previous
            .generate_into(&tolerance(0.), &mut c, &mut work, seq())
            .unwrap()
            .panels(),
        3
    );
    let saved = c.clone();
    // Three panels hold [0, 1/2], [1/2, 1] and the pending [1, 2].
    let report = plan
        .generate_into(&tolerance(1e-2), &mut c, &mut work, seq())
        .unwrap();
    assert_eq!(
        report.termination(),
        QuadratureTermination::Limit(crate::RefinementLimit::Capacity)
    );
    assert!(!report.has_enclosure());
    assert_eq!(c, saved);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    // A budget stop keeps its own termination. The four-panel cover takes ten
    // iterations: one per local bound and one per refinement step.
    let mut c = QuadratureCertificate::new(4, 1).unwrap();
    let fresh = c.clone();
    for iterations in [1, 7, 9, 10] {
        let budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::try_from(iterations).unwrap());
        let plan =
            QuadratureVerifier::<f64>::new(&pieces, 1, &rational, opts().with_budgets(budgets))
                .unwrap();
        let report = plan
            .generate_into(&tolerance(1e-2), &mut c, &mut work, seq())
            .unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::Inconclusive(QuadratureInconclusive::IterationLimit)
        );
        assert_eq!(report.panels(), if iterations == 10 { 4 } else { 0 });
        if iterations < 10 {
            assert_eq!(c, fresh);
        }
    }
    assert_ne!(c, fresh);
    assert_eq!(
        plan.verify_scoped(c.view(), &mut work, seq()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
}

#[test]
fn zero_tolerance_ties_cannot_starve_pending_panels() {
    // f(x) = x, with no finite enclosure on panels wider than one half.
    let oracle = |q: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        let (a, b) = bounded(D::from(q.domain), REQUEST)?;
        let value = if b - a > 0.5 {
            ExtendedInterval::entire()
        } else {
            q.domain
        };
        out.fill(Complex::new(value, D::point(r(0.)).unwrap().set()));
        Ok(QuadratureObligation::Established)
    };
    let regular = QuadratureAssumption::IntegrableRange;
    // The finite first piece wins every domain-order tie at zero tolerance.
    let pieces = [piece(0., 0.5, regular), piece(0.5, 1.5, regular)];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    let (report, certificate) = plan.generate(&tolerance(0.), 16).unwrap();
    assert_eq!(
        report.termination(),
        QuadratureTermination::Limit(crate::RefinementLimit::Capacity)
    );
    assert_eq!(report.panels(), 16);
    let certificate = certificate.unwrap();
    assert!(certificate.view().total[0].re.contains(1.125));
    assert_eq!(
        plan.verify(certificate.view()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
}

#[test]
fn unbounded_fixed_and_unsplittable_pieces_stay_inconclusive() {
    let entire = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        out.fill(Complex::new(
            ExtendedInterval::entire(),
            D::point(r(0.)).unwrap().set(),
        ));
        Ok(QuadratureObligation::Established)
    };
    let narrow = f64::from_bits(1_f64.to_bits() + 1);
    for pieces in [
        vec![piece(0., 1., QuadratureAssumption::IntegrableSingularity)],
        vec![
            piece(0., 1., QuadratureAssumption::IntegrableRange),
            piece(1., 2., QuadratureAssumption::IntegrableSingularity),
        ],
        vec![piece(1., narrow, QuadratureAssumption::IntegrableRange)],
        vec![piece(
            1.,
            narrow,
            QuadratureAssumption::EightTimesContinuouslyDifferentiable,
        )],
    ] {
        let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &entire, opts()).unwrap();
        let (report, certificate) = plan.generate(&tolerance(1.), 8).unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::Inconclusive(QuadratureInconclusive::Unbounded)
        );
        assert_eq!((report.panels(), certificate), (0, None));
    }
}

#[test]
fn fixed_bounds_include_outward_rounding_of_regular_totals() {
    let oracle = |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        out.fill(crate::point_enclosure(1.)?);
        Ok(QuadratureObligation::Established)
    };
    let tiny = 1. / (1_u64 << 60) as f64;
    let regular = QuadratureAssumption::IntegrableRange;
    let pieces = [piece(-tiny, 0., regular), piece(0., 1., regular)];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, opts()).unwrap();
    // Point local bounds, whose exact sum 1 + 2⁻⁶⁰ rounds outward.
    let (report, certificate) = plan.generate(&tolerance(0.), 8).unwrap();
    assert_eq!(report.termination(), QuadratureTermination::FixedBounds);
    assert_eq!(report.panels(), 2);
    let certificate = certificate.unwrap();
    let total = certificate.view().total[0];
    assert_eq!(total.re, D::new(r(1.), r(1. + f64::EPSILON)).unwrap().set());
    assert!(adaptive::radius_scale(total).unwrap().0 > 0.);
    assert_eq!(
        plan.verify(certificate.view()).unwrap(),
        QuadratureVerificationStatus::Verified
    );
}

#[test]
fn certificate_equality_and_debug_cover_only_the_published_view() {
    let pieces = [piece(0., 1., QuadratureAssumption::IntegrableRange)];
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &square::<f64>, opts()).unwrap();
    let mut work = plan.workspace(ProofRequest::refinement(8)).unwrap();
    let mut refined = QuadratureCertificate::new(8, 1).unwrap();
    let mut coarse = refined.clone();
    assert_eq!(refined, coarse);
    for (certificate, goals) in [(&mut refined, &[0., 1.][..]), (&mut coarse, &[1.][..])] {
        for &goal in goals {
            let report = plan
                .generate_into(&tolerance(goal), certificate, &mut work, seq())
                .unwrap();
            assert_eq!(report.panels(), if goal == 0. { 8 } else { 1 });
        }
    }
    // Unpublished storage still holds the earlier eight-panel proposal.
    assert_ne!(refined.panels[1], coarse.panels[1]);
    assert_eq!(refined, coarse);
    assert_eq!(alloc::format!("{refined:?}"), alloc::format!("{coarse:?}"));
    let mut small = QuadratureCertificate::new(4, 1).unwrap();
    let _ = plan
        .generate_into(&tolerance(1.), &mut small, &mut work, seq())
        .unwrap();
    assert_eq!(small.view().panels, coarse.view().panels);
    assert_ne!(small, coarse);
    assert_ne!(
        QuadratureCertificate::<f64>::new(8, 1).unwrap(),
        QuadratureCertificate::<f64>::new(8, 2).unwrap()
    );
}
