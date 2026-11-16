//! One-shot proofs equal their configured forms and never publish partial results.
#![cfg(test)]

use scioxide_core::{
    AlgebraicScalar, Complex, ConvergenceReason, ErrorTolerance, ExecutionContext, Interval,
    NumericScalar, RealScalar, SciOxideError, SciOxideErrorKind, SciOxideErrorView, StepTolerance,
};
use scioxide_linalg::{LuDecomposition, Matrix};
use scioxide_validated::{DecoratedInterval as D, *};

fn r<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn s<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(r(re), r(im))
}
fn seq() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}
fn convergence<T: core::fmt::Debug>(result: Result<T, SciOxideError>, expected: ConvergenceReason) {
    assert!(
        matches!(result.unwrap_err().view(), SciOxideErrorView::ConvergenceFailure { reason, .. } if reason == expected)
    );
}

fn same_bits<S: NumericScalar>(a: &[S], b: &[S]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(
            a.real_component().into_f64().to_bits(),
            b.real_component().into_f64().to_bits()
        );
        assert_eq!(
            a.imaginary_component().into_f64().to_bits(),
            b.imaginary_component().into_f64().to_bits()
        );
    }
}

fn linear<S: NumericScalar>() {
    for (entries, rhs, expected) in [
        (
            [s::<S>(3., 0.), s(1., 0.), s(1., 0.), s(2., 0.)],
            [s(1., 0.), s(1., 0.)],
            [s(0.2, 0.), s(0.4, 0.)],
        ),
        (
            [s(1., 0.), s(1., 0.), s(0., 0.), s(1., 0.)],
            [s(3., 1.), s(2., -1.)],
            [s(1., 2.), s(2., -1.)],
        ),
    ] {
        let a = Matrix::from_vec(2, 2, entries.to_vec()).unwrap();
        let found = solve(&a, &rhs).unwrap();
        let factor = LuDecomposition::factor(&a).unwrap();
        let system = DenseIntervalOperator::point(a.view()).unwrap();
        let inverse = LuIntervalInverse::from_dense(&factor, ValidatedOptions::new()).unwrap();
        let verifier = LinearVerifier::new(&system, &inverse, ValidatedOptions::new()).unwrap();
        let boxes = rhs.map(|x| point_enclosure(x).unwrap());
        let (report, configured) = verifier.solve(&boxes).unwrap();
        assert!(report.is_verified());
        let configured = configured.unwrap();
        same_bits(found.view().center, configured.view().center);
        same_bits(found.view().radii, configured.view().radii);
        for ((center, enclosure), expected) in found
            .view()
            .center
            .iter()
            .zip(found.enclosure().unwrap())
            .zip(expected)
        {
            assert!((*center - expected).abs() <= r::<S::Real>(8.) * S::Real::EPSILON);
            assert!(enclosure.re.contains(expected.real_component()));
            assert!(enclosure.im.contains(expected.imaginary_component()));
        }
        let (report, proposal) = verifier.propose(&boxes, found.view().center).unwrap();
        assert!(report.is_verified());
        let mut out = LinearCertificate::new(2).unwrap();
        let into = verifier
            .propose_into(
                &boxes,
                found.view().center,
                out.view_mut(),
                &mut verifier.workspace(1).unwrap(),
                seq(),
            )
            .unwrap();
        assert_eq!(report, into);
        let proposal = proposal.unwrap();
        same_bits(proposal.view().center, out.view().center);
        same_bits(proposal.view().radii, out.view().radii);
    }
    let a = Matrix::from_vec(2, 3, vec![S::ONE; 6]).unwrap();
    assert_eq!(
        solve(&a, &[S::ONE; 2]).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let a = Matrix::from_vec(2, 2, vec![S::ONE; 4]).unwrap();
    assert_eq!(
        solve(&a, &[S::ONE]).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        solve(&a, &[S::ONE; 2]).unwrap_err().kind(),
        SciOxideErrorKind::Singular
    );
    let a = Matrix::from_vec(1, 1, vec![S::from_components(S::Real::NAN, S::Real::ZERO)]).unwrap();
    assert_eq!(
        solve(&a, &[S::ONE]).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let a = Matrix::from_vec(1, 1, vec![S::ONE]).unwrap();
    assert_eq!(
        solve(&a, &[S::from_components(S::Real::NAN, S::Real::ZERO)])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
}

fn scalar<R: RealScalar>() {
    let one = D::point(R::ONE).unwrap();
    let two = D::point(R::TWO).unwrap();
    let f = |x: D<R>| x.square()?.sub(&one);
    let df = |x: D<R>| x.mul(&two);
    for (lower, upper, expected) in [
        (-2., 3., [(-2., -0.75), (0.5, 3.)]),
        (-2., 2., [(-1., -1.), (1., 1.)]),
    ] {
        let found = roots(f, df, r(lower), r(upper)).unwrap();
        let verifier = ScalarRootVerifier::new(&f, &df, ValidatedOptions::new());
        let mut tree = RootIsolation::new(1, 1023).unwrap();
        let mut workspace = RootWorkspace::new();
        let report = verifier
            .isolate_into(
                Interval::new(r(lower), r(upper)).unwrap().into(),
                StepTolerance::default(),
                &mut tree,
                &mut workspace,
                seq(),
            )
            .unwrap();
        assert!(report.is_complete());
        let mut configured: Vec<_> = tree
            .isolated_regions()
            .map(|x| Interval::try_from(x[0].re).unwrap())
            .collect();
        configured.sort_unstable_by(|a, b| a.lower().partial_cmp(&b.lower()).unwrap());
        let mut merged: Vec<Interval<R>> = Vec::new();
        for interval in configured {
            if let Some(last) = merged.last_mut()
                && last.upper() == interval.lower()
            {
                *last = Interval::new(interval.lower(), interval.lower()).unwrap();
            } else {
                merged.push(interval);
            }
        }
        assert_eq!(found, merged);
        assert_eq!(
            found,
            expected.map(|(a, b)| Interval::new(r(a), r(b)).unwrap())
        );
    }
    let f = |x: D<R>| x.square()?.sub(&two);
    assert_eq!(
        roots(f, df, r(-2.), r(2.)).unwrap(),
        [
            Interval::new(r(-2.), r(-1.)).unwrap(),
            Interval::new(r(1.), r(2.)).unwrap()
        ]
    );
    assert_eq!(
        roots(|x: D<R>| Ok(x), |_| Ok(one), r(-1.), r(1.)).unwrap(),
        [Interval::new(r(-1.), r(1.)).unwrap()]
    );
    assert!(
        roots(|x| x.square()?.add(&one), df, r(-2.), r(2.))
            .unwrap()
            .is_empty()
    );
    convergence(
        roots(|x: D<R>| x.square(), df, r(-1.), r(1.)),
        ConvergenceReason::StepSizeLimit,
    );
    assert_eq!(
        roots(f, df, r(2.), r(-2.)).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        roots(f, df, R::NAN, R::ONE).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let error = roots(
        |_: D<R>| Err(SciOxideError::singular(Some(7))),
        df,
        R::ZERO,
        R::ONE,
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::Singular { pivot: Some(7) }
    ));
    let found = roots(
        |x: D<R>| Ok(x.sin()?.hull()),
        |x| Ok(x.cos()?.hull()),
        r(-10.),
        r(10.),
    )
    .unwrap();
    assert_eq!(found.len(), 7);
    for (interval, k) in found.iter().zip(-3..=3) {
        assert!(interval.contains(r::<R>(f64::from(k)) * R::PI));
    }
    assert_eq!(found[3], Interval::new(R::ZERO, R::ZERO).unwrap());
    let three = D::point(r::<R>(3.)).unwrap();
    let twelve = D::point(r::<R>(12.)).unwrap();
    let eleven = D::point(r::<R>(11.)).unwrap();
    let found = roots(
        |x: D<R>| x.sub(&one)?.mul(&x.sub(&two)?)?.mul(&x.sub(&three)?),
        |x| x.square()?.mul(&three)?.sub(&x.mul(&twelve)?)?.add(&eleven),
        R::ZERO,
        r(4.),
    )
    .unwrap();
    assert_eq!(found.len(), 3);
    for (interval, k) in found.iter().zip(1..=3) {
        assert!(interval.contains(r(f64::from(k))));
    }
    assert_eq!(found[1], Interval::new(R::TWO, R::TWO).unwrap());
}

#[expect(
    clippy::approx_constant,
    reason = "rounded seeds exercise radius growth around an approximate root"
)]
fn polynomial<S: NumericScalar>() {
    let coefficients = [s::<S>(-2., 0.), S::ZERO, S::ONE];
    let approximations = [
        Complex::new(r(-1.4142), S::Real::ZERO),
        Complex::new(r(1.4142), S::Real::ZERO),
    ];
    let found = polynomial_roots(&coefficients, &approximations).unwrap();
    let plan = PolynomialRootPlan::new(&coefficients, ValidatedOptions::new()).unwrap();
    let seeds = approximations.map(|center| RootDisk {
        center,
        radius: S::Real::SQRT_EPSILON * center.re.abs().max(center.im.abs()).max(S::Real::ONE),
    });
    let (report, configured) = plan.isolate(&seeds).unwrap();
    assert!(report.is_complete());
    assert_eq!(found.view(), configured.view());
    for (disk, sign) in found.view().iter().zip([-1., 1.]) {
        assert_eq!(disk.count, 1);
        assert!(
            (disk.disk.center.re - r::<S::Real>(sign) * S::Real::TWO.sqrt()).abs()
                <= disk.disk.radius
        );
    }
    convergence(
        polynomial_roots(&coefficients, &approximations[..1]),
        ConvergenceReason::Stagnation,
    );
    assert_eq!(
        polynomial_roots(&[S::ZERO], &[]).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert!(
        polynomial_roots(&[s::<S>(3., 0.)], &[])
            .unwrap()
            .view()
            .is_empty()
    );
    assert_eq!(
        polynomial_roots(&coefficients, &[Complex::new(S::Real::NAN, S::Real::ZERO)])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    for (coefficients, approximations, counts) in [
        (vec![-1., 0., 1.], vec![(-1., 0.), (1., 0.)], vec![1, 1]),
        (
            vec![0., -1., 0., 1.],
            vec![(-1., 0.), (0., 0.), (1., 0.)],
            vec![1, 1, 1],
        ),
        (vec![1., -2., 1.], vec![(1., 0.), (1., 0.)], vec![2]),
        (
            vec![1., -2., 1.],
            vec![(1. - 1e-8, 0.), (1. + 1e-8, 0.)],
            vec![2],
        ),
        (vec![1., -2., 1.], vec![(1., 1e-8), (1., -1e-8)], vec![2]),
        (
            vec![-1., 3., -3., 1.],
            (0..3)
                .map(|j| {
                    let t = f64::from(j) * core::f64::consts::TAU / 3.;
                    (1. + 6e-6 * t.cos(), 6e-6 * t.sin())
                })
                .collect(),
            vec![3],
        ),
    ] {
        let coefficients: Vec<S> = coefficients.into_iter().map(|x| s(x, 0.)).collect();
        let approximations: Vec<_> = approximations
            .into_iter()
            .map(|(re, im)| Complex::new(r(re), r(im)))
            .collect();
        let cover = polynomial_roots(&coefficients, &approximations).unwrap();
        assert_eq!(
            cover.view().iter().map(|c| c.count).collect::<Vec<_>>(),
            counts
        );
        if counts.len() == 1 {
            assert!(
                (cover.view()[0].disk.center - Complex::new(S::Real::ONE, S::Real::ZERO)).abs()
                    <= cover.view()[0].disk.radius
            );
        }
    }
}

fn quadrature<R: RealScalar>() {
    let square = integrate(|x: D<R>| x.square(), |_| D::point(R::TWO), R::ZERO, R::ONE).unwrap();
    assert!(square.contains(R::ONE / r(3.)));
    let exp = |x: D<R>| Ok(x.exp()?.hull());
    let found = integrate(exp, exp, R::ZERO, R::ONE).unwrap();
    assert!(found.contains(R::ONE.exp() - R::ONE));
    let radius = (found.upper() - found.lower()) / R::TWO;
    assert!(
        radius <= R::SQRT_EPSILON * (R::ONE + ((found.upper() + found.lower()) / R::TWO).abs())
    );
    let pieces = [QuadraturePiece {
        domain: Interval::new(R::ZERO, R::ONE).unwrap().into(),
        assumption: QuadratureAssumption::TwiceContinuouslyDifferentiable,
    }];
    let oracle = |q: QuadratureRequest<R>, out: &mut [IntervalBox<R>]| {
        out[0] = Complex::new(exp(D::from(q.domain))?.set(), D::point(R::ZERO)?.set());
        Ok(QuadratureObligation::Established)
    };
    let verifier =
        QuadratureVerifier::<R>::new(&pieces, 1, &oracle, ValidatedOptions::new()).unwrap();
    let mut certificate = QuadratureCertificate::new(4096, 1).unwrap();
    let mut workspace = QuadratureWorkspace::new();
    assert_eq!(
        verifier
            .generate_into(
                &ErrorTolerance::default(),
                &mut certificate,
                &mut workspace,
                seq()
            )
            .unwrap()
            .termination(),
        QuadratureTermination::ToleranceReached
    );
    assert_eq!(
        verifier
            .verify_scoped(certificate.view(), &mut workspace, seq())
            .unwrap(),
        QuadratureVerificationStatus::Verified
    );
    assert_eq!(
        found,
        Interval::try_from(certificate.view().total[0].re).unwrap()
    );
    assert_eq!(
        integrate(exp, exp, R::ONE, R::ONE).unwrap(),
        Interval::new(R::ZERO, R::ZERO).unwrap()
    );
    assert_eq!(
        integrate(
            |x: D<R>| x.reciprocal(),
            |_| D::point(R::TWO),
            R::ZERO,
            R::ONE
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(
        integrate(exp, exp, R::ONE, R::ZERO).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn real32() {
    linear::<f32>();
    scalar::<f32>();
    polynomial::<f32>();
    quadrature::<f32>();
}
#[test]
fn real64() {
    linear::<f64>();
    scalar::<f64>();
    polynomial::<f64>();
    quadrature::<f64>();
}
#[test]
fn complex32() {
    linear::<Complex<f32>>();
    polynomial::<Complex<f32>>();
}
#[test]
fn complex64() {
    linear::<Complex<f64>>();
    polynomial::<Complex<f64>>();
}

#[test]
fn difficult_linear_systems_are_errors_and_propose_has_no_certificate() {
    for n in 12..=14 {
        let entries = (0..n)
            .flat_map(|i| (0..n).map(move |j| 1. / (i + j + 1) as f64))
            .collect();
        let a = Matrix::from_vec(n, n, entries).unwrap();
        let error = solve(&a, &vec![1.; n]).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::IllConditioned,
                ..
            }
        ));
        let factor = LuDecomposition::factor(&a).unwrap();
        let system = DenseIntervalOperator::point(a.view()).unwrap();
        let inverse = LuIntervalInverse::from_dense(&factor, ValidatedOptions::new()).unwrap();
        let verifier = LinearVerifier::new(&system, &inverse, ValidatedOptions::new()).unwrap();
        let (report, certificate) = verifier
            .propose(&vec![point_enclosure(1.).unwrap(); n], &vec![0.; n])
            .unwrap();
        assert!(!report.is_verified() && certificate.is_none());
    }
}
