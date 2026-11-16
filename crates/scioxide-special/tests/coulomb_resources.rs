//! Transactional layouts, stable order recurrences, allocation and work contracts.
#![allow(clippy::unwrap_used)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
#[path = "support/scaled.rs"]
mod scaled;
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, Complex, ExecutionContext, SciOxideErrorKind,
    SciOxideErrorView, SolveBudgets, SolveState,
};
use scioxide_special::*;

fn identity() -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(668),
        scioxide_core::algorithm_id!("coulomb", "test", 1),
        0,
        0,
    )
}

fn complex_sequence<R: scioxide_core::RealScalar>()
where
    Complex<R>: scioxide_core::NumericScalar<Real = R>,
{
    let mut checker = fixture::Checker::new(
        "coulomb_sequence",
        if R::MANTISSA_DIGITS == 24 { "32" } else { "64" },
    );
    let scalar = |x| R::try_from_f64_lossy(x).unwrap();
    let options = CoulombOptions::default();
    for (l, eta, re, im) in [
        (0.25, 1.0, 1.0, 0.5),
        (0.25, 1.0, -2.0, -0.1),
        (0.0, 40.0, 1.0, 1.0),
        (40.0, 0.0, 2.0, 1.0),
        (45.0, 100.0, 0.001, 0.001),
        (45.0, -100.0, -0.001, -0.001),
        (45.0, 100.0, 0.0, 5000.0),
        (45.0, -100.0, -4000.0, 3000.0),
        (0.25, 1.0, -5000.0, -0.0),
    ] {
        let z = Complex::new(scalar(re), scalar(im));
        let mut output = [CoulombWaveResult::<Complex<R>>::zero(); 6];
        let mut staging = output;
        coulomb_wave_sequence_into(
            scalar(l),
            scalar(eta),
            z,
            options,
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for (k, r) in output.iter().enumerate() {
            let reference = coulomb_wave_with_options(
                scalar(l + k as f64),
                scalar(eta),
                z,
                scioxide_special::CoulombOptions::default(),
            )
            .unwrap();
            for (a, b) in [
                (r.f(), reference.f()),
                (r.g(), reference.g()),
                (r.f_prime(), reference.f_prime()),
                (r.g_prime(), reference.g_prime()),
            ] {
                let ae = a.scaled().unwrap();
                let be = b.scaled().unwrap();
                let exponent = be.value().scale_exponent();
                let av = scaled::at_exponent(ae.value(), exponent);
                let bv = scaled::at_exponent(be.value(), exponent);
                let error = scaled::error_at_exponent(
                    ae.value(),
                    ae.absolute_error().try_to_f64_exact().unwrap(),
                    exponent,
                ) + scaled::error_at_exponent(
                    be.value(),
                    be.absolute_error().try_to_f64_exact().unwrap(),
                    exponent,
                );
                let difference = (av - bv).norm();
                assert!(
                    difference <= error,
                    "{l} {eta} {re} {im}, order {k}: {av:?}, {bv:?}, bound {error}"
                );
                checker.sample(
                    "component",
                    difference,
                    bv.norm(),
                    Some(error),
                    &format!("{l} {eta} {re} {im}, order {k}"),
                );
            }
        }
    }
}
#[test]
fn complex_sequences_in_both_precisions() {
    complex_sequence::<f32>();
    complex_sequence::<f64>();
}
#[test]
fn sequences_reuse_seeds_and_preserve_output_on_failure() {
    let options = CoulombOptions::default();
    let seed = CoulombWaveResult::<f64>::zero();
    for (l, eta, z) in [(0.0, 2.0, 1.0_f64), (0.25, -3.0, 10.0), (2.0, 20.0, 45.0)] {
        let mut output = [seed; 12];
        let mut scratch = output;
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        coulomb_wave_sequence_into(l, eta, z, options, &mut output, &mut scratch, &mut context)
            .unwrap();
        let mut scalar_work = 0;
        for (i, value) in output.iter().enumerate() {
            let reference = coulomb_wave_with_options(
                l + i as f64,
                eta,
                z,
                scioxide_special::CoulombOptions::default(),
            )
            .unwrap();
            scalar_work += reference.f().counts().iterations();
            for (a, b) in [
                (value.f(), reference.f()),
                (value.g(), reference.g()),
                (value.f_prime(), reference.f_prime()),
                (value.g_prime(), reference.g_prime()),
            ] {
                let x = a.try_value().unwrap();
                let y = b.try_value().unwrap();
                assert!(
                    (x - y).abs() <= 4096.0 * f64::EPSILON * y.abs().max(1e-10),
                    "{l} {eta} {z}, order {i}: {x} {y}"
                );
            }
        }
        assert_eq!(context.counts().own().iterations(), 0);
        assert!(context.counts().nested().iterations() < scalar_work);
        assert_eq!(output[0].f().counts(), context.counts().nested());
        let before = output;
        let error = coulomb_wave_sequence_into(
            l,
            eta,
            z,
            options.with_iteration_budget(Budget::ONE),
            &mut output,
            &mut scratch,
            &mut context,
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                diagnostics,
                ..
            } if diagnostics.counts().own().iterations() == 1
        ));
        for (a, b) in output.iter().zip(before) {
            assert_eq!(a.f(), b.f());
            assert_eq!(a.g(), b.g());
        }
    }
}
#[test]
fn regular_normalization_and_spherical_sequences() {
    let options = CoulombOptions::default();
    let seed = SpecialFunctionResult::<f64>::zero();
    let mut out = [seed; 8];
    let mut scratch = out;
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    for query in [
        CoulombComponent::F,
        CoulombComponent::G,
        CoulombComponent::SphericalF,
        CoulombComponent::FPrime,
        CoulombComponent::GPrime,
    ] {
        coulomb_sequence_into(
            query,
            0.25,
            1.0,
            2.0,
            options,
            &mut out,
            &mut scratch,
            &mut context,
        )
        .unwrap();
        for (k, r) in out.iter().enumerate() {
            let reference = match query {
                CoulombComponent::F => coulomb_f(0.25 + k as f64, 1.0, 2.0),
                CoulombComponent::G => coulomb_g(0.25 + k as f64, 1.0, 2.0),
                CoulombComponent::SphericalF => coulomb_spherical_f(0.25 + k as f64, 1.0, 2.0),
                CoulombComponent::FPrime => coulomb_f_prime(0.25 + k as f64, 1.0, 2.0),
                CoulombComponent::GPrime => coulomb_g_prime(0.25 + k as f64, 1.0, 2.0),
            }
            .unwrap();
            assert!(
                (r.value().unwrap().value() - reference).abs()
                    < 4096.0 * f64::EPSILON * reference.abs()
            );
        }
    }
    coulomb_sequence_into(
        CoulombComponent::F,
        0.0,
        1.0,
        0.0,
        options,
        &mut out,
        &mut scratch,
        &mut context,
    )
    .unwrap();
    assert!(out.iter().all(|r| *r.value().unwrap().value() == 0.0));
    coulomb_sequence_into(
        CoulombComponent::F,
        0.0,
        1.0,
        1.0,
        options,
        &mut [],
        &mut [],
        &mut context,
    )
    .unwrap();
    let before = out;
    assert_eq!(
        coulomb_sequence_into(
            CoulombComponent::F,
            0.0,
            1.0,
            1.0,
            options,
            &mut out,
            &mut [],
            &mut context
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, before);
    assert!(
        coulomb_sequence_into(
            CoulombComponent::F,
            1e20,
            1.0,
            1.0,
            options,
            &mut out,
            &mut scratch,
            &mut context
        )
        .is_err()
    );
}
#[test]
fn scoped_budget_cancellation_and_strict_tolerance() {
    let options = CoulombOptions::default();
    let nine = Budget::try_from(9).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, nine);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    // The options budget bounds the child epoch; the parent limit is scope-local.
    let report = coulomb_wave_scoped(
        1.0,
        1.0,
        1.0_f64,
        options.with_iteration_budget(nine),
        &mut context,
    )
    .unwrap_err();
    assert_eq!(publication::exhausted_counts(&report).iterations(), 9);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 9);
    assert!(
        coulomb_wave_scoped(1.0, 1.0, 1.0_f64, options, &mut context)
            .unwrap()
            .is_success()
    );
    let root =
        coulomb_wave_scoped(1.0, 1.0, 1.0_f64, options, ExecutionContext::sequential()).unwrap();
    let reference = coulomb_wave_with_options(1.0, 1.0, 1.0_f64, options).unwrap();
    assert_eq!(
        [root.f(), root.g(), root.f_prime(), root.g_prime()],
        [
            reference.f(),
            reference.g(),
            reference.f_prime(),
            reference.g_prime()
        ]
    );
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity(),
            &mut budgets,
        )
        .unwrap();
    assert!(matches!(
        coulomb_f_scoped(0.0, 0.0, 0.0, options, &mut context)
            .unwrap_err()
            .view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    let seed = CoulombWaveResult::<f64>::zero();
    let mut output = [seed; 2];
    let mut staging = output;
    assert!(matches!(
        coulomb_wave_sequence_into(
            0.0,
            0.0,
            1.0,
            options,
            &mut output,
            &mut staging,
            &mut context
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert!(
        coulomb_f_with_options(1.0, 1.0, 2.0, options)
            .unwrap()
            .termination()
            .is_success()
    );
}

#[test]
fn turning_paths_charge_budget_and_execute() {
    fn check<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>>() {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        // Continued below the turning point, then evaluated directly beyond it.
        for (l, eta, x) in [(0.74, 0.63, 1.26), (0.5, -0.5, 6.0), (2.9, 1.0, 7.5)] {
            let (l, eta, x) = (scalar(l), scalar(eta), scalar(x));
            let options = CoulombOptions::default();
            let complete = coulomb_wave_with_options(l, eta, x, options).unwrap();
            let reports = [
                complete.f(),
                complete.g(),
                complete.f_prime(),
                complete.g_prime(),
            ];
            let count = complete.f().counts().iterations();
            assert!(count > 2);
            assert!(reports.iter().all(|r| r.counts().iterations() == count));
            let plain = coulomb_wave(l, eta, x).unwrap();
            assert_eq!(plain, reports.map(|r| r.try_value().unwrap()));
            for limit in [1, count - 1, count] {
                let limited = options.with_iteration_budget(Budget::try_from(limit).unwrap());
                let result = coulomb_wave_with_options(l, eta, x, limited);
                if limit == count {
                    let result = result.unwrap();
                    assert_eq!(
                        [result.f(), result.g(), result.f_prime(), result.g_prime()],
                        reports
                    );
                } else {
                    assert_eq!(
                        publication::exhausted_counts(&result.unwrap_err()).iterations(),
                        limit
                    );
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn invalid_empty_sequences_are_rejected_before_execution() {
    let options = CoulombOptions::default();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    for (l, eta, z) in [
        (f64::NAN, 0.0, 1.0),
        (0.0, f64::INFINITY, 1.0),
        (0.0, 0.0, f64::NAN),
    ] {
        for component in [
            CoulombComponent::F,
            CoulombComponent::G,
            CoulombComponent::FPrime,
            CoulombComponent::GPrime,
            CoulombComponent::SphericalF,
        ] {
            let error = coulomb_sequence_into(
                component,
                l,
                eta,
                z,
                options,
                &mut [],
                &mut [],
                ExecutionContext::sequential().with_cancellation(&cancelled),
            )
            .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
        }
        let error = coulomb_wave_sequence_into(
            l,
            eta,
            z,
            options,
            &mut [],
            &mut [],
            ExecutionContext::sequential().with_cancellation(&cancelled),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    }
    for (l, eta) in [(f64::NAN, 0.0), (0.0, f64::NAN)] {
        let error = coulomb_normalization_sequence_into(
            l,
            eta,
            options,
            &mut [],
            &mut [],
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    }
    for z in [Complex::new(f64::NAN, 0.0), Complex::new(1.0, f64::NAN)] {
        assert_eq!(
            coulomb_wave_sequence_into(
                0.0,
                0.0,
                z,
                options,
                &mut [],
                &mut [],
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
}

#[test]
fn sequence_origin_domains_match_scalar_domains() {
    let options = CoulombOptions::default();
    for component in [
        CoulombComponent::F,
        CoulombComponent::G,
        CoulombComponent::FPrime,
        CoulombComponent::GPrime,
        CoulombComponent::SphericalF,
    ] {
        for l in [-0.25, 0.0, 0.25, 1.0] {
            for eta in [0.0, 1.0] {
                let mut output = [SpecialFunctionResult::<f64>::zero(); 1];
                let mut staging = output;
                let reference = match component {
                    CoulombComponent::F => coulomb_f(l, eta, 0.0_f64),
                    CoulombComponent::G => coulomb_g(l, eta, 0.0_f64),
                    CoulombComponent::FPrime => coulomb_f_prime(l, eta, 0.0_f64),
                    CoulombComponent::GPrime => coulomb_g_prime(l, eta, 0.0_f64),
                    CoulombComponent::SphericalF => coulomb_spherical_f(l, eta, 0.0_f64),
                };
                let result = coulomb_sequence_into(
                    component,
                    l,
                    eta,
                    0.0,
                    options,
                    &mut output,
                    &mut staging,
                    ExecutionContext::sequential(),
                );
                match reference {
                    Ok(value) => {
                        result.unwrap();
                        assert!(
                            (output[0].try_value().unwrap() - value).abs()
                                <= 4.0 * f64::EPSILON * value.abs()
                        );
                    }
                    Err(error) => {
                        assert_eq!(result.unwrap_err().kind(), error.kind());
                        assert_eq!(output[0], SpecialFunctionResult::zero());
                    }
                }
            }
        }
    }
    let mut output = [CoulombWaveResult::<f64>::zero(); 2];
    let mut staging = output;
    coulomb_wave_sequence_into(
        0.0,
        0.0,
        0.0,
        options,
        &mut output[..1],
        &mut staging,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output[0].g().try_value().unwrap(), 1.0);
    let before = output;
    assert!(
        coulomb_wave_sequence_into(
            0.0,
            0.0,
            0.0,
            options,
            &mut output,
            &mut staging,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output[0].g(), before[0].g());
    assert_eq!(output[1].g(), before[1].g());
    assert!(
        coulomb_wave_sequence_into(
            0.0,
            1.0,
            0.0_f64,
            options,
            &mut [],
            &mut [],
            ExecutionContext::sequential()
        )
        .is_err()
    );
}

#[test]
fn real_normalization_sequences_share_counts_and_commit_atomically() {
    let options = CoulombOptions::default();
    let mut output = [SpecialFunctionResult::<f64>::zero(); 8];
    let mut staging = output;
    for eta in [-1e4, -1000.0, -1.0, 0.0, 1.0, 1000.0, 1e4] {
        coulomb_normalization_sequence_into(
            0.0,
            eta,
            options,
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for (k, result) in output.iter().enumerate() {
            let reference = coulomb_normalization_with_options(k as f64, eta, options).unwrap();
            let actual = result.logarithmic().unwrap();
            let expected = reference.logarithmic().unwrap();
            assert!(
                (actual.value().log_magnitude().unwrap()
                    - expected.value().log_magnitude().unwrap())
                .abs()
                    <= actual.absolute_error() + expected.absolute_error()
            );
            assert_eq!(result.counts(), output[0].counts());
        }
        let before = output;
        let error = coulomb_normalization_sequence_into(
            0.0,
            eta,
            options.with_iteration_budget(Budget::ONE),
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(publication::exhausted_counts(&error).iterations(), 1);
        assert_eq!(output, before);
    }
}

#[test]
fn large_complex_radii_have_bounded_work_and_execute() {
    fn check<R: scioxide_core::RealScalar>()
    where
        Complex<R>: scioxide_core::NumericScalar<Real = R>,
    {
        let r = |x| R::try_from_f64_lossy(x).unwrap();
        for (l, eta) in [(0.0, 0.0), (0.25, 1.0), (5.0, -2.0)] {
            let mut counts = Vec::new();
            for radius in [5000.0, 10000.0, 1000000.0] {
                let z = Complex::new(r(radius), r(1.0));
                let report =
                    coulomb_wave_with_options(r(l), r(eta), z, CoulombOptions::default()).unwrap();
                assert!(report.is_success());
                let count = report.f().counts().iterations();
                assert!(count < 2000, "{l} {eta} {radius}: {count}");
                counts.push(count);
            }
            assert!(counts[2] <= counts[0]);
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn grouped_complex_sequences_reuse_the_computed_regular_wave() {
    let options = CoulombOptions::default();
    for (l, eta, z) in [
        (0.25, 1.0, Complex::new(1.0_f64, 1.0)),
        (40.0, 0.0, Complex::new(2.0, 1.0)),
        (0.0, 0.0, Complex::new(5000.0, 1.0)),
    ] {
        let scalar = coulomb_wave_with_options(l, eta, z, options).unwrap();
        let mut output = [CoulombWaveResult::zero(); 1];
        let mut staging = output;
        coulomb_wave_sequence_into(
            l,
            eta,
            z,
            options,
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output[0].f().counts(), scalar.f().counts());
        for component in [
            CoulombComponent::F,
            CoulombComponent::G,
            CoulombComponent::FPrime,
            CoulombComponent::GPrime,
            CoulombComponent::SphericalF,
        ] {
            let mut out = [SpecialFunctionResult::zero(); 4];
            let mut scratch = out;
            coulomb_sequence_into(
                component,
                l,
                eta,
                z,
                options,
                &mut out,
                &mut scratch,
                ExecutionContext::sequential(),
            )
            .unwrap();
            for (k, actual) in out.into_iter().enumerate() {
                let expected = match component {
                    CoulombComponent::F => coulomb_f(l + k as f64, eta, z),
                    CoulombComponent::G => coulomb_g(l + k as f64, eta, z),
                    CoulombComponent::FPrime => coulomb_f_prime(l + k as f64, eta, z),
                    CoulombComponent::GPrime => coulomb_g_prime(l + k as f64, eta, z),
                    CoulombComponent::SphericalF => coulomb_spherical_f(l + k as f64, eta, z),
                }
                .unwrap();
                assert!(
                    (actual.try_value().unwrap() - expected).norm()
                        <= 4096.0 * f64::EPSILON * expected.norm()
                );
            }
        }
    }
}

const COMPONENTS: [CoulombComponent; 5] = [
    CoulombComponent::F,
    CoulombComponent::G,
    CoulombComponent::FPrime,
    CoulombComponent::GPrime,
    CoulombComponent::SphericalF,
];

/// Checks that the owned sequences return the ordinary values of the `_into` members.
fn owned_sequences_project_into_members<T: scioxide_core::NumericScalar>(
    l: T::Real,
    eta: T::Real,
    z: T,
) {
    const COUNT: usize = 6;
    let options = CoulombOptions::default();
    let scope = ExecutionContext::sequential_native;
    let mut output = [CoulombWaveResult::<T>::zero(); COUNT];
    let mut staging = output;
    coulomb_wave_sequence_into(l, eta, z, options, &mut output, &mut staging, scope()).unwrap();
    let expected = output.map(|wave| {
        [wave.f(), wave.g(), wave.f_prime(), wave.g_prime()].map(|r| r.try_value().unwrap())
    });
    assert_eq!(coulomb_wave_sequence(l, eta, z, COUNT).unwrap(), expected);
    for component in COMPONENTS {
        let mut output = [SpecialFunctionResult::<T>::zero(); COUNT];
        let mut staging = output;
        coulomb_sequence_into(
            component,
            l,
            eta,
            z,
            options,
            &mut output,
            &mut staging,
            scope(),
        )
        .unwrap();
        assert_eq!(
            coulomb_sequence(component, l, eta, z, COUNT).unwrap(),
            output.map(|member| member.try_value().unwrap()),
            "{component:?}"
        );
    }
}

fn owned_sequences_in<R: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = R>>()
where
    Complex<R>: scioxide_core::NumericScalar<Real = R>,
{
    let r = |x| R::try_from_f64_lossy(x).unwrap();
    for (l, eta, x) in [(0.0, 0.0, 1.0), (0.25, 1.0, 2.0), (2.0, -3.0, 10.0)] {
        owned_sequences_project_into_members::<R>(r(l), r(eta), r(x));
        owned_sequences_project_into_members::<Complex<R>>(
            r(l),
            r(eta),
            Complex::new(r(x), r(0.5)),
        );
        let mut output = [SpecialFunctionResult::<R>::zero(); 5];
        let mut staging = output;
        coulomb_normalization_sequence_into(
            r(l),
            r(eta),
            CoulombOptions::default(),
            &mut output,
            &mut staging,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(
            coulomb_normalization_sequence(r(l), r(eta), 5).unwrap(),
            output.map(|member| member.try_value().unwrap())
        );
    }
}

#[test]
fn owned_sequences_equal_into_values_in_both_precisions() {
    owned_sequences_in::<f32>();
    owned_sequences_in::<f64>();
}

#[test]
fn empty_owned_sequences_still_check_parameters() {
    assert!(
        coulomb_wave_sequence(0.0, 0.0, 1.0_f64, 0)
            .unwrap()
            .is_empty()
    );
    assert!(
        coulomb_wave_sequence(0.0, 0.0, Complex::new(1.0_f32, 0.5), 0)
            .unwrap()
            .is_empty()
    );
    assert!(
        coulomb_normalization_sequence(0.0_f64, 0.0, 0)
            .unwrap()
            .is_empty()
    );
    for component in COMPONENTS {
        assert!(
            coulomb_sequence(component, 0.0, 0.0, 1.0_f64, 0)
                .unwrap()
                .is_empty()
        );
        for (l, eta, z) in [
            (f64::NAN, 0.0, 1.0),
            (0.0, f64::INFINITY, 1.0),
            (0.0, 0.0, f64::NAN),
        ] {
            assert_eq!(
                coulomb_sequence(component, l, eta, z, 0)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::NonFinite
            );
        }
    }
    for (l, eta, z) in [
        (f64::NAN, 0.0, Complex::new(1.0, 0.0)),
        (0.0, f64::NAN, Complex::new(1.0, 0.0)),
        (0.0, 0.0, Complex::new(1.0, f64::NAN)),
    ] {
        assert_eq!(
            coulomb_wave_sequence(l, eta, z, 0).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    for (l, eta) in [(f64::NAN, 0.0), (0.0, f64::NAN)] {
        assert_eq!(
            coulomb_normalization_sequence(l, eta, 0)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    // The domain is checked too: `z = 0` admits only `l = η = 0`, and `l ≤ −1/2` is outside.
    for (l, eta, z) in [(0.0, 1.0, 0.0_f64), (-0.5, 0.0, 1.0)] {
        assert_eq!(
            coulomb_wave_sequence(l, eta, z, 0).unwrap_err().kind(),
            SciOxideErrorKind::DomainError
        );
    }
}

#[test]
fn owned_sequences_reject_members_that_into_keeps_scaled() {
    use scioxide_core::NumericalIssue;
    // G₀(300, 1) ≈ e^(300π) exceeds f64::MAX; its scaled form is still finite.
    let (l, eta, x) = (0.0, 300.0, 1.0_f64);
    let scope = ExecutionContext::sequential_native;
    let mut output = [CoulombWaveResult::<f64>::zero(); 3];
    let mut staging = output;
    coulomb_wave_sequence_into(
        l,
        eta,
        x,
        CoulombOptions::default(),
        &mut output,
        &mut staging,
        scope(),
    )
    .unwrap();
    for wave in output {
        assert!(wave.is_success());
        assert_eq!(
            wave.g().scalar_representation(),
            ScalarRepresentation::Overflow
        );
        assert!(wave.g().scaled().is_some());
    }
    let mut output = [SpecialFunctionResult::<f64>::zero(); 3];
    let mut staging = output;
    coulomb_sequence_into(
        CoulombComponent::G,
        l,
        eta,
        x,
        CoulombOptions::default(),
        &mut output,
        &mut staging,
        scope(),
    )
    .unwrap();
    assert!(
        output
            .iter()
            .all(|r| r.scalar_representation() == ScalarRepresentation::Overflow)
    );
    for error in [
        coulomb_wave_sequence(l, eta, x, 3).map(drop).unwrap_err(),
        coulomb_sequence(CoulombComponent::G, l, eta, x, 3)
            .map(drop)
            .unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }
    // F₀(300, 1) is a rounded zero, an ordinary value.
    assert!(
        coulomb_sequence(CoulombComponent::F, l, eta, x, 3)
            .unwrap()
            .iter()
            .all(|value| value.abs() < f64::MIN_POSITIVE)
    );
}
