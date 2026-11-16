//! Exact scalar and manufactured descriptor control equations, workspaces and resources.
#![allow(clippy::unwrap_used)]

use scioxide_core::{AlgebraicScalar, NumericScalar};
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Complex, ExecutionContext, ProblemToken, RealScalar,
    SciOxideError, SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::{
    DirectSolveTermination, LinalgWorkspace, Matrix, RiccatiBalancing, RiccatiKind, RiccatiOptions,
    RiccatiPlan, RiccatiProblem,
};
fn run<S: NumericScalar>() {
    let v = |x| S::from_real(S::Real::try_from_f64_lossy(x).unwrap());
    for verify in [
        scioxide_linalg::Estimation::Skip,
        scioxide_linalg::Estimation::Estimate,
    ] {
        for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
            for descriptor in [false, true] {
                for balancing in [RiccatiBalancing::None, RiccatiBalancing::Scale] {
                    let (aa, bb, ee, ss, x, rr) = if descriptor {
                        (0.5, 0.75, 2., 0.25, 1.5, 2.)
                    } else {
                        (0., 1., 1., 0., 1., 1.)
                    };
                    let qq = if kind == RiccatiKind::Continuous {
                        (ee * x * bb + ss) * (bb * x * ee + ss) / rr - 2. * ee * x * aa
                    } else {
                        ee * x * ee - aa * x * aa
                            + (aa * x * bb + ss) * (bb * x * aa + ss) / (rr + bb * x * bb)
                    };
                    let a = Matrix::from_vec(1, 1, vec![v(aa)]).unwrap();
                    let b = Matrix::from_vec(1, 1, vec![v(bb)]).unwrap();
                    let q = Matrix::from_vec(1, 1, vec![v(qq)]).unwrap();
                    let r = Matrix::from_vec(1, 1, vec![v(rr)]).unwrap();
                    let e = Matrix::from_vec(1, 1, vec![v(ee)]).unwrap();
                    let s = Matrix::from_vec(1, 1, vec![v(ss)]).unwrap();
                    let p = RiccatiPlan::new(
                        RiccatiProblem {
                            a: a.view(),
                            b: b.view(),
                            q: q.view(),
                            r: r.view(),
                            e: descriptor.then_some(e.view()),
                            s: descriptor.then_some(s.view()),
                        },
                        kind,
                        RiccatiOptions::new()
                            .with_balancing(balancing)
                            .with_closed_loop_verification(verify),
                    )
                    .unwrap();
                    let mut w = p.workspace(()).unwrap();
                    let mut out = Matrix::<S>::zeros(1, 1).unwrap();
                    let mut gain = Matrix::<S>::zeros(1, 1).unwrap();
                    let report = p
                        .solve_into(
                            out.view_mut(),
                            gain.view_mut(),
                            &mut w,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                    assert_eq!(
                        report.termination(),
                        DirectSolveTermination::Accepted,
                        "{report:?}"
                    );
                    assert!(report.work() > 0);
                    assert_eq!(
                        report.work(),
                        w.last_attempt()
                            .unwrap()
                            .counts()
                            .total()
                            .unwrap()
                            .iterations()
                    );
                    if !descriptor {
                        assert_eq!(p.state_scaling(), &[S::Real::ONE]);
                    }
                    assert_eq!(
                        report.stability_margin().is_some(),
                        verify == scioxide_linalg::Estimation::Estimate
                    );
                    let tol = S::Real::try_from_usize_exact(2048).unwrap()
                        * <S::Real as RealScalar>::EPSILON;
                    assert!(
                        (out.view().get(0, 0).unwrap() - v(x)).abs() < tol,
                        "{kind:?} {descriptor} {balancing:?}: {:?}",
                        out.as_slice()
                    );
                }
            }
        }
    }
}
#[test]
fn real32() {
    run::<f32>()
}
#[test]
fn real64() {
    run::<f64>()
}
#[test]
fn complex32() {
    run::<Complex<f32>>()
}
#[test]
fn complex64() {
    run::<Complex<f64>>()
}

fn hermitian_errors<S: NumericScalar>() {
    let one = Matrix::<S>::identity(2).unwrap();
    let mut invalid = one.clone();
    invalid.as_mut_slice()[1] = S::ONE;
    for diagonal in [false, true] {
        if diagonal {
            if S::IS_REAL {
                continue;
            }
            invalid = one.clone();
            invalid.as_mut_slice()[0] = S::from_components(S::Real::ONE, S::Real::ONE);
        }
        for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
            for bad_q in [true, false] {
                let error = RiccatiPlan::new(
                    RiccatiProblem {
                        a: one.view(),
                        b: one.view(),
                        q: if bad_q { invalid.view() } else { one.view() },
                        r: if bad_q { one.view() } else { invalid.view() },
                        e: None,
                        s: None,
                    },
                    kind,
                    RiccatiOptions::new(),
                )
                .unwrap_err();
                assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
                let expected = if bad_q {
                    scioxide_core::argument!("q")
                } else {
                    scioxide_core::argument!("r")
                };
                assert!(
                    matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, .. } if argument == expected)
                );
            }
        }
    }
}

#[test]
fn non_hermitian_costs_identify_the_argument() {
    hermitian_errors::<f32>();
    hermitian_errors::<f64>();
    hermitian_errors::<Complex<f32>>();
    hermitian_errors::<Complex<f64>>();
}

#[test]
fn absent_descriptor_matches_identity_gemm() {
    let a = Matrix::from_vec(2, 2, vec![0.25_f64, 0.5, 0., -0.75]).unwrap();
    let b = Matrix::from_vec(2, 1, vec![0.5, 2.]).unwrap();
    let q = Matrix::from_vec(2, 2, vec![4., 0.25, 0.25, 0.5]).unwrap();
    let r = Matrix::identity(1).unwrap();
    let e = Matrix::identity(2).unwrap();
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        for balancing in [RiccatiBalancing::None, RiccatiBalancing::Scale] {
            let solve = |descriptor| {
                RiccatiPlan::new(
                    RiccatiProblem {
                        a: a.view(),
                        b: b.view(),
                        q: q.view(),
                        r: r.view(),
                        e: descriptor,
                        s: None,
                    },
                    kind,
                    RiccatiOptions::new().with_balancing(balancing),
                )
                .unwrap()
                .solve()
                .unwrap()
            };
            let absent = solve(None);
            let explicit = solve(Some(e.view()));
            assert_eq!(
                absent.report().termination(),
                DirectSolveTermination::Accepted
            );
            assert_eq!(
                explicit.report().termination(),
                DirectSolveTermination::Accepted
            );
            for (left, right) in absent
                .solution()
                .unwrap()
                .as_slice()
                .iter()
                .zip(explicit.solution().unwrap().as_slice())
                .chain(
                    absent
                        .gain()
                        .unwrap()
                        .as_slice()
                        .iter()
                        .zip(explicit.gain().unwrap().as_slice()),
                )
            {
                assert!((left - right).abs() <= 64. * f64::EPSILON * left.abs().max(1.));
            }
        }
    }
}

// Independent cancellation-resistant binary64 formulas for binary32 inputs.
fn scalar_reference(a: f64, b: f64, q: f64, r: f64, kind: RiccatiKind) -> (f64, f64) {
    let g = b * b / r;
    if kind == RiccatiKind::Continuous {
        let root = a.hypot((g * q).sqrt());
        let x = if a < 0. {
            q / (root - a)
        } else {
            (a + root) / g
        };
        (x, root / root.max(1.))
    } else {
        let linear = 1. - a * a - g * q;
        let root = linear.hypot(2. * (g * q).sqrt());
        let x = if linear >= 0. {
            2. * q / (linear + root)
        } else {
            (root - linear) / (2. * g)
        };
        (x, 1. - a.abs() / (1. + g * x))
    }
}

#[test]
fn binary32_scaled_scalars_respect_stability_exclusion() {
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        for aa in [-1e6_f32, -1., -1e-6, 0., 1e-6, 1., 1e6] {
            for bb in [1e-6_f32, 1., 1e6] {
                for qq in [1e-6_f32, 1., 1e6] {
                    for rr in [1e-6_f32, 1., 1e6] {
                        let (expected, margin) =
                            scalar_reference(aa.into(), bb.into(), qq.into(), rr.into(), kind);
                        let a = Matrix::from_vec(1, 1, vec![aa]).unwrap();
                        let b = Matrix::from_vec(1, 1, vec![bb]).unwrap();
                        let q = Matrix::from_vec(1, 1, vec![qq]).unwrap();
                        let r = Matrix::from_vec(1, 1, vec![rr]).unwrap();
                        let plan = RiccatiPlan::new(
                            RiccatiProblem {
                                a: a.view(),
                                b: b.view(),
                                q: q.view(),
                                r: r.view(),
                                e: None,
                                s: None,
                            },
                            kind,
                            RiccatiOptions::new(),
                        );
                        if margin <= f64::from(f32::EPSILON) {
                            assert_eq!(
                                plan.unwrap_err().kind(),
                                SciOxideErrorKind::NumericalInstability
                            );
                        } else {
                            let result = plan.unwrap().solve().unwrap();
                            assert_eq!(
                                result.report().termination(),
                                DirectSolveTermination::Accepted
                            );
                            assert!(
                                (f64::from(result.solution().unwrap().as_slice()[0]) / expected
                                    - 1.)
                                    .abs()
                                    <= 500. * f64::from(f32::EPSILON),
                                "{kind:?} {aa} {bb} {qq} {rr}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn reported_scalar_care_is_accurate_and_allocation_free() {
    let a = Matrix::from_vec(1, 1, vec![1e3_f64]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1e-3]).unwrap();
    let q = Matrix::from_vec(1, 1, vec![1e-6]).unwrap();
    let r = Matrix::identity(1).unwrap();
    let plan = RiccatiPlan::new(
        RiccatiProblem {
            a: a.view(),
            b: b.view(),
            q: q.view(),
            r: r.view(),
            e: None,
            s: None,
        },
        RiccatiKind::Continuous,
        RiccatiOptions::new(),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut x = Matrix::zeros(1, 1).unwrap();
    let mut gain = Matrix::zeros(1, 1).unwrap();
    let report = plan
        .solve_into(
            x.view_mut(),
            gain.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    assert!((x.as_slice()[0] / 2e9 - 1.).abs() <= 500. * f64::EPSILON);
}

#[test]
fn extreme_binary32_scalar_equations_bypass_pencil_balancing() -> Result<(), SciOxideError> {
    let b = Matrix::from_vec(1, 1, vec![1e-15])?;
    let q = Matrix::from_vec(1, 1, vec![1e30])?;
    let r = Matrix::identity(1)?;
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        for aa in [-1_f32, 0., 1.] {
            let a = Matrix::from_vec(1, 1, vec![aa])?;
            let (expected, _) = scalar_reference(
                f64::from(aa),
                f64::from(b.as_slice()[0]),
                f64::from(q.as_slice()[0]),
                1.,
                kind,
            );
            for balancing in [RiccatiBalancing::None, RiccatiBalancing::Scale] {
                let plan = RiccatiPlan::new(
                    RiccatiProblem {
                        a: a.view(),
                        b: b.view(),
                        q: q.view(),
                        r: r.view(),
                        e: None,
                        s: None,
                    },
                    kind,
                    RiccatiOptions::new().with_balancing(balancing),
                )?;
                assert_eq!(plan.state_scaling(), &[1.]);
                let mut workspace = plan.workspace(())?;
                let mut x = Matrix::zeros(1, 1)?;
                let mut gain = Matrix::zeros(1, 1)?;
                let report = plan.solve_into(
                    x.view_mut(),
                    gain.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )?;
                assert_eq!(report.termination(), DirectSolveTermination::Accepted);
                assert!(
                    (f64::from(x.as_slice()[0]) / expected - 1.).abs()
                        <= 64. * f64::from(f32::EPSILON)
                );
                let expected_gain = if kind == RiccatiKind::Continuous {
                    f64::from(b.as_slice()[0]) * expected
                } else {
                    f64::from(aa) * f64::from(b.as_slice()[0]) * expected
                        / (1. + f64::from(b.as_slice()[0]).powi(2) * expected)
                };
                assert!(
                    (f64::from(gain.as_slice()[0]) - expected_gain).abs()
                        <= 64. * f64::from(f32::EPSILON) * expected_gain.abs().max(1.)
                );
                assert!(report.backward_error() <= 64. * f32::EPSILON);
                assert!(report.work() > 0);
                assert_eq!(
                    report.work(),
                    workspace
                        .last_attempt()
                        .unwrap()
                        .counts()
                        .total()?
                        .iterations()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn extreme_binary64_scalar_care_keeps_coefficients_extended() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(1, 1, vec![0_f64])?;
    let b = Matrix::from_vec(1, 1, vec![1e-150])?;
    let q = Matrix::from_vec(1, 1, vec![1e300])?;
    let r = Matrix::identity(1)?;
    let plan = RiccatiPlan::new(
        RiccatiProblem {
            a: a.view(),
            b: b.view(),
            q: q.view(),
            r: r.view(),
            e: None,
            s: None,
        },
        RiccatiKind::Continuous,
        RiccatiOptions::new(),
    )?;
    assert_eq!(plan.state_scaling(), &[1.]);
    let result = plan.solve()?;
    assert_eq!(
        result.report().termination(),
        DirectSolveTermination::Accepted
    );
    assert!((result.solution().unwrap().as_slice()[0] / 1e300 - 1.).abs() <= 64. * f64::EPSILON);
    assert!((result.gain().unwrap().as_slice()[0] / 1e150 - 1.).abs() <= 64. * f64::EPSILON);
    // A root beyond binary64 range must still return a structured error.
    let tiny_b = Matrix::from_vec(1, 1, vec![1e-200])?;
    let error = RiccatiPlan::new(
        RiccatiProblem {
            a: a.view(),
            b: tiny_b.view(),
            q: q.view(),
            r: r.view(),
            e: None,
            s: None,
        },
        RiccatiKind::Continuous,
        RiccatiOptions::new(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    Ok(())
}

fn complex_scalar<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let real = |value| R::try_from_f64_lossy(value).unwrap();
    let a = Matrix::from_vec(1, 1, vec![Complex::new(real(0.25), real(0.125))]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![Complex::new(real(0.5), real(0.75))]).unwrap();
    let r = Matrix::from_vec(1, 1, vec![Complex::new(real(2.), R::ZERO)]).unwrap();
    let x = 1.5;
    let aa = 0.25 * 0.25 + 0.125 * 0.125;
    let bb = 0.5 * 0.5 + 0.75 * 0.75;
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        let q = if kind == RiccatiKind::Continuous {
            bb * x * x / 2. - 0.5 * x
        } else {
            x * (1. - aa * 2. / (2. + bb * x))
        };
        let q = Matrix::from_vec(1, 1, vec![Complex::new(real(q), R::ZERO)]).unwrap();
        for balancing in [RiccatiBalancing::None, RiccatiBalancing::Scale] {
            let result = RiccatiPlan::new(
                RiccatiProblem {
                    a: a.view(),
                    b: b.view(),
                    q: q.view(),
                    r: r.view(),
                    e: None,
                    s: None,
                },
                kind,
                RiccatiOptions::new()
                    .with_balancing(balancing)
                    .with_closed_loop_verification(scioxide_linalg::Estimation::Estimate),
            )
            .unwrap()
            .solve()
            .unwrap();
            assert_eq!(
                result.report().termination(),
                DirectSolveTermination::Accepted
            );
            assert!(result.report().stability_margin().unwrap() > R::ZERO);
            assert!(
                (result.solution().unwrap().as_slice()[0] - Complex::new(real(x), R::ZERO)).abs()
                    <= real(500. * x) * R::EPSILON
            );
        }
    }
}

#[test]
fn scalar_roots_support_complex_coefficients() {
    complex_scalar::<f32>();
    complex_scalar::<f64>();
}
#[test]
fn nonstabilizable_and_singular_descriptor() {
    let a = Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap();
    let z = Matrix::zeros(1, 1).unwrap();
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        assert!(
            RiccatiPlan::new(
                RiccatiProblem {
                    a: a.view(),
                    b: z.view(),
                    q: a.view(),
                    r: a.view(),
                    e: None,
                    s: None
                },
                kind,
                RiccatiOptions::new()
            )
            .is_err()
        );
        assert!(
            RiccatiPlan::new(
                RiccatiProblem {
                    a: a.view(),
                    b: a.view(),
                    q: a.view(),
                    r: a.view(),
                    e: Some(z.view()),
                    s: None
                },
                kind,
                RiccatiOptions::new()
            )
            .is_err()
        );
    }
}

#[test]
fn stabilizable_uncontrollable_care_and_optional_verification() -> Result<(), SciOxideError> {
    // The second state is stable but unactuated. X is diagonal and analytic:
    // x11 = 1 + sqrt(2), x22 = 1/4; K = [x11, 0].
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0., 0., -2.])?;
    let b = Matrix::from_vec(2, 1, vec![1., 0.])?;
    let q = Matrix::identity(2)?;
    let r = Matrix::identity(1)?;
    let problem = RiccatiProblem {
        a: a.view(),
        b: b.view(),
        q: q.view(),
        r: r.view(),
        e: None,
        s: None,
    };
    assert_eq!(
        RiccatiOptions::<f64>::new().closed_loop_verification(),
        scioxide_linalg::Estimation::Skip
    );
    let mut sizes = Vec::new();
    for verify in [
        scioxide_linalg::Estimation::Skip,
        scioxide_linalg::Estimation::Estimate,
    ] {
        let plan = RiccatiPlan::new(
            problem,
            RiccatiKind::Continuous,
            RiccatiOptions::new().with_closed_loop_verification(verify),
        )?;
        sizes.push(plan.workspace_requirements(())?.scalar_len()?);
        let mut workspace = plan.workspace(())?;
        let mut x = Matrix::zeros(2, 2)?;
        let mut gain = Matrix::zeros(1, 2)?;
        let report = plan
            .solve_into(
                x.view_mut(),
                gain.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), DirectSolveTermination::Accepted);
        assert_eq!(
            report.work(),
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .total()?
                .iterations()
        );
        assert_eq!(
            report.stability_margin().is_some(),
            verify == scioxide_linalg::Estimation::Estimate
        );
        if let Some(margin) = report.stability_margin() {
            assert!((margin - 1.).abs() < 2048. * f64::EPSILON);
        }
        for (&actual, expected) in x.as_slice().iter().zip([1. + 2.0_f64.sqrt(), 0., 0., 0.25]) {
            assert!((actual - expected).abs() < 2048. * f64::EPSILON * expected.abs().max(1.));
        }
        assert!((gain.as_slice()[0] - (1. + 2.0_f64.sqrt())).abs() < 2048. * f64::EPSILON);
        assert!(gain.as_slice()[1].abs() < 2048. * f64::EPSILON);
    }
    assert!(sizes[0] < sizes[1]);
    Ok(())
}

#[test]
fn empty_state_has_no_stability_margin() -> Result<(), SciOxideError> {
    let empty = Matrix::<f64>::zeros(0, 0)?;
    let problem = RiccatiProblem {
        a: empty.view(),
        b: empty.view(),
        q: empty.view(),
        r: empty.view(),
        e: None,
        s: None,
    };
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        for verify in [
            scioxide_linalg::Estimation::Skip,
            scioxide_linalg::Estimation::Estimate,
        ] {
            let result = RiccatiPlan::new(
                problem,
                kind,
                RiccatiOptions::new().with_closed_loop_verification(verify),
            )?
            .solve()?;
            assert_eq!(
                result.report().termination(),
                DirectSolveTermination::Accepted
            );
            assert_eq!(result.report().residual(), 0.);
            assert!(result.report().stability_margin().is_none());
        }
    }
    Ok(())
}
#[test]
fn large_finite_homogeneous_care_preserves_stability() {
    let a = Matrix::from_vec(1, 1, vec![Complex::new(-1.5e308_f64, 1.5e308)]).unwrap();
    let e = Matrix::from_vec(1, 1, vec![Complex::new(1e308, 0.)]).unwrap();
    let q = Matrix::from_vec(1, 1, vec![Complex::new(3e307, 0.)]).unwrap();
    let b = Matrix::zeros(1, 0).unwrap();
    let r = Matrix::zeros(0, 0).unwrap();
    let plan = RiccatiPlan::new(
        RiccatiProblem {
            a: a.view(),
            b: b.view(),
            q: q.view(),
            r: r.view(),
            e: Some(e.view()),
            s: None,
        },
        RiccatiKind::Continuous,
        RiccatiOptions::new().with_closed_loop_verification(scioxide_linalg::Estimation::Estimate),
    )
    .unwrap();
    let result = plan.solve().unwrap();
    assert_eq!(
        result.report().termination(),
        DirectSolveTermination::Accepted
    );
    assert!(
        (result.solution().unwrap().view().get(0, 0).unwrap().re / 1e-309 - 1.).abs()
            < 512. * f64::EPSILON
    );
    assert!(
        (result.report().stability_margin().unwrap() - core::f64::consts::FRAC_1_SQRT_2).abs()
            < 512. * f64::EPSILON
    );
}

/// Two-state, one-input continuous problem with identity costs.
struct Care {
    a: Matrix<f64>,
    b: Matrix<f64>,
    q: Matrix<f64>,
    r: Matrix<f64>,
}
impl Care {
    fn new() -> Result<Self, SciOxideError> {
        Ok(Self {
            a: Matrix::from_vec(2, 2, vec![0., 1., -2., -3.])?,
            b: Matrix::from_vec(2, 1, vec![0., 1.])?,
            q: Matrix::identity(2)?,
            r: Matrix::identity(1)?,
        })
    }
    fn problem(&self) -> RiccatiProblem<'_, f64> {
        RiccatiProblem {
            a: self.a.view(),
            b: self.b.view(),
            q: self.q.view(),
            r: self.r.view(),
            e: None,
            s: None,
        }
    }
}

#[test]
fn workspace_failures_preserve_outputs() -> Result<(), SciOxideError> {
    let care = Care::new()?;
    let options = RiccatiOptions::new();
    let plan = RiccatiPlan::new(care.problem(), RiccatiKind::Continuous, options)?;
    let mut x = Matrix::from_vec(2, 2, vec![7.; 4])?;
    let mut k = Matrix::from_vec(1, 2, vec![7.; 2])?;
    // The shared cap now also applies to preparation's retained arenas.
    assert_eq!(
        RiccatiPlan::new(
            care.problem(),
            RiccatiKind::Continuous,
            options.with_execution_options(options.execution_options().with_byte_limit(1)),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut short = [0.0; 1];
    assert_eq!(
        plan.solve_into(
            x.view_mut(),
            k.view_mut(),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let mut workspace = plan.workspace(())?;
    let mut wrong = Matrix::from_vec(2, 1, vec![7.; 2])?;
    assert_eq!(
        plan.solve_into(
            x.view_mut(),
            wrong.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(x.as_slice(), &[7.; 4]);
    assert_eq!(k.as_slice(), &[7.; 2]);
    assert_eq!(wrong.as_slice(), &[7.; 2]);
    // Preparation reports QZ work exhaustion instead of a partial plan.
    assert_eq!(
        RiccatiPlan::new(
            care.problem(),
            RiccatiKind::Continuous,
            options.with_execution_options(
                options
                    .execution_options()
                    .with_iteration_budget(Budget::ONE)
            )
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    Ok(())
}

#[test]
fn solve_context_sees_nested_counts_without_consuming_its_allowance() -> Result<(), SciOxideError> {
    let care = Care::new()?;
    let identity = SolveIdentity::new(
        ProblemToken::new(944),
        scioxide_core::algorithm_id!("test", "riccati_parent", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut parent = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let plan = RiccatiPlan::build(
        care.problem(),
        RiccatiKind::Discrete,
        RiccatiOptions::new(),
        &mut parent,
    )?;
    let preparation = parent.counts().nested().iterations();
    // Five stage admissions plus QZ, reordering and SVD work.
    assert!(preparation > 5);
    let mut workspace = plan.workspace(())?;
    let mut x = Matrix::zeros(2, 2)?;
    let mut k = Matrix::zeros(1, 2)?;
    let report = plan.solve_into(x.view_mut(), k.view_mut(), &mut workspace, &mut parent)?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    assert_eq!(parent.counts().own().iterations(), 0);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert!(report.work() > 0);
    assert_eq!(report.work(), attempt.counts().total()?.iterations());
    // Nine recovery stages plus the closed-loop QZ sweeps.
    assert!(attempt.counts().own().iterations() >= 9);
    assert_eq!(
        parent.counts().nested().iterations(),
        preparation + attempt.counts().own().iterations()
    );
    Ok(())
}

#[test]
fn time_limits_require_a_clock() -> Result<(), SciOxideError> {
    use crate::support::clocks::Frozen;
    use scioxide_core::ClockBinding;
    let care = Care::new()?;
    let options = RiccatiOptions::new().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new()
            .with_time_limit(core::time::Duration::from_secs(60)),
    );
    assert_eq!(
        RiccatiPlan::new(care.problem(), RiccatiKind::Continuous, options)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let clock = Frozen;
    let binding = ClockBinding::new(&clock);
    let timed = ExecutionContext::sequential().with_clock(&binding)?;
    let plan = RiccatiPlan::build(care.problem(), RiccatiKind::Continuous, options, timed)?;
    assert_eq!(
        plan.solve().unwrap_err().kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let mut x = Matrix::zeros(2, 2)?;
    let mut k = Matrix::zeros(1, 2)?;
    let report = plan.solve_into(x.view_mut(), k.view_mut(), &mut plan.workspace(())?, timed)?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    Ok(())
}

#[test]
fn nested_preparation_stages_charge_the_parent_scope() -> Result<(), SciOxideError> {
    // A descriptor adds one stage admission plus the nested LU of E; both must
    // reach the parent instead of running in a disconnected root epoch.
    let care = Care::new()?;
    let e = Matrix::identity(2)?;
    let nested = |descriptor: Option<&Matrix<f64>>| -> Result<usize, SciOxideError> {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut parent = state.begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(945),
                scioxide_core::algorithm_id!("test", "riccati_nested", 1),
                0,
                0,
            ),
            &mut budgets,
        )?;
        let _ = RiccatiPlan::build(
            RiccatiProblem {
                e: descriptor.map(Matrix::view),
                ..care.problem()
            },
            RiccatiKind::Continuous,
            RiccatiOptions::new(),
            &mut parent,
        )?;
        Ok(parent.counts().nested().iterations())
    };
    assert!(nested(Some(&e))? >= nested(None)? + 2);
    Ok(())
}

#[test]
fn exact_recovery_products_fit_the_planned_workspace() -> Result<(), SciOxideError> {
    let n = 70;
    let mut a = Matrix::<f64>::identity(n)?;
    for (index, value) in a.as_mut_slice().iter_mut().enumerate() {
        *value = if index % (n + 1) == 0 {
            -2.0
        } else if index % n == index / n + 1 {
            0.5
        } else {
            0.0
        };
    }
    let b = Matrix::from_vec(n, 1, (0..n).map(|i| 1.0 / (i + 1) as f64).collect())?;
    let (q, r, e) = (
        Matrix::identity(n)?,
        Matrix::identity(1)?,
        Matrix::identity(n)?,
    );
    let options = RiccatiOptions::new().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new()
            .with_accuracy(scioxide_core::Accuracy::Exact),
    );
    let plan = RiccatiPlan::new(
        RiccatiProblem {
            a: a.view(),
            b: b.view(),
            q: q.view(),
            r: r.view(),
            e: Some(e.view()),
            s: None,
        },
        RiccatiKind::Continuous,
        options,
    )?;
    let mut workspace = plan.workspace(())?;
    let mut x = Matrix::zeros(n, n)?;
    let mut k = Matrix::zeros(1, n)?;
    let report = plan.solve_into(
        x.view_mut(),
        k.view_mut(),
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    Ok(())
}
