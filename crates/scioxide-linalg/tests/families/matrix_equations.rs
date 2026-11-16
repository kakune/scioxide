//! Manufactured original-coordinate equation, workspace and resource tests.
#![allow(clippy::unwrap_used)]

use scioxide_core::AlgebraicScalar;
use scioxide_core::NumericScalar;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Complex, ExecutionContext, ProblemToken, RealScalar,
    SciOxideError, SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::{
    DirectSolveTermination, Estimation, LinalgWorkspace, LyapunovKind, LyapunovPlan, Matrix,
    MatrixEquationOptions, SylvesterPlan, SylvesterSign,
};
fn val<S: NumericScalar>(r: f64, i: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(r).unwrap(),
        S::Real::try_from_f64_lossy(i).unwrap(),
    )
}
fn close<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>) {
    let tol = S::Real::try_from_usize_exact(4096).unwrap() * <S::Real as RealScalar>::EPSILON;
    for (&x, &y) in a.as_slice().iter().zip(b.as_slice()) {
        assert!(
            (x - y).abs() <= tol * y.abs().max(S::Real::ONE),
            "{x:?} != {y:?}"
        );
    }
}
fn run<S: NumericScalar>() {
    // Real conjugate pair plus nonnormal coupling; complex tests add genuine phases.
    let a = Matrix::from_vec(
        3,
        3,
        vec![
            val::<S>(-2., 0.2),
            val(3., -0.1),
            val(0.5, 0.3),
            val(-1., 0.1),
            val(-2., 0.4),
            val(1., 0.),
            val(0., 0.),
            val(0., 0.),
            val(-4., 0.3),
        ],
    )
    .unwrap();
    let b = Matrix::from_vec(
        2,
        2,
        vec![val::<S>(5., 0.4), val(2., -0.2), val(-1., 0.), val(5., 0.1)],
    )
    .unwrap();
    let x = Matrix::from_vec(
        3,
        2,
        (0..6)
            .map(|i| val::<S>((i + 1) as f64 / 3., (i % 3) as f64 / 5.))
            .collect(),
    )
    .unwrap();
    for sign in [SylvesterSign::Plus, SylvesterSign::Minus] {
        for adjoint in [false, true] {
            let av = if adjoint {
                a.view().adjoint()
            } else {
                a.view()
            };
            let bv = if adjoint {
                b.view().transpose()
            } else {
                b.view()
            };
            let aa = av.try_to_matrix().unwrap();
            let bb = bv.try_to_matrix().unwrap();
            let c = aa
                .view()
                .mul_mat(&x)
                .unwrap()
                .view()
                .try_add(
                    &x.view()
                        .mul_mat(&bb)
                        .unwrap()
                        .view()
                        .try_scale(if sign == SylvesterSign::Minus {
                            -S::ONE
                        } else {
                            S::ONE
                        })
                        .unwrap(),
                )
                .unwrap();
            let p = SylvesterPlan::new(av, bv, sign, MatrixEquationOptions::new()).unwrap();
            let mut w = p.workspace(()).unwrap();
            let mut out = Matrix::zeros(3, 2).unwrap();
            let report = p
                .solve_into(
                    c.view(),
                    out.view_mut(),
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            assert!(report.separation().unwrap() > S::Real::ZERO);
            close(&out, &x);
        }
    }
    let x = Matrix::from_vec(
        3,
        3,
        vec![
            val::<S>(2., 0.),
            val(1., 0.2),
            val(0.5, -0.1),
            val(1., -0.2),
            val(3., 0.),
            val(0., 0.4),
            val(0.5, 0.1),
            val(0., -0.4),
            val(1., 0.),
        ],
    )
    .unwrap();
    for kind in [LyapunovKind::Continuous, LyapunovKind::Discrete] {
        let a = if kind == LyapunovKind::Discrete {
            a.view().try_scale(val::<S>(0.1, 0.)).unwrap()
        } else {
            a.clone()
        };
        let ah = a.view().adjoint().try_to_matrix().unwrap();
        let c = if kind == LyapunovKind::Discrete {
            x.view()
                .try_sub(&a.view().mul_mat(&x).unwrap().view().mul_mat(&ah).unwrap())
                .unwrap()
        } else {
            a.view()
                .mul_mat(&x)
                .unwrap()
                .view()
                .try_add(&x.view().mul_mat(&ah).unwrap())
                .unwrap()
        };
        let p = LyapunovPlan::new(a.view(), kind, MatrixEquationOptions::new()).unwrap();
        let mut w = p.workspace(()).unwrap();
        let mut out = Matrix::zeros(3, 3).unwrap();
        let report = p
            .solve_into(
                c.view(),
                out.view_mut(),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), DirectSolveTermination::Accepted);
        close(&out, &x);
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

// Upper Schur forms with pairs straddling both forward and reverse panel edges.
fn panel_coefficient<S: NumericScalar>(n: usize, diagonal: f64) -> Matrix<S> {
    let mut a = Matrix::zeros(n, n).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, val::<S>(diagonal, 0.03)).unwrap();
        for j in i + 1..n {
            a.view_mut()
                .set(i, j, val::<S>(((i + j) % 7) as f64 / 200., 0.002))
                .unwrap();
        }
    }
    if S::IS_REAL {
        for i in [0, 31, n - 33, n - 2] {
            a.view_mut().set(i, i + 1, val::<S>(0.4, 0.)).unwrap();
            a.view_mut().set(i + 1, i, val::<S>(-0.3, 0.)).unwrap();
        }
    }
    a
}

fn blocked<S: NumericScalar>() {
    let (m, n) = (67, 69);
    let a = panel_coefficient::<S>(m, -2.);
    let b = panel_coefficient::<S>(n, 8.);
    let x = Matrix::from_vec(
        m,
        n,
        (0..m * n)
            .map(|k| val::<S>(1. + (k % 11) as f64 / 20., (k % 5) as f64 / 50.))
            .collect(),
    )
    .unwrap();
    for sign in [SylvesterSign::Plus, SylvesterSign::Minus] {
        let ax = a.view().mul_mat(&x).unwrap();
        let mut xb = x.view().mul_mat(&b).unwrap();
        if sign == SylvesterSign::Minus {
            xb = xb.view().try_scale(-S::ONE).unwrap();
        }
        let c = ax.view().try_add(&xb).unwrap();
        let plan = SylvesterPlan::new(&a, &b, sign, MatrixEquationOptions::new()).unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        let mut out = Matrix::zeros(m, n).unwrap();
        let report = plan
            .solve_into(
                &c,
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), DirectSolveTermination::Accepted);
        assert!(report.separation().unwrap() > S::Real::ZERO);
        close(&out, &x);
    }
    let x = Matrix::from_vec(
        m,
        m,
        (0..m * m)
            .map(|k| {
                let (i, j) = (k / m, k % m);
                val::<S>(
                    1. + ((i + j) % 11) as f64 / 20.,
                    (i as f64 - j as f64) / 100.,
                )
            })
            .collect(),
    )
    .unwrap();
    for kind in [LyapunovKind::Continuous, LyapunovKind::Discrete] {
        let a = if kind == LyapunovKind::Discrete {
            a.view().try_scale(val::<S>(0.1, 0.)).unwrap()
        } else {
            a.clone()
        };
        let ax = a.view().mul_mat(&x).unwrap();
        let c = if kind == LyapunovKind::Continuous {
            ax.view()
                .try_add(&x.view().mul_mat(a.view().adjoint()).unwrap())
                .unwrap()
        } else {
            x.view()
                .try_sub(&ax.view().mul_mat(a.view().adjoint()).unwrap())
                .unwrap()
        };
        let plan = LyapunovPlan::new(&a, kind, MatrixEquationOptions::new()).unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        let mut out = Matrix::zeros(m, m).unwrap();
        let report = plan
            .solve_into(
                &c,
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), DirectSolveTermination::Accepted);
        assert!(report.separation().unwrap() > S::Real::ZERO);
        close(&out, &x);
    }
}

#[test]
fn blocked_real32() {
    blocked::<f32>();
}
#[test]
fn blocked_real64() {
    blocked::<f64>();
}
#[test]
fn blocked_complex32() {
    blocked::<Complex<f32>>();
}
#[test]
fn blocked_complex64() {
    blocked::<Complex<f64>>();
}

#[test]
fn nearly_common_eigenvalues_have_small_separation() {
    let n = 65;
    let delta = 2.0_f64.powi(-20);
    let mut a = Matrix::zeros(n, n).unwrap();
    let mut b = Matrix::zeros(n, n).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, 1. + i as f64).unwrap();
        b.view_mut().set(i, i, 1. + i as f64 + delta).unwrap();
    }
    let c = Matrix::identity(n)
        .unwrap()
        .view()
        .try_scale(-delta)
        .unwrap();
    let result = SylvesterPlan::new(&a, &b, SylvesterSign::Minus, MatrixEquationOptions::new())
        .unwrap()
        .solve(&c)
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DirectSolveTermination::Accepted
    );
    // Schur preparation is backward stable; forward accuracy is amplified by
    // the inverse eigenvalue gap. Existing residual tolerances are unchanged.
    let bound = 256. * f64::EPSILON * n as f64 / delta;
    for (i, &value) in result.solution().unwrap().as_slice().iter().enumerate() {
        assert!((value - if i / n == i % n { 1. } else { 0. }).abs() < bound);
    }
    assert!((result.report().separation().unwrap() / delta - 1.).abs() < bound);
}

#[test]
fn block_scaling_preserves_independent_complex_ranges_and_transactions() {
    let n = 35;
    let a = Matrix::<Complex<f64>>::identity(n).unwrap();
    let mut x = Matrix::zeros(n, n).unwrap();
    for i in 0..n {
        let value = if i % 2 == 0 {
            Complex::new(1e280, 1e-280)
        } else {
            Complex::new(1e-280, -1e280)
        };
        x.view_mut().set(i, i, value).unwrap();
    }
    let c = x.view().try_scale(Complex::new(2., 0.)).unwrap();
    let plan = SylvesterPlan::new(
        &a,
        &a,
        SylvesterSign::Plus,
        MatrixEquationOptions::new().with_separation(Estimation::Skip),
    )
    .unwrap();
    let result = plan.solve(&c).unwrap();
    assert_eq!(
        result.report().termination(),
        DirectSolveTermination::Accepted
    );
    for (&actual, &expected) in result
        .solution()
        .unwrap()
        .as_slice()
        .iter()
        .zip(x.as_slice())
    {
        for (u, v) in [(actual.re, expected.re), (actual.im, expected.im)] {
            assert!((u - v).abs() <= 4096. * f64::EPSILON * v.abs());
        }
    }
    let a = Matrix::identity(n)
        .unwrap()
        .view()
        .try_scale(1e-280)
        .unwrap();
    let c = Matrix::identity(n)
        .unwrap()
        .view()
        .try_scale(1e280)
        .unwrap();
    let plan = SylvesterPlan::new(
        &a,
        &a,
        SylvesterSign::Plus,
        MatrixEquationOptions::new().with_separation(Estimation::Skip),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = Matrix::from_vec(n, n, vec![42.; n * n]).unwrap();
    let error = plan
        .solve_into(
            &c,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    assert!(output.as_slice().iter().all(|&value| value == 42.));
}

#[test]
fn small_pivots_reframe_a_partially_solved_panel() {
    let n = 35;
    let a = Matrix::identity(n)
        .unwrap()
        .view()
        .try_scale(1e-70)
        .unwrap();
    let mut c = Matrix::zeros(n, n).unwrap();
    let mut x = Matrix::zeros(n, n).unwrap();
    for i in 0..n {
        c.view_mut()
            .set(i, i, if i % 2 == 0 { 1e-70 } else { 1e70 })
            .unwrap();
        x.view_mut()
            .set(i, i, if i % 2 == 0 { 0.5 } else { 5e139 })
            .unwrap();
    }
    let result = SylvesterPlan::new(
        &a,
        &a,
        SylvesterSign::Plus,
        MatrixEquationOptions::new().with_separation(Estimation::Skip),
    )
    .unwrap()
    .solve(&c)
    .unwrap();
    assert_eq!(
        result.report().termination(),
        DirectSolveTermination::Accepted
    );
    close(result.solution().unwrap(), &x);
}
#[test]
fn range_resonance_and_transactions() {
    for scale in [1e-280, 1., 1e280] {
        let a = Matrix::from_vec(1, 1, vec![scale]).unwrap();
        let p = SylvesterPlan::new(
            a.view(),
            a.view(),
            SylvesterSign::Plus,
            MatrixEquationOptions::new(),
        )
        .unwrap();
        let c = Matrix::from_vec(1, 1, vec![scale]).unwrap();
        let result = p.solve(c.view()).unwrap();
        assert_eq!(result.solution().unwrap().as_slice(), &[0.5]);
    }
    let a = Matrix::from_vec(1, 1, vec![1.]).unwrap();
    let p = SylvesterPlan::new(
        a.view(),
        a.view(),
        SylvesterSign::Minus,
        MatrixEquationOptions::new(),
    )
    .unwrap();
    let mut w = p.workspace(()).unwrap();
    let mut out = Matrix::from_vec(1, 1, vec![42.]).unwrap();
    assert!(matches!(
        (p.solve_into(
            a.view(),
            out.view_mut(),
            &mut w,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
    ));
    assert_eq!(out.as_slice(), &[42.]);
    let p = LyapunovPlan::new(
        a.view(),
        LyapunovKind::Continuous,
        MatrixEquationOptions::new().with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
        ),
    )
    .unwrap();
    assert_eq!(
        p.solve_into(
            a.view(),
            out.view_mut(),
            &mut w,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(out.as_slice(), &[42.]);
    let a = Matrix::<f64>::zeros(0, 0).unwrap();
    assert_eq!(
        LyapunovPlan::new(
            a.view(),
            LyapunovKind::Discrete,
            MatrixEquationOptions::new()
        )
        .unwrap()
        .solve(a.view())
        .unwrap()
        .report()
        .termination(),
        DirectSolveTermination::Accepted
    );
}

/// `A` (3 x 3), `B` (2 x 2) and a right-hand side `C` (3 x 2).
fn coefficients() -> Result<[Matrix<f64>; 3], SciOxideError> {
    Ok([
        Matrix::from_vec(3, 3, vec![-2., 3., 0.5, -1., -2., 1., 0., 0., -4.])?,
        Matrix::from_vec(2, 2, vec![5., 2., -1., 5.])?,
        Matrix::from_vec(3, 2, vec![1., 2., 3., 4., 5., 6.])?,
    ])
}

#[test]
fn workspace_failures_preserve_output() -> Result<(), SciOxideError> {
    let [a, b, c] = coefficients()?;
    let options = MatrixEquationOptions::new();
    let plan = SylvesterPlan::new(&a, &b, SylvesterSign::Plus, options)?;
    let mut out = Matrix::from_vec(3, 2, vec![7.; 6])?;
    let before = out.clone();
    // A configured byte limit rejects preparation scratch, solve requirements
    // and owned growth.
    let limit =
        |bytes| options.with_execution_options(options.execution_options().with_byte_limit(bytes));
    assert_eq!(
        SylvesterPlan::new(&a, &b, SylvesterSign::Plus, limit(1))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let solve_bytes = plan.workspace_requirements(())?.required_bytes::<f64>()?;
    let limited = SylvesterPlan::new(&a, &b, SylvesterSign::Plus, limit(solve_bytes - 1))?;
    assert_eq!(
        limited.workspace_requirements(()).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut owned = LinalgWorkspace::new();
    assert_eq!(
        limited
            .solve_into(
                &c,
                out.view_mut(),
                &mut owned,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(out, before);
    assert_eq!(owned.retained_bytes()?, 0);
    // Borrowed storage never grows.
    let mut short = [0.0; 1];
    assert_eq!(
        plan.solve_into(
            &c,
            out.view_mut(),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, before);
    // Output, right-hand side and coefficient shapes are dimension errors.
    let mut workspace = plan.workspace(())?;
    let mut wrong = Matrix::from_vec(2, 3, vec![7.; 6])?;
    assert_eq!(
        plan.solve_into(
            &c,
            wrong.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(wrong.as_slice(), &[7.; 6]);
    assert_eq!(
        plan.solve_into(
            c.view().transpose(),
            out.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        SylvesterPlan::new(&c, &b, SylvesterSign::Plus, options)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let mut nan = c.clone();
    nan.as_mut_slice()[0] = f64::NAN;
    assert_eq!(
        plan.solve_into(
            &nan,
            out.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out, before);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    // Work exhaustion fails inside the attempt without publishing.
    let starved = LyapunovPlan::new(
        &a,
        LyapunovKind::Continuous,
        options.with_execution_options(
            (options)
                .execution_options()
                .with_iteration_budget(Budget::ONE),
        ),
    )?;
    let mut square = Matrix::from_vec(3, 3, vec![7.; 9])?;
    let identity = Matrix::<f64>::identity(3)?;
    let mut lyapunov = starved.workspace(())?;
    assert_eq!(
        starved
            .solve_into(
                &identity,
                square.view_mut(),
                &mut lyapunov,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(square.as_slice(), &[7.; 9]);
    assert_eq!(
        lyapunov.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    Ok(())
}

#[test]
fn solve_context_sees_nested_counts_without_consuming_its_allowance() -> Result<(), SciOxideError> {
    // A symmetric coefficient needs QR sweeps to reach its Schur form.
    let a = Matrix::from_vec(
        4,
        4,
        vec![
            4., 1., 2., 0.5, 1., 3., 0., 1., 2., 0., 5., 1., 0.5, 1., 1., 2.,
        ],
    )?;
    let [_, b, _] = coefficients()?;
    let c = Matrix::from_vec(4, 2, vec![1., 2., 3., 4., 5., 6., 7., 8.])?;
    let identity = SolveIdentity::new(
        ProblemToken::new(944),
        scioxide_core::algorithm_id!("test", "matrix_equation_parent", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut parent = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let plan = SylvesterPlan::build(
        &a,
        &b,
        SylvesterSign::Plus,
        MatrixEquationOptions::new(),
        &mut parent,
    )?;
    let preparation = parent.counts().nested().iterations();
    assert!(preparation > 0);
    let mut workspace = plan.workspace(())?;
    let mut out = Matrix::zeros(4, 2)?;
    let report = plan.solve_into(&c, out.view_mut(), &mut workspace, &mut parent)?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    assert_eq!(parent.counts().own().iterations(), 0);
    assert_eq!(
        parent.counts().nested().iterations(),
        preparation + report.work()
    );
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.counts().own().iterations(), report.work());
    Ok(())
}

#[test]
fn time_limits_require_a_clock() -> Result<(), SciOxideError> {
    use crate::support::clocks::Frozen;
    use scioxide_core::ClockBinding;
    let [a, b, c] = coefficients()?;
    let options = MatrixEquationOptions::new().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new()
            .with_time_limit(core::time::Duration::from_secs(60)),
    );
    assert_eq!(
        SylvesterPlan::new(&a, &b, SylvesterSign::Plus, options)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let clock = Frozen;
    let binding = ClockBinding::new(&clock);
    let timed = ExecutionContext::sequential().with_clock(&binding)?;
    let plan = SylvesterPlan::build(&a, &b, SylvesterSign::Plus, options, timed)?;
    assert_eq!(
        plan.solve(&c).unwrap_err().kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let mut out = Matrix::zeros(3, 2)?;
    let report = plan.solve_into(&c, out.view_mut(), &mut plan.workspace(())?, timed)?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    Ok(())
}

#[test]
fn real_recovery_reuses_reserved_workspace() -> Result<(), SciOxideError> {
    let n = 65;
    let a = Matrix::<f64>::identity(n)?;
    let expected = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|k| {
                if k / n == k % n {
                    if (k / n) % 2 == 0 { 1e280 } else { 1e-280 }
                } else {
                    0.
                }
            })
            .collect(),
    )?;
    let rhs = expected.view().try_scale(2.)?;
    let plan = SylvesterPlan::new(
        &a,
        &a,
        SylvesterSign::Plus,
        MatrixEquationOptions::new().with_separation(Estimation::Skip),
    )?;
    let requirements = plan.workspace_requirements(())?;
    let mut workspace = LinalgWorkspace::new();
    workspace.try_reserve(&requirements)?;
    let mut output = Matrix::zeros(n, n)?;
    for _ in 0..2 {
        let report = plan.solve_into(
            &rhs,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )?;
        assert_eq!(report.termination(), DirectSolveTermination::Accepted);
        for (&actual, &wanted) in output.as_slice().iter().zip(expected.as_slice()) {
            assert!((actual - wanted).abs() <= 4096. * f64::EPSILON * wanted.abs());
        }
    }
    Ok(())
}

#[test]
fn final_schur_pair_does_not_create_an_empty_panel() -> Result<(), SciOxideError> {
    for n in [33, 65] {
        let mut a = Matrix::<f64>::identity(n)?.view().try_scale(2.)?;
        a.view_mut().set(n - 2, n - 1, 0.5)?;
        a.view_mut().set(n - 1, n - 2, -0.25)?;
        let b = Matrix::<f64>::identity(3)?.view().try_scale(4.)?;
        let expected =
            Matrix::from_vec(n, 3, (0..n * 3).map(|k| 1. + (k % 7) as f64 / 8.).collect())?;
        let ax = a.view().mul_mat(&expected)?;
        let xb = expected.view().mul_mat(&b)?;
        let rhs = ax.view().try_add(&xb)?;
        let result = SylvesterPlan::new(&a, &b, SylvesterSign::Plus, MatrixEquationOptions::new())?
            .solve(&rhs)?;
        assert_eq!(
            result.report().termination(),
            DirectSolveTermination::Accepted
        );
        close(result.solution().unwrap(), &expected);
    }
    Ok(())
}

#[test]
fn preparation_scratch_honors_the_byte_limit() {
    let a = Matrix::from_vec(3, 3, vec![-2.0, 1.0, 0.0, 0.0, -3.0, 1.0, 0.0, 0.0, -4.0]).unwrap();
    let limited = MatrixEquationOptions::new()
        .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(1));
    for error in [
        SylvesterPlan::new(a.view(), a.view(), SylvesterSign::Plus, limited).unwrap_err(),
        LyapunovPlan::new(a.view(), LyapunovKind::Continuous, limited).unwrap_err(),
    ] {
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
    }
}
