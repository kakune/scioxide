//! Analytic Jordan identities, noncommuting Frechet directions, branches and resources.
#![allow(clippy::unwrap_used)]

use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptOutcome, Budget, Complex, ExecutionContext, NumericScalar, ProblemToken, RealScalar,
    SciOxideError, SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::{
    AnalyticMatrixFunction, LinalgWorkspace, Matrix, MatrixFunction as F, MatrixFunctionOptions,
    MatrixFunctionPlan, MatrixFunctionWorkspaceRequest as Request,
};

fn plan<'f, S: NumericScalar>(
    a: &Matrix<S>,
    function: F<'f, S::Real>,
    options: &MatrixFunctionOptions<S::Real>,
) -> MatrixFunctionPlan<'f, S::Real>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    MatrixFunctionPlan::new(a, function, options).unwrap()
}

fn run<S: NumericScalar>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let s = |x| S::from_real(S::Real::try_from_f64_lossy(x).unwrap());
    let a = Matrix::from_vec(
        3,
        3,
        vec![
            s(2.),
            s(1.),
            s(0.),
            s(0.),
            s(2.),
            s(1.),
            s(0.),
            s(0.),
            s(2.),
        ],
    )
    .unwrap();
    let options = MatrixFunctionOptions::default();
    let mut w = plan(&a, F::Exp, &options)
        .workspace(&Request::Frechet)
        .unwrap();
    let mut out = Matrix::<S>::zeros(3, 3).unwrap();
    let tol = S::Real::try_from_usize_exact(8192).unwrap() * <S::Real as RealScalar>::EPSILON;
    // Independent scalar constants and Jordan Taylor coefficients.
    let e = 7.389_056_098_930_65_f64;
    let sqrt = core::f64::consts::SQRT_2;
    let cases = [
        (F::Exp, [e, e, e / 2.]),
        (F::Log, [core::f64::consts::LN_2, 0.5, -0.125]),
        (F::Sqrt, [sqrt, 1. / (2. * sqrt), -1. / (16. * sqrt)]),
        (F::Sign, [1., 0., 0.]),
        (
            F::Sin,
            [
                0.909_297_426_825_681_7,
                -0.416_146_836_547_142_4,
                -0.454_648_713_412_840_85,
            ],
        ),
        (
            F::Cos,
            [
                -0.416_146_836_547_142_4,
                -0.909_297_426_825_681_7,
                0.208_073_418_273_571_2,
            ],
        ),
        (
            F::Sinh,
            [
                3.626_860_407_847_019,
                3.762_195_691_083_631_4,
                1.813_430_203_923_509_5,
            ],
        ),
        (
            F::Cosh,
            [
                3.762_195_691_083_631_4,
                3.626_860_407_847_019,
                1.881_097_845_541_815_7,
            ],
        ),
    ];
    for (f, c) in cases {
        let p = plan(&a, f, &options);
        p.evaluate_into(out.view_mut(), &mut w, ExecutionContext::sequential())
            .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if j >= i { s(c[j - i]) } else { S::ZERO };
                assert!(
                    (out.view().get(i, j).unwrap() - expected).abs()
                        <= tol * expected.abs().max(S::Real::ONE),
                    "{f:?}: {:?} != {expected:?}",
                    out.view().get(i, j).unwrap()
                );
            }
        }
    }
    let direction = Matrix::from_vec(
        3,
        3,
        vec![
            s(0.),
            s(0.),
            s(0.),
            s(0.),
            s(0.),
            s(0.),
            s(1.),
            s(0.),
            s(0.),
        ],
    )
    .unwrap();
    let exp = plan(&a, F::Exp, &options);
    exp.frechet_into(
        &direction,
        out.view_mut(),
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    // Integral exp((1-t)N) E exp(tN) dt has N^p E N^q/(p+q+1)!.
    let factorial = [1., 1., 2., 6., 24., 120.];
    for i in 0..3 {
        for j in 0..3 {
            let expected = s(e / factorial[3 - i + j]);
            assert!((out.view().get(i, j).unwrap() - expected).abs() < tol * expected.abs());
        }
    }
    let mut complex = Matrix::<Complex<S::Real>>::zeros(3, 3).unwrap();
    for f in [
        F::Tan,
        F::Tanh,
        F::Power(S::Real::try_from_f64_lossy(0.3).unwrap()),
    ] {
        let report = plan(&a, f, &options)
            .evaluate_into(complex.view_mut(), &mut w, ExecutionContext::sequential())
            .unwrap();
        assert!(report.output_norm().round().unwrap().is_finite());
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

fn evaluate_complex(p: &MatrixFunctionPlan<'_, f64>) -> Matrix<Complex<f64>> {
    let mut value = Matrix::zeros(p.order(), p.order()).unwrap();
    p.evaluate_into(
        value.view_mut(),
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    value
}

#[test]
fn branch_transactions_empty_and_power() {
    let a = Matrix::from_vec(1, 1, vec![-1.]).unwrap();
    let options = MatrixFunctionOptions::default();
    let sqrt = plan(&a, F::Sqrt, &options);
    let root = evaluate_complex(&sqrt);
    assert!((root.view().get(0, 0).unwrap() - Complex::new(0., 1.)).norm() < 1e-14);
    // Branch cuts are validated when the plan is built.
    assert!(matches!(
        MatrixFunctionPlan::new(&a, F::Log, &options)
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
    assert_eq!(
        MatrixFunctionPlan::new(&a, F::Power(f64::NAN), &options)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(
        evaluate_complex(&plan(&a, F::Power(0.), &options)).as_slice(),
        &[Complex::new(1., 0.)]
    );
    // Real output of a complex branch is rejected per call, preserving output.
    let mut out = Matrix::from_vec(1, 1, vec![42.]).unwrap();
    let mut w = sqrt.workspace(&Request::Condition).unwrap();
    assert_eq!(
        sqrt.evaluate_into(out.view_mut(), &mut w, ExecutionContext::sequential())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(out.as_slice(), &[42.]);
    let limited = plan(
        &a,
        F::Exp,
        &options.with_execution_options(
            (options)
                .execution_options()
                .with_iteration_budget(Budget::ONE),
        ),
    );
    assert_eq!(
        limited
            .evaluate_into(out.view_mut(), &mut w, ExecutionContext::sequential())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(out.as_slice(), &[42.]);
    let empty = Matrix::<f64>::zeros(0, 0).unwrap();
    assert!(
        evaluate_complex(&plan(&empty, F::Exp, &options))
            .as_slice()
            .is_empty()
    );
}
#[test]
fn nonnormal_reconstruction() {
    let a = Matrix::from_vec(3, 3, vec![3., -2., 1., 1., 0., 1., 0., 0., 2.]).unwrap();
    let options = MatrixFunctionOptions::default();
    let root = evaluate_complex(&plan(&a, F::Sqrt, &options));
    let square = root.view().mul_mat(&root).unwrap();
    for (&v, &x) in square.as_slice().iter().zip(a.as_slice()) {
        assert!((v - Complex::new(x, 0.)).norm() < 1e-10);
    }
    let logarithm = evaluate_complex(&plan(&a, F::Log, &options));
    let mut restored = Matrix::<Complex<f64>>::zeros(3, 3).unwrap();
    plan(&logarithm, F::Exp, &options)
        .evaluate_into(
            restored.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    for (&v, &x) in restored.as_slice().iter().zip(a.as_slice()) {
        assert!((v - Complex::new(x, 0.)).norm() < 1e-10);
    }
}

struct Polynomial;
impl AnalyticMatrixFunction<f64> for Polynomial {
    fn radius(&self, _: Complex<f64>) -> f64 {
        10.
    }
    fn coefficient(&self, c: Complex<f64>, k: usize) -> Result<Complex<f64>, SciOxideError> {
        Ok(match k {
            0 => c * c,
            1 => c * 2.,
            2 => Complex::new(1., 0.),
            _ => Complex::new(0., 0.),
        })
    }
    fn tail_bound(&self, c: Complex<f64>, k: usize, r: f64) -> Result<f64, SciOxideError> {
        Ok(match k {
            0 => 2. * c.norm() * r + r * r,
            1 => r * r,
            _ => 0.,
        })
    }
}

#[test]
fn condition_and_analytic_callback() {
    let a = Matrix::from_vec(1, 1, vec![2.0_f64]).unwrap();
    let options = MatrixFunctionOptions::default();
    let exp = plan(&a, F::Exp, &options);
    let mut w = exp.workspace(&Request::Condition).unwrap();
    let mut value = Matrix::<f64>::zeros(1, 1).unwrap();
    let result = exp
        .condition_into(value.view_mut(), &mut w, ExecutionContext::sequential())
        .unwrap();
    assert!((result.absolute() - 7.38905609893065).abs() < 1e-12);
    assert!((result.relative().unwrap() - 2.).abs() < 1e-12);
    assert!((value.view().get(0, 0).unwrap() - 7.38905609893065).abs() < 1e-12);
    let result = plan(&a, F::Sign, &options)
        .condition_into(value.view_mut(), &mut w, ExecutionContext::sequential())
        .unwrap();
    assert_eq!(result.absolute(), 0.);
    assert_eq!(value.as_slice(), &[1.]);
    let polynomial = plan(&a, F::Analytic(&Polynomial), &options);
    let mut out = Matrix::<f64>::zeros(1, 1).unwrap();
    polynomial
        .evaluate_into(out.view_mut(), &mut w, ExecutionContext::sequential())
        .unwrap();
    assert_eq!(out.as_slice(), &[4.]);
    // Adjoint Frechet applications need real Taylor coefficients.
    assert_eq!(
        polynomial
            .condition_into(out.view_mut(), &mut w, ExecutionContext::sequential())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(out.as_slice(), &[4.]);
    assert_eq!(format!("{:?}", F::Analytic(&Polynomial)), "Analytic(..)");
}
#[test]
fn clustered_schur_parlett_and_custom_frechet() {
    struct Square;
    impl AnalyticMatrixFunction<f64> for Square {
        fn radius(&self, _: Complex<f64>) -> f64 {
            1000.
        }
        fn coefficient(&self, c: Complex<f64>, k: usize) -> Result<Complex<f64>, SciOxideError> {
            Polynomial.coefficient(c, k)
        }
        fn tail_bound(&self, c: Complex<f64>, k: usize, r: f64) -> Result<f64, SciOxideError> {
            Polynomial.tail_bound(c, k, r)
        }
    }
    for diagonal in [[1., 3., 5.], [1., 3., 1.], [2., 2., 2.]] {
        let a = Matrix::from_vec(
            3,
            3,
            vec![
                diagonal[0],
                0.25,
                -0.5,
                0.,
                diagonal[1],
                0.125,
                0.,
                0.,
                diagonal[2],
            ],
        )
        .unwrap();
        let e = Matrix::from_vec(3, 3, vec![1., 0., 2., 3., -1., 0., 0., 1., 2.]).unwrap();
        let p = plan(&a, F::Analytic(&Square), &MatrixFunctionOptions::default());
        let mut w = p.workspace(&Request::Frechet).unwrap();
        let mut out = Matrix::<f64>::zeros(3, 3).unwrap();
        p.evaluate_into(out.view_mut(), &mut w, ExecutionContext::sequential())
            .unwrap();
        let expected = a.view().mul_mat(&a).unwrap();
        for (&x, &y) in out.as_slice().iter().zip(expected.as_slice()) {
            assert!((x - y).abs() < 1e-11);
        }
        p.frechet_into(&e, out.view_mut(), &mut w, ExecutionContext::sequential())
            .unwrap();
        let expected = a
            .view()
            .mul_mat(&e)
            .unwrap()
            .view()
            .try_add(&e.view().mul_mat(&a).unwrap())
            .unwrap();
        for (&x, &y) in out.as_slice().iter().zip(expected.as_slice()) {
            assert!((x - y).abs() < 1e-10, "{x} != {y}");
        }
    }
}
#[test]
fn real_direction_can_use_prepared_complex_output() {
    let a = Matrix::from_vec(1, 1, vec![Complex::new(2.0_f64, 0.5)]).unwrap();
    let p = plan(&a, F::Exp, &MatrixFunctionOptions::default());
    let e = Matrix::<f64>::identity(1).unwrap();
    let mut out = Matrix::<Complex<f64>>::zeros(1, 1).unwrap();
    let mut workspace = p.workspace(&Request::Frechet).unwrap();
    p.frechet_into(
        &e,
        out.view_mut(),
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let expected = Complex::new(2.0_f64.exp() * 0.5_f64.cos(), 2.0_f64.exp() * 0.5_f64.sin());
    assert!((out.view().get(0, 0).unwrap() - expected).norm() < 1e-12);
}

#[test]
fn resource_storage_and_capability_failures_preserve_output() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0.5, -0.25, 2.0]).unwrap();
    let options = MatrixFunctionOptions::default();
    assert_eq!(
        MatrixFunctionPlan::new(
            &a,
            F::Exp,
            &options.with_execution_options((options).execution_options().with_byte_limit(1))
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        MatrixFunctionPlan::new(
            &a,
            F::Exp,
            &options.with_execution_options(
                (options)
                    .execution_options()
                    .with_time_limit(core::time::Duration::from_secs(1))
            )
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    // Construction scratch (2n^2 + n complex values) fits; evaluation does not.
    let limited = plan(
        &a,
        F::Exp,
        &options.with_execution_options((options).execution_options().with_byte_limit(200)),
    );
    assert_eq!(
        limited
            .workspace_requirements(&Request::Evaluate)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut out = Matrix::from_vec(2, 2, vec![9.0, 8.0, 7.0, 6.0]).unwrap();
    let before = out.clone();
    let mut workspace = LinalgWorkspace::new();
    assert_eq!(
        limited
            .evaluate_into(
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(out, before);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let p = plan(&a, F::Exp, &options);
    let mut short = [Complex::new(0.0, 0.0); 4];
    assert_eq!(
        p.evaluate_into(
            out.view_mut(),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, before);
    let mut wide = Matrix::<f64>::zeros(2, 3).unwrap();
    assert_eq!(
        p.evaluate_into(
            wide.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let nonfinite = Matrix::from_vec(2, 2, vec![f64::NAN, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        p.frechet_into(
            &nonfinite,
            out.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out, before);
}

#[test]
fn solve_context_callers_see_nested_counts() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0.5, -0.25, 2.0]).unwrap();
    let mut budgets = SolveBudgets::unbounded();
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "matrix_function", 1),
        2,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let options = MatrixFunctionOptions::default();
    let p = MatrixFunctionPlan::build(&a, F::Exp, &options, &mut context).unwrap();
    let schur_steps = context.counts().nested().iterations();
    assert_eq!(context.counts().own().iterations(), 0);
    let mut workspace = p.workspace(&Request::Evaluate).unwrap();
    let mut out = Matrix::<f64>::zeros(2, 2).unwrap();
    let report = p
        .evaluate_into(out.view_mut(), &mut workspace, &mut context)
        .unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        schur_steps + report.work()
    );
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.counts().own().iterations(), report.work());
    // Work exhaustion fails the started attempt without touching output.
    let before = out.clone();
    let limited = plan(
        &a,
        F::Exp,
        &options.with_execution_options(
            (options)
                .execution_options()
                .with_iteration_budget(Budget::try_from(report.work() - 1).unwrap()),
        ),
    );
    assert_eq!(
        limited
            .evaluate_into(out.view_mut(), &mut workspace, &mut context)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(out, before);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(attempt.counts().own().iterations(), report.work() - 1);
}

fn conjugate_pairs<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let cast = |x| R::try_from_f64_lossy(x).unwrap();
    let near_cut = if R::MANTISSA_DIGITS <= 24 { 1e-3 } else { 1e-8 };
    for (real, imaginary) in [(0.7, 0.6), (-1.0, near_cut), (1.0, near_cut)] {
        let z = Complex::new(cast(real), cast(imaginary));
        // The real representation of multiplication by z has a conjugate pair.
        let a = Matrix::from_vec(2, 2, vec![z.re, -z.im, z.im, z.re]).unwrap();
        for (function, expected) in [(F::Exp, z.exp()), (F::Log, z.ln()), (F::Sqrt, z.sqrt())] {
            let p = plan(&a, function, &MatrixFunctionOptions::new());
            let mut workspace = p.workspace(&Request::Condition).unwrap();
            let mut out = Matrix::<R>::zeros(2, 2).unwrap();
            {
                p.evaluate_into(
                    out.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            };
            let allowance = cast(512.) * R::EPSILON * expected.norm().max(R::ONE);
            for (&value, expected) in
                out.as_slice()
                    .iter()
                    .zip([expected.re, -expected.im, expected.im, expected.re])
            {
                assert!(
                    (value - expected).abs() <= allowance,
                    "{function:?}: {value:?}, expected {expected:?}"
                );
            }
            // A complex direction exercises both real-linear components. The
            // identity direction commutes, so its derivative is scalar f'(z).
            let derivative = match function {
                F::Exp => z.exp(),
                F::Log => Complex::ONE / z,
                F::Sqrt => Complex::ONE / (z.sqrt() * Complex::from_real(R::TWO)),
                _ => unreachable!(),
            };
            let e = Matrix::from_vec(
                2,
                2,
                vec![
                    Complex::new(R::ONE, R::ONE),
                    Complex::ZERO,
                    Complex::ZERO,
                    Complex::new(R::ONE, R::ONE),
                ],
            )
            .unwrap();
            let mut complex = Matrix::<Complex<R>>::zeros(2, 2).unwrap();
            {
                p.frechet_into(
                    &e,
                    complex.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            };
            for (&value, component) in complex.as_slice().iter().zip([
                derivative.re,
                -derivative.im,
                derivative.im,
                derivative.re,
            ]) {
                let expected = Complex::new(component, component);
                assert!(
                    (value - expected).norm()
                        <= cast(2048.) * R::EPSILON * derivative.norm().max(R::ONE)
                );
            }
        }
    }
}
#[test]
fn conjugate_pairs_real32() {
    conjugate_pairs::<f32>();
}
#[test]
fn conjugate_pairs_real64() {
    conjugate_pairs::<f64>();
}

#[test]
fn nearly_defective_jordan_blocks() {
    // N^3 = delta I. Evaluate independent analytic power series at I+N;
    // small delta makes this close to a defective Jordan block.
    let delta = 1e-12;
    let n = [0., 0.7, 0., 0., 0., 0.7, delta, 0., 0.];
    let a = Matrix::from_vec(
        3,
        3,
        n.iter()
            .enumerate()
            .map(|(i, &x)| x + if i / 3 == i % 3 { 1. } else { 0. })
            .collect(),
    )
    .unwrap();
    for function in [F::Exp, F::Log, F::Sqrt] {
        let mut expected = [0.; 9];
        let mut power = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let mut coefficient = match function {
            F::Exp => core::f64::consts::E,
            F::Log => 0.,
            _ => 1.,
        };
        for k in 0..24 {
            if k > 0 {
                coefficient = match function {
                    F::Exp => coefficient / k as f64,
                    F::Log => (if k % 2 == 0 { -1. } else { 1. }) / k as f64,
                    _ => coefficient * (1.5 - k as f64) / k as f64,
                };
            }
            for (entry, &term) in expected.iter_mut().zip(&power) {
                *entry += coefficient * term;
            }
            let mut next = [0.; 9];
            for i in 0..3 {
                for j in 0..3 {
                    next[i * 3 + j] = (0..3).map(|k| power[i * 3 + k] * n[k * 3 + j]).sum();
                }
            }
            power = next;
        }
        let result = evaluate_complex(&plan(&a, function, &MatrixFunctionOptions::new()));
        for (&value, &expected) in result.as_slice().iter().zip(&expected) {
            assert!(
                (value - Complex::new(expected, 0.)).norm()
                    <= 512. * f64::EPSILON * expected.abs().max(1.),
                "{function:?}: {value:?} != {expected}"
            );
        }
    }
}

#[test]
fn real_schur_tiles_preserve_conjugate_pairs() {
    for n in [15, 16, 17, 31, 32, 33, 47, 48, 49, 64, 65] {
        let mut a = Matrix::<f64>::zeros(n, n).unwrap();
        for i in 0..n {
            a.view_mut().set(i, i, 1. + i as f64 / 128.).unwrap();
            if i + 1 < n {
                a.view_mut().set(i, i + 1, 0.02).unwrap();
            }
        }
        for i in [0, 14, 16, 30, 32, 46, 48, 62]
            .into_iter()
            .filter(|&i| i + 1 < n)
        {
            a.view_mut().set(i, i, 0.7).unwrap();
            a.view_mut().set(i + 1, i + 1, 0.7).unwrap();
            a.view_mut().set(i, i + 1, -0.6).unwrap();
            a.view_mut().set(i + 1, i, 0.6).unwrap();
        }
        let root_plan = plan(&a, F::Sqrt, &MatrixFunctionOptions::new());
        let mut root = Matrix::<f64>::zeros(n, n).unwrap();
        let mut workspace = root_plan.workspace(&Request::Frechet).unwrap();
        {
            root_plan
                .evaluate_into(
                    root.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
        };
        // Independently multiply, keeping the existing 512-epsilon bound.
        for i in 0..n {
            for j in 0..n {
                let square: f64 = (0..n)
                    .map(|k| root.as_slice()[i * n + k] * root.as_slice()[k * n + j])
                    .sum();
                let expected = a.as_slice()[i * n + j];
                assert!((square - expected).abs() <= 512. * f64::EPSILON * expected.abs().max(1.));
            }
        }
        let log = evaluate_complex(&plan(&a, F::Log, &MatrixFunctionOptions::new()));
        let reconstructed = evaluate_complex(&plan(&log, F::Exp, &MatrixFunctionOptions::new()));
        for (&value, &expected) in reconstructed.as_slice().iter().zip(a.as_slice()) {
            assert!(
                (value - Complex::new(expected, 0.)).norm()
                    <= 512. * f64::EPSILON * expected.abs().max(1.)
            );
        }
    }
}

#[test]
fn exponential_avoids_overscaling_and_reports_range_failure() {
    let options = MatrixFunctionOptions::new();
    for value in [1e8, 1e100, f64::MAX / 16.] {
        let a = Matrix::from_vec(2, 2, vec![0., value, 0., 0.]).unwrap();
        let result = evaluate_complex(&plan(&a, F::Exp, &options));
        for (&x, expected) in result.as_slice().iter().zip([1., value, 0., 1.]) {
            assert!(
                (x - Complex::new(expected, 0.)).norm()
                    <= 512. * f64::EPSILON * expected.abs().max(1.)
            );
        }
    }
    let a = Matrix::from_vec(1, 1, vec![1000.]).unwrap();
    let p = plan(&a, F::Exp, &options);
    let mut output = Matrix::from_vec(1, 1, vec![42.]).unwrap();
    assert!(
        p.evaluate_into(
            output.view_mut(),
            &mut p.workspace(&Request::Evaluate).unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output.as_slice(), &[42.]);
}

#[test]
fn logarithm_near_identity_and_branch_errors_remain_transactional() {
    let a = Matrix::from_vec(2, 2, vec![1. + 1e-12, 0.3, 0., 1. + 2e-12]).unwrap();
    let result = evaluate_complex(&plan(&a, F::Log, &MatrixFunctionOptions::new()));
    let diagonal = [
        (a.as_slice()[0] - 1.).ln_1p(),
        (a.as_slice()[3] - 1.).ln_1p(),
    ];
    let off = 0.3 * (((a.as_slice()[3] - a.as_slice()[0]) / a.as_slice()[0]).ln_1p())
        / (a.as_slice()[3] - a.as_slice()[0]);
    for (&value, expected) in result
        .as_slice()
        .iter()
        .zip([diagonal[0], off, 0., diagonal[1]])
    {
        assert!(
            (value - Complex::new(expected, 0.)).norm()
                <= 64. * f64::EPSILON * expected.abs().max(1.)
        );
    }
    for value in [0., -1.] {
        let a = Matrix::from_vec(1, 1, vec![value]).unwrap();
        assert_eq!(
            MatrixFunctionPlan::new(&a, F::Log, &MatrixFunctionOptions::new())
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::DomainError
        );
        let p = plan(&a, F::Sqrt, &MatrixFunctionOptions::new());
        let mut out = Matrix::from_vec(1, 1, vec![Complex::new(42., 0.)]).unwrap();
        let e = Matrix::from_vec(1, 1, vec![1.]).unwrap();
        let mut workspace = p.workspace(&Request::Condition).unwrap();
        assert_eq!(
            p.frechet_into(
                &e,
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::DomainError
        );
        assert_eq!(out.as_slice(), &[Complex::new(42., 0.)]);
    }
}

#[test]
fn graded_roots_and_singular_recurrences() {
    // Zero root blocks may have a consistent singular local recurrence.
    for n in [2, 33] {
        let zero = Matrix::<f64>::zeros(n, n).unwrap();
        let root = evaluate_complex(&plan(&zero, F::Sqrt, &MatrixFunctionOptions::new()));
        assert!(root.as_slice().iter().all(|z| z.re == 0. && z.im == 0.));
        let complex = Matrix::<Complex<f64>>::zeros(n, n).unwrap();
        let root = evaluate_complex(&plan(&complex, F::Sqrt, &MatrixFunctionOptions::new()));
        assert!(root.as_slice().iter().all(|z| z.re == 0. && z.im == 0.));
    }
    let a = Matrix::from_vec(2, 2, vec![2_f64.powi(900), 0., 0., 2_f64.powi(-900)]).unwrap();
    let root = evaluate_complex(&plan(&a, F::Sqrt, &MatrixFunctionOptions::new()));
    for (index, expected) in [(0, 2_f64.powi(450)), (3, 2_f64.powi(-450))] {
        let value = root.as_slice()[index];
        assert!((value.re / expected - 1.).abs() <= 64. * f64::EPSILON);
        assert_eq!(value.im, 0.);
    }
    let nilpotent = Matrix::from_vec(2, 2, vec![0., 1., 0., 0.]).unwrap();
    let p = plan(&nilpotent, F::Sqrt, &MatrixFunctionOptions::new());
    let mut out = Matrix::from_vec(2, 2, vec![42.; 4]).unwrap();
    assert_eq!(
        p.evaluate_into(
            out.view_mut(),
            &mut p.workspace(&Request::Evaluate).unwrap(),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(out.as_slice(), &[42.; 4]);
}

fn pade_degrees<S: NumericScalar>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    for value in [0.001, 0.1, 0.5, 1., 2., 3., 8., 50., -50.] {
        let real = S::Real::try_from_f64_lossy(value).unwrap();
        let lambda = S::from_components(
            real,
            if S::IS_REAL {
                S::Real::ZERO
            } else {
                real * S::Real::try_from_f64_lossy(0.3).unwrap()
            },
        );
        let a = Matrix::from_vec(2, 2, vec![lambda, S::ONE, S::ZERO, lambda]).unwrap();
        let p = plan(&a, F::Exp, &MatrixFunctionOptions::new());
        let mut out = Matrix::<S>::zeros(2, 2).unwrap();
        p.evaluate_into(
            out.view_mut(),
            &mut p.workspace(&Request::Evaluate).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        let expected = Complex::new(lambda.real_component(), lambda.imaginary_component()).exp();
        for (&value, expected) in
            out.as_slice()
                .iter()
                .zip([expected, expected, Complex::ZERO, expected])
        {
            let actual = Complex::new(value.real_component(), value.imaginary_component());
            assert!(
                (actual - expected).norm()
                    <= S::Real::try_from_f64_lossy(512.).unwrap()
                        * S::Real::EPSILON
                        * expected.norm().max(S::Real::ONE)
            );
        }
    }
}
#[test]
fn adaptive_pade_all_scalar_domains() {
    pade_degrees::<f32>();
    pade_degrees::<f64>();
    pade_degrees::<Complex<f32>>();
    pade_degrees::<Complex<f64>>();
}

#[test]
fn negative_scalar_root_with_a_real_conjugate_pair() {
    let a = Matrix::from_vec(3, 3, vec![0.7, -0.6, 0.2, 0.6, 0.7, -0.1, 0., 0., -1.]).unwrap();
    let root = evaluate_complex(&plan(&a, F::Sqrt, &MatrixFunctionOptions::new()));
    assert!((root.as_slice()[8] - Complex::new(0., 1.)).norm() < 64. * f64::EPSILON);
    for i in 0..3 {
        for j in 0..3 {
            let square = (0..3).fold(Complex::ZERO, |sum, k| {
                sum + root.as_slice()[i * 3 + k] * root.as_slice()[k * 3 + j]
            });
            assert!(
                (square - Complex::new(a.as_slice()[i * 3 + j], 0.)).norm() < 512. * f64::EPSILON
            );
        }
    }
}

#[test]
fn complex_logarithm_clustered_divided_difference() {
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(2., 0.),
            Complex::new(0.3, 0.),
            Complex::ZERO,
            Complex::new(2. + 1e-12, 0.),
        ],
    )
    .unwrap();
    let result = evaluate_complex(&plan(&a, F::Log, &MatrixFunctionOptions::new()));
    let delta = a.as_slice()[3].re - a.as_slice()[0].re;
    let expected = 0.3 * (delta / 2.).ln_1p() / delta;
    assert!((result.as_slice()[1] - Complex::new(expected, 0.)).norm() < 64. * f64::EPSILON);
}

fn small_trigonometric<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let cast = |x| R::try_from_f64_lossy(x).unwrap();
    for scale in [1e-4, 1e-8, 1e-12] {
        let x = cast(scale);
        for (function, real, imaginary) in [
            (F::Sin, x.sin(), x.sinh()),
            (F::Tan, x.tan(), x.tanh()),
            (F::Sinh, x.sinh(), x.sin()),
            (F::Tanh, x.tanh(), x.tan()),
        ] {
            let cases = [
                (
                    Matrix::from_vec(1, 1, vec![Complex::from_real(x)]).unwrap(),
                    vec![Complex::from_real(real)],
                ),
                (
                    Matrix::from_vec(1, 1, vec![Complex::new(R::ZERO, x)]).unwrap(),
                    vec![Complex::new(R::ZERO, imaginary)],
                ),
            ];
            for (a, expected) in cases {
                let p = plan(&a, function, &MatrixFunctionOptions::new());
                let mut out = Matrix::<Complex<R>>::zeros(1, 1).unwrap();
                p.evaluate_into(
                    out.view_mut(),
                    &mut p.workspace(&Request::Evaluate).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
                assert!(
                    (out.as_slice()[0] - expected[0]).norm()
                        <= cast(10.) * R::EPSILON * expected[0].norm(),
                    "{function:?}, {scale}, {:?}",
                    out.as_slice()
                );
            }
            let a = Matrix::from_vec(2, 2, vec![R::ZERO, -x, x, R::ZERO]).unwrap();
            let p = plan(&a, function, &MatrixFunctionOptions::new());
            let mut out = Matrix::<R>::zeros(2, 2).unwrap();
            let mut workspace = p.workspace(&Request::Evaluate).unwrap();
            p.evaluate_into(
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            let error = out
                .as_slice()
                .iter()
                .zip([R::ZERO, -imaginary, imaginary, R::ZERO])
                .fold(R::ZERO, |m, (&v, e)| m.max((v - e).abs()));
            assert!(
                error <= cast(10.) * R::EPSILON * imaginary.abs(),
                "{function:?}, {scale}: {error:?}"
            );
        }
    }
}

#[test]
fn small_trigonometric_relative_accuracy() {
    small_trigonometric::<f32>();
    small_trigonometric::<f64>();
}

#[derive(Clone, Copy)]
enum EntireReference {
    Exp,
    Sin,
    Cosh,
}
impl<R: RealScalar> AnalyticMatrixFunction<R> for EntireReference
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn radius(&self, _: Complex<R>) -> R {
        R::MAX
    }
    fn coefficient(&self, c: Complex<R>, k: usize) -> Result<Complex<R>, SciOxideError> {
        let value = match self {
            Self::Exp => c.exp(),
            Self::Sin => match k % 4 {
                0 => c.sin(),
                1 => c.cos(),
                2 => -c.sin(),
                _ => -c.cos(),
            },
            Self::Cosh => {
                if k.is_multiple_of(2) {
                    c.cosh()
                } else {
                    c.sinh()
                }
            }
        };
        (1..=k).try_fold(value, |v, j| {
            Ok(v / Complex::from_real(R::try_from_usize_exact(j)?))
        })
    }
    fn tail_bound(&self, c: Complex<R>, k: usize, r: R) -> Result<R, SciOxideError> {
        let growth = match self {
            Self::Sin => c.im.abs(),
            _ => c.re.abs(),
        };
        (1..=k + 1).try_fold((growth + r).exp(), |v, j| {
            Ok(v * (r / R::try_from_usize_exact(j)?))
        })
    }
}

#[test]
fn analytic_functions_match_builtins_across_cluster_gaps() {
    let options = MatrixFunctionOptions::new().with_function_settings(
        scioxide_linalg::MatrixFunctionSettings::new()
            .with_tolerance(scioxide_core::Tolerance::new(0., f64::EPSILON).unwrap()),
    );
    for gap in [
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        0.01,
        0.075,
        0.1,
        0.100000000000001,
        1.,
    ] {
        for diagonal in [
            [1., 1. + gap, 1. + 2. * gap],
            [1., 3., 1. + gap],
            [1., 3., 1.],
        ] {
            let a = Matrix::from_vec(
                3,
                3,
                vec![
                    diagonal[0],
                    0.25,
                    -0.5,
                    0.,
                    diagonal[1],
                    0.125,
                    0.,
                    0.,
                    diagonal[2],
                ],
            )
            .unwrap();
            for (f, callback) in [
                (F::Exp, EntireReference::Exp),
                (F::Sin, EntireReference::Sin),
                (F::Cosh, EntireReference::Cosh),
            ] {
                let expected = evaluate_complex(&plan(&a, f, &options));
                let p = plan(&a, F::Analytic(&callback), &options);
                let mut workspace = p.workspace(&Request::Evaluate).unwrap();
                let mut out = Matrix::<Complex<f64>>::zeros(3, 3).unwrap();
                p.evaluate_into(
                    out.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
                let norm: f64 = expected.as_slice().iter().map(|v| v.norm()).sum();
                let error: f64 = out
                    .as_slice()
                    .iter()
                    .zip(expected.as_slice())
                    .map(|(&a, &b)| (a - b).norm())
                    .sum();
                assert!(
                    error <= 100. * f64::EPSILON * norm,
                    "{f:?}, {diagonal:?}: {error:e}, {norm:e}"
                );
            }
        }
    }
}

fn frechet_binary_scales<S: NumericScalar, I: NumericScalar<Real = S::Real>>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let cast = |x| S::Real::try_from_f64_lossy(x).unwrap();
    let scalar =
        |x, y| S::from_components(cast(x), if S::IS_REAL { S::Real::ZERO } else { cast(y) });
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            scalar(2., 0.125),
            scalar(0.25, 0.),
            scalar(0.125, 0.),
            scalar(3., -0.25),
        ],
    )
    .unwrap();
    let direction: Vec<I> = [0.5, -0.125, 0.25, 0.5]
        .into_iter()
        .map(|x| {
            I::from_components(
                cast(x),
                if I::IS_REAL {
                    S::Real::ZERO
                } else {
                    cast(x / 2.)
                },
            )
        })
        .collect();
    let mut e = Matrix::from_vec(2, 2, direction.clone()).unwrap();
    let mut out = Matrix::<Complex<S::Real>>::zeros(2, 2).unwrap();
    let options = MatrixFunctionOptions::new().with_function_settings(
        scioxide_linalg::MatrixFunctionSettings::new().with_tolerance(
            scioxide_core::Tolerance::new(S::Real::ZERO, S::Real::EPSILON).unwrap(),
        ),
    );
    let bound = if S::Real::MANTISSA_DIGITS <= 24 {
        120
    } else {
        1000
    };
    for function in [
        F::Exp,
        F::Log,
        F::Sqrt,
        F::Power(cast(0.3)),
        F::Sin,
        F::Analytic(&EntireReference::Exp),
    ] {
        let p = plan(&a, function, &options);
        let mut workspace = p.workspace(&Request::Frechet).unwrap();
        e.as_mut_slice().copy_from_slice(&direction);
        let baseline = p
            .frechet_into(
                &e,
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        let expected = out.clone();
        for k in -bound..=bound {
            for (v, d) in e.as_mut_slice().iter_mut().zip(&direction) {
                *v = I::from_components(
                    d.real_component().scalbn(k),
                    d.imaginary_component().scalbn(k),
                );
            }
            let report = p
                .frechet_into(
                    &e,
                    out.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.work(), baseline.work(), "{function:?}, exponent {k}");
            for (&v, &expected) in out.as_slice().iter().zip(expected.as_slice()) {
                for (value, expected) in [
                    (v.re.scalbn(-k), expected.re),
                    (v.im.scalbn(-k), expected.im),
                ] {
                    assert!(
                        (value - expected).abs()
                            <= cast(4.)
                                * (expected.abs().next_after(S::Real::MAX) - expected.abs()),
                        "{function:?}, exponent {k}: {value:?} != {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn frechet_power_of_two_homogeneity_and_work() {
    frechet_binary_scales::<f64, f64>();
    frechet_binary_scales::<Complex<f64>, Complex<f64>>();
    frechet_binary_scales::<f64, Complex<f64>>();
    frechet_binary_scales::<f32, f32>();
    frechet_binary_scales::<Complex<f32>, Complex<f32>>();
    frechet_binary_scales::<f32, Complex<f32>>();
}

#[test]
fn frechet_zero_subnormal_and_restoration_overflow() {
    let zero = Matrix::from_vec(1, 1, vec![0.]).unwrap();
    let positive = Matrix::from_vec(1, 1, vec![2.]).unwrap();
    let mut out = Matrix::from_vec(1, 1, vec![42.]).unwrap();
    for function in [
        F::Exp,
        F::Log,
        F::Sqrt,
        F::Power(0.3),
        F::Sin,
        F::Analytic(&EntireReference::Exp),
    ] {
        let p = plan(&positive, function, &MatrixFunctionOptions::new());
        p.frechet_into(
            &zero,
            out.view_mut(),
            &mut p.workspace(&Request::Frechet).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(out.as_slice(), &[0.]);
    }
    for value in [f64::from_bits(1), f64::MIN_POSITIVE, 1., f64::MAX] {
        let e = Matrix::from_vec(1, 1, vec![value]).unwrap();
        for function in [F::Exp, F::Sin, F::Sinh, F::Analytic(&EntireReference::Exp)] {
            let p = plan(&zero, function, &MatrixFunctionOptions::new());
            let report = p
                .frechet_into(
                    &e,
                    out.view_mut(),
                    &mut p.workspace(&Request::Frechet).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(out.as_slice(), &[value]);
            assert_eq!(report.output_norm().round().unwrap(), value);
        }
    }
    let e = Matrix::from_vec(1, 1, vec![f64::MAX]).unwrap();
    let p = plan(&positive, F::Exp, &MatrixFunctionOptions::new());
    out.as_mut_slice().fill(42.);
    assert_eq!(
        p.frechet_into(
            &e,
            out.view_mut(),
            &mut p.workspace(&Request::Frechet).unwrap(),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(out.as_slice(), &[42.]);
}

#[test]
fn analytic_reordering_uses_block_mean_and_preserves_failed_output() {
    struct DiskExp;
    impl AnalyticMatrixFunction<f64> for DiskExp {
        fn radius(&self, _: Complex<f64>) -> f64 {
            0.06
        }
        fn coefficient(&self, c: Complex<f64>, k: usize) -> Result<Complex<f64>, SciOxideError> {
            EntireReference::Exp.coefficient(c, k)
        }
        fn tail_bound(&self, c: Complex<f64>, k: usize, r: f64) -> Result<f64, SciOxideError> {
            EntireReference::Exp.tail_bound(c, k, r)
        }
    }
    // Reordering excludes the middle eigenvalue; centering at the mean fits
    // the callback disk, whereas the first eigenvalue as center does not.
    let a = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., 3., 0., 0., 0., 1.09]).unwrap();
    let options = MatrixFunctionOptions::new().with_function_settings(
        scioxide_linalg::MatrixFunctionSettings::new()
            .with_tolerance(scioxide_core::Tolerance::new(0., f64::EPSILON).unwrap()),
    );
    let p = plan(&a, F::Analytic(&DiskExp), &options);
    let mut workspace = p.workspace(&Request::Evaluate).unwrap();
    let mut out = Matrix::<f64>::zeros(3, 3).unwrap();
    let report = p
        .evaluate_into(
            out.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    for i in 0..3 {
        assert!(
            (out.as_slice()[4 * i] - a.as_slice()[4 * i].exp()).abs()
                <= 10. * f64::EPSILON * out.as_slice()[4 * i]
        );
    }
    for options in [
        options.with_execution_options(
            (options)
                .execution_options()
                .with_iteration_budget(Budget::try_from(report.work() - 1).unwrap()),
        ),
        options
            .with_function_settings((options).function_settings().with_series_limit(Budget::ONE)),
    ] {
        let p = plan(&a, F::Analytic(&DiskExp), &options);
        out.as_mut_slice().fill(42.);
        assert_eq!(
            p.evaluate_into(
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(out.as_slice(), &[42.; 9]);
    }
    // A genuinely inadmissible block still fails the callback domain check.
    let a = Matrix::from_vec(2, 2, vec![1., 1., 0., 1.09]).unwrap();
    let p = plan(&a, F::Analytic(&DiskExp), &options);
    let mut out = Matrix::from_vec(2, 2, vec![42.; 4]).unwrap();
    assert_eq!(
        p.evaluate_into(
            out.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(out.as_slice(), &[42.; 4]);
}

#[test]
fn representable_results_report_norms_beyond_the_scalar_range() {
    // L_exp(0, E) = E: finite entries whose entrywise one-norm is 4 MAX.
    let zero = Matrix::from_vec(2, 2, vec![0.; 4]).unwrap();
    let e = Matrix::from_vec(2, 2, vec![f64::MAX; 4]).unwrap();
    let mut out = Matrix::from_vec(2, 2, vec![0.; 4]).unwrap();
    let p = plan(&zero, F::Exp, &MatrixFunctionOptions::new());
    let report = p
        .frechet_into(
            &e,
            out.view_mut(),
            &mut p.workspace(&Request::Frechet).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out.as_slice(), &[f64::MAX; 4]);
    let norm = report.output_norm();
    assert_eq!(norm.shifted(-2).unwrap().round().unwrap(), f64::MAX);
    assert_eq!(
        norm.round().unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn finite_complex_components_report_magnitudes_beyond_the_scalar_range() {
    // L_exp(0, E) = E even when a single complex modulus exceeds MAX.
    let zero = Matrix::from_vec(1, 1, vec![Complex::new(0.0, 0.0)]).unwrap();
    let value = Complex::new(f64::MAX, -f64::MAX);
    let e = Matrix::from_vec(1, 1, vec![value]).unwrap();
    let mut out = Matrix::from_vec(1, 1, vec![Complex::new(0.0, 0.0)]).unwrap();
    let p = plan(&zero, F::Exp, &MatrixFunctionOptions::new());
    let report = p
        .frechet_into(
            &e,
            out.view_mut(),
            &mut p.workspace(&Request::Frechet).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out.as_slice(), &[value]);
    let half_norm = report.output_norm().shifted(-1).unwrap().round().unwrap();
    assert!((half_norm / f64::MAX - core::f64::consts::FRAC_1_SQRT_2).abs() <= 4.0 * f64::EPSILON);
}

#[test]
fn entire_functions_converge_on_nonnormal_clustered_blocks() {
    // The eigenvalues 1 and 1.01 form one cluster; the coupling `c` enters
    // the remainder bound only to powers below the block order.
    for c in [100., 1000.] {
        let a = Matrix::from_vec(2, 2, vec![1., c, 0., 1.01]).unwrap();
        for (function, f) in [(F::Sin, f64::sin as fn(f64) -> f64), (F::Cosh, f64::cosh)] {
            let value = scioxide_linalg::matrix_function(&a, function).unwrap();
            let divided = c * (f(1.01) - f(1.)) / 0.01;
            let expected = [f(1.), divided, 0., f(1.01)];
            for (&x, e) in value.as_slice().iter().zip(expected) {
                assert!((x - e).abs() <= 1e-12 * e.abs().max(1.), "{c}: {x} vs {e}");
            }
        }
    }
    // Eigenvalues 1, 1 + 1e-14, 1 + 2e-14 with coupling 40 differ from the
    // nilpotent case `sin(I + N) = sin 1 I + cos 1 N − sin 1 N²/2` by O(1e-11).
    let mut a = Matrix::from_vec(3, 3, vec![1., 40., 40., 0., 1., 40., 0., 0., 1.]).unwrap();
    a.view_mut().set(1, 1, 1. + 1e-14).unwrap();
    a.view_mut().set(2, 2, 1. + 2e-14).unwrap();
    let value = scioxide_linalg::matrix_function(&a, F::Sin).unwrap();
    let (s, c) = (1f64.sin(), 1f64.cos());
    let expected = [s, 40. * c, 40. * c - 800. * s, 0., s, 40. * c, 0., 0., s];
    for (&x, e) in value.as_slice().iter().zip(expected) {
        assert!((x - e).abs() <= 1e-10 * e.abs().max(1.), "{x} vs {e}");
    }
    // The Fréchet derivative of the 2 x 2 upper triangular case in an upper
    // triangular direction follows from `f([[a, b], [0, d]])_{01} = b f[a, d]`.
    let a = Matrix::from_vec(2, 2, vec![1., 40., 0., 1.01]).unwrap();
    let p = plan(&a, F::Sin, &MatrixFunctionOptions::new());
    let mut workspace = p.workspace(&Request::Condition).unwrap();
    let direction = Matrix::from_vec(2, 2, vec![1., 1., 0., 0.]).unwrap();
    let mut derivative = Matrix::<f64>::zeros(2, 2).unwrap();
    p.frechet_into(
        &direction,
        derivative.view_mut(),
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let h = -0.01;
    let divided = (1.01f64.sin() - s) / 0.01;
    let slope = (c * h - (s - 1.01f64.sin())) / (h * h);
    let expected = [c, divided + 40. * slope, 0., 0.];
    for (&x, e) in derivative.as_slice().iter().zip(expected) {
        assert!((x - e).abs() <= 1e-9 * e.abs().max(1.), "{x} vs {e}");
    }
}

#[test]
fn principal_square_root_ignores_the_sign_of_a_zero_imaginary_part() {
    let z = |re: f64, im: f64| Complex::new(re, im);
    let a = Matrix::from_vec(2, 2, vec![z(-1., 0.), z(0., 0.), z(0., 0.), z(-4., 0.)]).unwrap();
    // The adjoint view negates the zero imaginary parts.
    for root in [
        scioxide_linalg::matrix_function(&a, F::Sqrt).unwrap(),
        scioxide_linalg::matrix_function(a.view().adjoint(), F::Sqrt).unwrap(),
    ] {
        for (&x, e) in root
            .as_slice()
            .iter()
            .zip([z(0., 1.), z(0., 0.), z(0., 0.), z(0., 2.)])
        {
            assert!((x - e).norm() <= 4. * f64::EPSILON, "{x:?} vs {e:?}");
        }
    }
    // Both diagonal entries map to i, so the recurrence divides by 2i.
    let a = Matrix::from_vec(2, 2, vec![z(-1., -0.), z(1., 0.), z(0., 0.), z(-1., 0.)]).unwrap();
    let root = scioxide_linalg::matrix_function(&a, F::Sqrt).unwrap();
    for (&x, e) in root
        .as_slice()
        .iter()
        .zip([z(0., 1.), z(0., -0.5), z(0., 0.), z(0., 1.)])
    {
        assert!((x - e).norm() <= 4. * f64::EPSILON, "{x:?} vs {e:?}");
    }
}

#[test]
fn single_precision_exponential_keeps_tenth_powers_in_range() {
    // exp(−1024 · ones(16)) = I − ones/16 up to e^(−16384). Every entry of A is
    // far from overflow, but ‖A‖₁¹⁰ = 2¹⁴⁰ exceeds the f32 range. The relative
    // condition number of exp at this normal A is ‖A‖₂ = 2¹⁴.
    let n = 16;
    let a = Matrix::from_vec(n, n, vec![-1024f32; n * n]).unwrap();
    let value = scioxide_linalg::matrix_function(&a, F::Exp).unwrap();
    let tolerance = 4. * 16384. * f32::EPSILON;
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1. } else { 0. } - 1. / 16.;
            let x = value.view().get(i, j).unwrap();
            assert!((x - expected).abs() <= tolerance, "{x} vs {expected}");
        }
    }
}

#[test]
fn integer_powers_beyond_the_i64_range_square_the_reduced_power() {
    let identity = Matrix::<f64>::identity(3).unwrap();
    for p in [1e19, -1e19, 1e300] {
        let value = scioxide_linalg::matrix_function(&identity, F::Power(p)).unwrap();
        assert_eq!(value.as_slice(), identity.as_slice());
    }
    // [[1, x], [0, 1]]^p = [[1, p x], [0, 1]], and (−I)^p = I for even p.
    let a = Matrix::from_vec(2, 2, vec![1., 1e-20, 0., 1.]).unwrap();
    let value = scioxide_linalg::matrix_function(&a, F::Power(1e19)).unwrap();
    for (&x, e) in value.as_slice().iter().zip([1., 0.1, 0., 1.]) {
        assert!((x - e).abs() <= 1e-12, "{x} vs {e}");
    }
    let a = Matrix::from_vec(2, 2, vec![-1., 0., 0., -1.]).unwrap();
    let value = scioxide_linalg::matrix_function(&a, F::Power(-1e19)).unwrap();
    assert_eq!(value.as_slice(), &[1., 0., 0., 1.]);
}
