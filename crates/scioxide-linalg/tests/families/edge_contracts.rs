//! Non-finite, empty and extreme-scale public linear algebra contracts.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::backend::RealBackend;
use scioxide_core::{
    AlgebraicScalar, AttemptOutcome, Budget, Complex, ExecutionContext, NumericScalar, RealScalar,
    ResidualTolerance, SciOxideErrorKind,
};
use scioxide_linalg::*;

fn real<S: NumericScalar>(x: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(x).unwrap())
}
fn rejected<S: NumericScalar>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let a = Matrix::from_vec(2, 2, vec![real::<S>(2.), S::ZERO, S::ZERO, real(3.)]).unwrap();
    let general = GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
    let hermitian = HermitianEigenOptions::default();
    let mut gv = scioxide_linalg::GeneralEigenvaluesResult::compute_with_options(&a, &general)
        .unwrap()
        .into_parts()
        .0;
    let mut ge = GeneralEigenDecomposition::decompose_with_options(&a, &general).unwrap();
    let mut hv = scioxide_linalg::HermitianEigenvaluesResult::compute(
        &a,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap()
    .into_parts()
    .0;
    let mut he =
        HermitianEigenDecomposition::decompose(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    let mut schur = SchurDecomposition::decompose(&a).unwrap();
    let mut qz = GeneralizedSchurDecomposition::decompose(&a, &a).unwrap();
    let mut complex_workspace = LinalgWorkspace::new();
    let mut workspace = LinalgWorkspace::new();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (imaginary, index) in [false, true]
            .into_iter()
            .flat_map(|imaginary| (0..4).map(move |index| (imaginary, index)))
        {
            if imaginary && S::IS_REAL {
                continue;
            }
            let mut invalid = a.clone();
            let component = if bad.is_nan() {
                S::Real::NAN
            } else if bad.is_sign_positive() {
                S::Real::INFINITY
            } else {
                S::Real::NEG_INFINITY
            };
            invalid.as_mut_slice()[index] = if imaginary {
                S::from_components(S::Real::ONE, component)
            } else {
                S::from_real(component)
            };
            let mut state = scioxide_core::SolveState::new();
            let mut budgets = scioxide_core::SolveBudgets::unbounded();
            let mut parent = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    scioxide_core::SolveIdentity::new(
                        scioxide_core::ProblemToken::new(1106),
                        scioxide_core::algorithm_id!("test", "nonfinite_dense", 1),
                        0,
                        0,
                    ),
                    &mut budgets,
                )
                .unwrap();
            macro_rules! rejects {
                ($out:expr,$work:expr,$call:expr) => {{
                    let before = format!("{:?}", $out);
                    let counts_before = parent.counts();
                    assert_eq!($call.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
                    assert_eq!(parent.counts(), counts_before);
                    assert_eq!(format!("{:?}", $out), before);
                    let attempt = $work.last_attempt().unwrap();
                    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
                    assert!(attempt.epoch().is_none());
                }};
            }
            rejects!(
                gv,
                complex_workspace,
                GeneralEigenvaluesBuffer::compute_into(
                    &invalid,
                    &general,
                    &mut gv,
                    &mut complex_workspace,
                    &mut parent
                )
            );
            rejects!(
                ge,
                complex_workspace,
                GeneralEigenDecomposition::decompose_into(
                    &invalid,
                    &general,
                    &mut ge,
                    &mut complex_workspace,
                    &mut parent
                )
            );
            rejects!(
                hv,
                workspace,
                HermitianEigenvaluesBuffer::compute_into(
                    &invalid,
                    scioxide_linalg::HermitianInput::Exact,
                    &hermitian,
                    &mut hv,
                    &mut workspace,
                    &mut parent
                )
            );
            rejects!(
                he,
                workspace,
                HermitianEigenDecomposition::decompose_into(
                    &invalid,
                    scioxide_linalg::HermitianInput::Exact,
                    &hermitian,
                    &mut he,
                    &mut workspace,
                    &mut parent
                )
            );
            rejects!(
                schur,
                workspace,
                SchurDecomposition::decompose_into(
                    &invalid,
                    &SchurOptions::new(),
                    &mut schur,
                    &mut workspace,
                    &mut parent
                )
            );
            for (left, right) in [(&invalid, &a), (&a, &invalid)] {
                rejects!(
                    qz,
                    workspace,
                    GeneralizedSchurDecomposition::decompose_into(
                        left,
                        right,
                        &SchurOptions::new(),
                        &mut qz,
                        &mut workspace,
                        &mut parent
                    )
                );
            }
        }
    }
    // A rejected call must not poison subsequent use of the same workspace.
    GeneralEigenDecomposition::decompose_into(
        &a,
        &general,
        &mut ge,
        &mut complex_workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(
        complex_workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    HermitianEigenDecomposition::decompose_into(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        &hermitian,
        &mut he,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
}
#[test]
fn nonfinite_real32() {
    rejected::<f32>();
}
#[test]
fn nonfinite_real64() {
    rejected::<f64>();
}
#[test]
fn nonfinite_complex32() {
    rejected::<Complex<f32>>();
}
#[test]
fn nonfinite_complex64() {
    rejected::<Complex<f64>>();
}

fn empty<S: NumericScalar>() {
    let a = Matrix::<S>::zeros(0, 0).unwrap();
    let mut schur = SchurDecomposition::decompose(&a).unwrap();
    let mut pencil = GeneralizedSchurDecomposition::decompose(&a, &a).unwrap();
    assert!(pencil.eigenvalues().is_empty());
    for factor in [pencil.s(), pencil.t(), pencil.q(), pencil.z()] {
        assert_eq!((factor.rows(), factor.cols()), (0, 0));
    }
    for estimates in [Estimation::Skip, Estimation::Estimate] {
        let options = SchurReorderOptions::new().with_condition_estimates(estimates);
        let (_, report) = schur.reorder_with_options(&[], &options).unwrap();
        assert_eq!(
            (
                report.selected_dimension(),
                report.swaps(),
                report.iterations()
            ),
            (0, 0, 0)
        );
        schur
            .reorder_in_place(
                &[],
                &options,
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        let options = SchurReorderOptions::new().with_condition_estimates(estimates);
        let (_, report) = pencil.reorder_with_options(&[], &options).unwrap();
        assert_eq!(
            (
                report.selected_dimension(),
                report.swaps(),
                report.iterations()
            ),
            (0, 0, 0)
        );
        pencil
            .reorder_in_place(
                &[],
                &options,
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
    }
}
#[test]
fn empty_real32() {
    empty::<f32>();
}
#[test]
fn empty_real64() {
    empty::<f64>();
}
#[test]
fn empty_complex32() {
    empty::<Complex<f32>>();
}
#[test]
fn empty_complex64() {
    empty::<Complex<f64>>();
}

fn riccati_nonfinite<S: NumericScalar>() {
    for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
        for coefficient in 0..6 {
            for bad in [S::Real::NAN, S::Real::INFINITY, S::Real::NEG_INFINITY] {
                for imaginary in [false, true] {
                    if imaginary && S::IS_REAL {
                        continue;
                    }
                    let mut matrices = [real::<S>(-1.), S::ONE, S::ONE, S::ONE, S::ONE, S::ZERO]
                        .map(|x| Matrix::from_vec(1, 1, vec![x]).unwrap());
                    matrices[coefficient].as_mut_slice()[0] = if imaginary {
                        S::from_components(S::Real::ONE, bad)
                    } else {
                        S::from_real(bad)
                    };
                    let [a, b, q, r, e, s] = &matrices;
                    let problem = RiccatiProblem {
                        a: a.view(),
                        b: b.view(),
                        q: q.view(),
                        r: r.view(),
                        e: Some(e.view()),
                        s: Some(s.view()),
                    };
                    assert_eq!(
                        RiccatiPlan::new(problem, kind, RiccatiOptions::new())
                            .unwrap_err()
                            .kind(),
                        SciOxideErrorKind::NonFinite
                    );
                }
            }
        }
    }
}
#[test]
fn riccati_nonfinite_real32() {
    riccati_nonfinite::<f32>();
}
#[test]
fn riccati_nonfinite_real64() {
    riccati_nonfinite::<f64>();
}
#[test]
fn riccati_nonfinite_complex32() {
    riccati_nonfinite::<Complex<f32>>();
}
#[test]
fn riccati_nonfinite_complex64() {
    riccati_nonfinite::<Complex<f64>>();
}

fn reconstruction<S: NumericScalar>(
    a: &Matrix<S>,
    q: &Matrix<S>,
    t: &Matrix<S>,
    z: &Matrix<S>,
    scale: S,
    epsilon: S::Real,
) {
    let n = a.rows();
    for i in 0..n {
        for j in 0..n {
            let mut value = S::ZERO;
            for k in 0..n {
                for l in 0..n {
                    value = value
                        + q.as_slice()[i * n + k]
                            * (t.as_slice()[k * n + l] / scale)
                            * z.as_slice()[j * n + l].conj();
                }
            }
            assert!((value - a.as_slice()[i * n + j] / scale).abs() <= epsilon);
            for factor in [q, z] {
                let dot = (0..n).fold(S::ZERO, |sum, k| {
                    sum + factor.as_slice()[k * n + i].conj() * factor.as_slice()[k * n + j]
                });
                assert!((dot - if i == j { S::ONE } else { S::ZERO }).abs() <= epsilon);
            }
        }
    }
}
fn extreme_eigen<S: NumericScalar>(scales: &[f64])
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let epsilon = S::Real::try_from_f64_lossy(512.).unwrap() * <S::Real as RealScalar>::EPSILON;
    for &scale in scales {
        let a = Matrix::from_vec(
            2,
            2,
            vec![
                real::<S>(scale),
                real(scale / 4.),
                S::ZERO,
                real(2. * scale),
            ],
        )
        .unwrap();
        let options =
            GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
        let values =
            scioxide_linalg::GeneralEigenvaluesResult::compute_with_options(&a, &options).unwrap();
        let vectors = GeneralEigenDecomposition::decompose_with_options(&a, &options).unwrap();
        for values in [values.eigenvalues(), vectors.eigenvalues()] {
            for (&value, expected) in values.iter().zip([1., 2.]) {
                assert!(
                    (value / real::<Complex<S::Real>>(scale) - real(expected)).abs() <= epsilon
                );
            }
        }
        for (left, basis) in [
            (false, vectors.right_eigenvectors().unwrap()),
            (true, vectors.left_eigenvectors().unwrap()),
        ] {
            for column in 0..2 {
                for i in 0..2 {
                    let mut image = Complex::<S::Real>::ZERO;
                    for j in 0..2 {
                        let value = if left {
                            a.as_slice()[j * 2 + i].conj()
                        } else {
                            a.as_slice()[i * 2 + j]
                        } / real::<S>(scale);
                        image += Complex::new(value.real_component(), value.imaginary_component())
                            * basis.as_slice()[j * 2 + column];
                    }
                    let lambda = vectors.eigenvalues()[column] / real::<Complex<S::Real>>(scale);
                    let expected = (if left { lambda.conj() } else { lambda })
                        * basis.as_slice()[i * 2 + column];
                    assert!((image - expected).abs() <= epsilon);
                }
            }
        }
        let factor = SchurDecomposition::decompose(&a).unwrap();
        let (mut reordered, report) = factor.reorder(&[false, true]).unwrap();
        assert_eq!(report.selected_dimension(), 1);
        reconstruction(
            &a,
            reordered.q(),
            reordered.t(),
            reordered.q(),
            real(scale),
            epsilon,
        );
        assert!(
            (reordered.eigenvalues()[0] / real::<Complex<S::Real>>(scale) - real(2.)).abs()
                <= epsilon
        );
        reordered
            .reorder_in_place(
                &[false, true],
                &SchurReorderOptions::new(),
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        reconstruction(
            &a,
            reordered.q(),
            reordered.t(),
            reordered.q(),
            real(scale),
            epsilon,
        );
        let b =
            Matrix::from_vec(2, 2, vec![real::<S>(scale), S::ZERO, S::ZERO, real(scale)]).unwrap();
        let factor = GeneralizedSchurDecomposition::decompose(&a, &b).unwrap();
        let (mut reordered, report) = factor.reorder(&[false, true]).unwrap();
        assert_eq!(report.selected_dimension(), 1);
        reconstruction(
            &a,
            reordered.q(),
            reordered.s(),
            reordered.z(),
            real(scale),
            epsilon,
        );
        reconstruction(
            &b,
            reordered.q(),
            reordered.t(),
            reordered.z(),
            real(scale),
            epsilon,
        );
        assert!(
            (reordered.eigenvalues()[0].ratio().unwrap() - real::<Complex<S::Real>>(2.)).abs()
                <= epsilon
        );
        reordered
            .reorder_in_place(
                &[false, true],
                &SchurReorderOptions::new(),
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        reconstruction(
            &a,
            reordered.q(),
            reordered.s(),
            reordered.z(),
            real(scale),
            epsilon,
        );
        reconstruction(
            &b,
            reordered.q(),
            reordered.t(),
            reordered.z(),
            real(scale),
            epsilon,
        );
    }
}
#[test]
fn extreme_eigen_real64() {
    extreme_eigen::<f64>(&[1e-300, 1e300]);
}
#[test]
fn extreme_eigen_complex64() {
    extreme_eigen::<Complex<f64>>(&[1e-300, 1e300]);
}
#[test]
fn extreme_eigen_real32() {
    extreme_eigen::<f32>(&[1e-30, 1e30]);
}
#[test]
fn extreme_eigen_complex32() {
    extreme_eigen::<Complex<f32>>(&[1e-30, 1e30]);
}

fn krylov_scales<R: RealScalar + NumericScalar<Real = R>>(scales: &[f64]) {
    let c = |x| R::try_from_f64_lossy(x).unwrap();
    let tolerance = ResidualTolerance::new(R::ZERO, c(512.) * <R as RealScalar>::EPSILON).unwrap();
    let budget = Budget::try_from(30).unwrap();
    for (operator, rhs) in [
        (1., scales[0]),
        (1., scales[1]),
        (scales[0], 1.),
        (scales[1], 1.),
    ] {
        let a =
            Matrix::from_vec(2, 2, vec![c(operator), R::ZERO, R::ZERO, c(2. * operator)]).unwrap();
        let a = a.view().try_finite().unwrap();
        let b = Vector::from_vec(vec![c(rhs), c(-rhs)]);
        macro_rules! check {
            ($solver:expr $(, $inverse:expr)?) => {{
                let result = $solver
                    .solve_with_options(&a, &b, $($inverse,)? None)
                    .unwrap_or_else(|e| {
                        panic!("{} a={operator} b={rhs}: {e:?}", stringify!($solver))
                    });
                assert!(
                    result.is_success(),
                    "{} {} {:?}",
                    operator,
                    rhs,
                    result.report()
                );
                let residual = result.report().true_residual().unwrap();
                assert!(residual.raw_norm().is_finite());
                assert!(
                    residual.scaled_norm() <= c(512.) * <R as RealScalar>::EPSILON,
                    "a={operator} b={rhs} {:?}",
                    result.report()
                );
                for (&x, expected) in result
                    .solution()
                    .as_slice()
                    .iter()
                    .zip([c(rhs / operator), c(-rhs / (2. * operator))])
                {
                    assert!(
                        (x / expected - R::ONE).abs() <= c(1024.) * <R as RealScalar>::EPSILON,
                        "{x:?} != {expected:?}"
                    );
                }
            }};
        }
        check!(CgSolver::new(tolerance, budget), Preconditioning::None);
        check!(
            BicgstabSolver::new(tolerance, budget),
            Preconditioning::None
        );
        check!(MinresSolver::new(tolerance, budget), Preconditioning::None);
        check!(LsqrSolver::new(tolerance, budget));
        check!(LsmrSolver::new(tolerance, budget));
    }
}

#[test]
fn krylov_extreme_real64() {
    krylov_scales::<f64>(&[1e300, 1e-300]);
}
#[test]
fn krylov_extreme_real32() {
    krylov_scales::<f32>(&[1e30, 1e-30]);
}

fn complex_cg<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let c = |re, im| {
        Complex::new(
            R::try_from_f64_lossy(re).unwrap(),
            R::try_from_f64_lossy(im).unwrap(),
        )
    };
    let a = Matrix::from_vec(2, 2, vec![c(4., 0.), c(1., 1.), c(1., -1.), c(3., 0.)]).unwrap();
    let b = Vector::from_vec(vec![c(3., 7.), c(0., 1.)]);
    let tolerance =
        ResidualTolerance::new(R::ZERO, R::try_from_f64_lossy(128.).unwrap() * R::EPSILON).unwrap();
    let result = CgSolver::new(tolerance, Budget::try_from(10).unwrap())
        .solve_with_options(
            &a.view().try_finite().unwrap(),
            &b,
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert!(result.is_success());
    for (&x, expected) in result
        .solution()
        .as_slice()
        .iter()
        .zip([c(1., 2.), c(-1., 0.)])
    {
        assert!((x - expected).abs() <= R::try_from_f64_lossy(128.).unwrap() * R::EPSILON);
    }
}
#[test]
fn cg_complex32() {
    complex_cg::<f32>();
}
#[test]
fn cg_complex64() {
    complex_cg::<f64>();
}

fn scaled_hermitian_jobs<S: NumericScalar>()
where
    S::Real: NumericScalar<Real = S::Real>,
{
    let cast = |x| S::Real::try_from_f64_lossy(x).unwrap();
    for n in [21, 40, 65] {
        let diagonal: Vec<_> = (0..n)
            .map(|i| cast((i as f64 - (n / 2) as f64).abs()))
            .collect();
        let off = vec![S::Real::ONE; n - 1];
        let phase = |i: usize| {
            if S::IS_REAL {
                S::ONE
            } else {
                match i % 4 {
                    0 => S::ONE,
                    1 => S::from_components(S::Real::ZERO, S::Real::ONE),
                    2 => -S::ONE,
                    _ => S::from_components(S::Real::ZERO, -S::Real::ONE),
                }
            }
        };
        let base = Matrix::from_vec(
            n,
            n,
            (0..n * n)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    if n != 65 {
                        let x = match r.abs_diff(c) {
                            0 => diagonal[r],
                            1 => S::Real::ONE,
                            _ => S::Real::ZERO,
                        };
                        phase(r) * S::from_real(x) * phase(c).conj()
                    } else {
                        // Fixed seeded symmetric real part and skew-symmetric imaginary part.
                        let mut state = (r.min(c) * n + r.max(c)) as u64 + 1189;
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let re = S::Real::from_u64_half_open01(state) - cast(0.5);
                        let im = if S::IS_REAL || r == c {
                            S::Real::ZERO
                        } else {
                            let x =
                                S::Real::from_u64_half_open01(state.rotate_left(29)) - cast(0.5);
                            if r < c { x } else { -x }
                        };
                        S::from_components(re, im)
                    }
                })
                .collect(),
        )
        .unwrap();
        let reference = scioxide_linalg::HermitianEigenvaluesResult::compute(
            &base,
            scioxide_linalg::HermitianInput::Exact,
        )
        .unwrap();
        let norm = base
            .as_slice()
            .iter()
            .fold(S::Real::ZERO, |m, x| m.max(x.abs()));
        let bound = cast(64.0 * n as f64) * <S::Real as RealScalar>::EPSILON;
        for power in -12..=12 {
            let scale = cast(10.0_f64.powi(power));
            let a = Matrix::from_vec(
                n,
                n,
                base.as_slice()
                    .iter()
                    .map(|&x| x * S::from_real(scale))
                    .collect(),
            )
            .unwrap();
            let values = scioxide_linalg::HermitianEigenvaluesResult::compute(
                &a,
                scioxide_linalg::HermitianInput::Exact,
            )
            .unwrap();
            let vectors =
                HermitianEigenDecomposition::decompose(&a, scioxide_linalg::HermitianInput::Exact)
                    .unwrap();
            for spectrum in [values.eigenvalues(), vectors.eigenvalues()] {
                for (&value, &expected) in spectrum.iter().zip(reference.eigenvalues()) {
                    assert!(
                        (value / scale - expected).abs() <= bound * norm,
                        "n={n} scale={scale:?} eigenvalue"
                    );
                }
            }
            assert_hermitian_bounds(
                &a,
                vectors.eigenvalues(),
                vectors.eigenvectors(),
                scale,
                bound * norm,
                bound,
            );
            let d: Vec<_> = diagonal.iter().map(|&x| x * scale).collect();
            let e: Vec<_> = off.iter().map(|&x| x * scale).collect();
            let tri = HermitianEigenDecomposition::tridiagonal(&d, &e).unwrap();
            let matrix = Matrix::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|i| match (i / n).abs_diff(i % n) {
                        0 => d[i / n],
                        1 => scale,
                        _ => S::Real::ZERO,
                    })
                    .collect(),
            )
            .unwrap();
            assert_hermitian_bounds(
                &matrix,
                tri.eigenvalues(),
                tri.eigenvectors(),
                scale,
                bound * diagonal.iter().copied().fold(S::Real::ONE, S::Real::max),
                bound,
            );
        }
    }
}

fn assert_hermitian_bounds<S: NumericScalar>(
    a: &Matrix<S>,
    values: &[S::Real],
    vectors: &Matrix<S>,
    scale: S::Real,
    residual_bound: S::Real,
    orthogonality_bound: S::Real,
) {
    let n = a.rows();
    for (col, &lambda) in values.iter().enumerate() {
        for row in 0..n {
            let product = (0..n).fold(S::ZERO, |sum, k| {
                sum + (a.as_slice()[row * n + k] / S::from_real(scale))
                    * vectors.as_slice()[k * n + col]
            });
            let residual =
                product - vectors.as_slice()[row * n + col] * S::from_real(lambda / scale);
            assert!(
                residual.abs() <= residual_bound,
                "n={n} scale={scale:?} residual={:?} bound={residual_bound:?}",
                residual.abs()
            );
            let dot = (0..n).fold(S::ZERO, |sum, k| {
                sum + vectors.as_slice()[k * n + row].conj() * vectors.as_slice()[k * n + col]
            });
            let identity = if row == col { S::ONE } else { S::ZERO };
            assert!(
                (dot - identity).abs() <= orthogonality_bound,
                "n={n} scale={scale:?} orthogonality"
            );
        }
    }
}

#[test]
fn hermitian_deflation_is_scale_invariant() {
    scaled_hermitian_jobs::<f64>();
    scaled_hermitian_jobs::<f32>();
    scaled_hermitian_jobs::<Complex<f64>>();
    scaled_hermitian_jobs::<Complex<f32>>();
}

fn small_hermitian_coupling<R: RealScalar + NumericScalar<Real = R>>(diagonal: R, off: R) {
    let a = Matrix::from_vec(2, 2, vec![diagonal, off, off, diagonal]).unwrap();
    let values = scioxide_linalg::HermitianEigenvaluesResult::compute(
        &a,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let bound = R::try_from_f64_exact(8.0).unwrap() * <R as RealScalar>::EPSILON;
    for result in [
        HermitianEigenDecomposition::decompose(&a, scioxide_linalg::HermitianInput::Exact).unwrap(),
        HermitianEigenDecomposition::tridiagonal(&[diagonal; 2], &[off]).unwrap(),
    ] {
        for spectrum in [result.eigenvalues(), values.eigenvalues()] {
            for (&actual, expected) in spectrum.iter().zip([diagonal - off, diagonal + off]) {
                assert!((actual - expected).abs() <= bound * diagonal);
            }
        }
        for &entry in result.eigenvectors().as_slice() {
            assert!((entry.abs() - R::HALF.sqrt()).abs() <= bound);
        }
    }
}

#[test]
fn small_hermitian_coupling_retains_the_rotated_eigenbasis() {
    small_hermitian_coupling(2e-8_f64, 1e-16);
    small_hermitian_coupling(2e-3_f32, 1e-8);
}

#[test]
fn default_normal_stopping_does_not_accept_small_consistent_data() {
    let tolerance = ResidualTolerance::new(0., 1e-12).unwrap();
    let budget = Budget::try_from(200).unwrap();
    for scale in [1., 1e-5] {
        let mut values = vec![0.; 16 * 16];
        for i in 0..16 {
            values[i * 16 + i] = 2. * scale;
            if i > 0 {
                values[i * 16 + i - 1] = -scale;
            }
            if i < 15 {
                values[i * 16 + i + 1] = -scale;
            }
        }
        let matrix = Matrix::from_vec(16, 16, values).unwrap();
        let a = matrix.view().try_finite().unwrap();
        for magnitude in [1., 1e-6, 1e-10, 1e-300] {
            let b = Vector::from_vec(vec![magnitude / 4.; 16]);
            for result in [
                MinresSolver::new(tolerance, budget)
                    .solve_with_options(&a, &b, Preconditioning::None, None)
                    .unwrap(),
                LsqrSolver::new(tolerance, budget)
                    .solve_with_options(&a, &b, None)
                    .unwrap(),
                LsmrSolver::new(tolerance, budget)
                    .solve_with_options(&a, &b, None)
                    .unwrap(),
            ] {
                assert!(
                    result.report().iterations() > 0,
                    "{} {magnitude}: {:?}",
                    scale,
                    result.report()
                );
                if result.is_success() {
                    assert!(
                        result.report().true_residual().unwrap().scaled_norm() <= 1e-12,
                        "{:?}",
                        result.report()
                    );
                }
            }
        }
    }
}
