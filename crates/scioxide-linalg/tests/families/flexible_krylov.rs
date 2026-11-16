//! Independent numerical tests for scalar and coupled block Arnoldi families.
#![allow(clippy::unwrap_used)]
use core::cell::Cell;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    BlockVectorView, Budget, Complex, NumericScalar, Preconditioner, RealScalar, ResidualTolerance,
    SciOxideError,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BlockGcrotSolver, BlockGmresSolver, BlockLgmresSolver, GcrotSolver, GmresSolver, LgmresSolver,
    LinearSolveTermination, Matrix, Vector,
};

fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn real<S: NumericScalar>(x: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(x).unwrap())
}
fn fixture<S: NumericScalar>() -> (Matrix<S>, Vector<S>, Vec<S>) {
    let a = vec![
        real(4.),
        real(1.),
        real(-2.),
        real(0.),
        real(3.),
        real(1.),
        real(1.),
        real(0.),
        real(2.),
    ];
    let a: Vec<S> = a
        .into_iter()
        .enumerate()
        .map(|(i, value): (usize, S)| {
            S::from_components(
                value.real_component(),
                S::Real::try_from_f64_lossy(if i % 4 == 0 {
                    0.
                } else {
                    0.125 * (i as f64 - 3.)
                })
                .unwrap(),
            )
        })
        .collect();
    let exact = [1., -2., 3.]
        .into_iter()
        .enumerate()
        .map(|(i, value)| {
            S::from_components(
                S::Real::try_from_f64_lossy(value).unwrap(),
                S::Real::try_from_f64_lossy(0.25 * i as f64).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let b = (0..3)
        .map(|row| (0..3).fold(S::ZERO, |sum, col| sum + a[row * 3 + col] * exact[col]))
        .collect();
    (
        Matrix::from_vec(3, 3, a).unwrap(),
        Vector::from_vec(b),
        exact,
    )
}
fn error<S: NumericScalar>(actual: &[S], expected: &[S]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&x, &y)| (x - y).abs().try_to_f64_exact().unwrap())
        .fold(0., f64::max)
}
struct Changing {
    calls: Cell<usize>,
    n: usize,
}
impl<S: NumericScalar> Preconditioner<S> for Changing {
    fn dimension(&self) -> usize {
        self.n
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn precondition_into(
        &self,
        input: &[S],
        output: &mut [S],
        _: &mut [S],
    ) -> Result<(), SciOxideError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        for (i, (out, &value)) in output.iter_mut().zip(input).enumerate() {
            *out = value / real::<S>(1. + ((call + i) % 3) as f64);
        }
        Ok(())
    }
}
fn qualify<S: NumericScalar>(tol: f64) {
    let (a, b, exact) = fixture::<S>();
    let tolerance =
        ResidualTolerance::new(S::Real::try_from_f64_lossy(tol).unwrap(), S::Real::ZERO).unwrap();
    let p = Changing {
        calls: Cell::new(0),
        n: 3,
    };
    macro_rules! scalar {
        ($solver:ident) => {{
            let solver = $solver::<S>::new(tolerance, budget(30)).with_restart(budget(3));
            let result = solver
                .solve_with_options(
                    &a.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::Inverse(&p),
                    None,
                )
                .unwrap();
            assert!(
                result.is_success(),
                "{} {:?}",
                stringify!($solver),
                result.report().termination()
            );
            assert!(error(result.solution().as_slice(), &exact) < tol * 10.);
            // Independent dense multiplication, not the solver's operator or residual helper.
            let residual = (0..3)
                .map(|row| {
                    let sum = (0..3).fold(S::ZERO, |sum, col| {
                        sum + a.as_slice()[row * 3 + col] * result.solution().as_slice()[col]
                    });
                    (sum - b.as_slice()[row])
                        .abs()
                        .try_to_f64_exact()
                        .unwrap()
                        .powi(2)
                })
                .sum::<f64>()
                .sqrt();
            assert!(residual <= tol * 1.05);
        }};
    }
    scalar!(GmresSolver);
    scalar!(LgmresSolver);
    scalar!(GcrotSolver);
    let mut block = b.as_slice().to_vec();
    block.extend(b.as_slice().iter().map(|&v| v * real::<S>(2.)));
    block.extend([S::ZERO; 3]);
    // The fourth column is independent: exact solution e_1.
    block.extend((0..3).map(|row| a.as_slice()[row * 3]));
    let mut expected = exact.clone();
    expected.extend(exact.iter().map(|&v| v * real::<S>(2.)));
    expected.extend([S::ZERO; 3]);
    expected.extend([S::ONE, S::ZERO, S::ZERO]);
    macro_rules! block {
        ($solver:ident, $p:expr) => {{
            let solver = $solver::<S>::new(tolerance, budget(30)).with_restart(budget(3));
            let result = solver
                .solve(
                    &a.view().try_finite().unwrap(),
                    BlockVectorView::new(&block, 3, 4).unwrap(),
                    $p,
                    None,
                )
                .unwrap();
            assert_eq!(
                result.report().termination(),
                LinearSolveTermination::Converged,
                "{}",
                stringify!($solver)
            );
            assert!(error(result.solution(), &expected) < tol * 10.);
            assert!(result.report().deflated() > 0);
            assert!(result.converged().iter().all(|&value| value));
            assert!(result.report().iterations() <= 3);
        }};
    }
    block!(BlockGmresSolver, Preconditioning::None);
    block!(BlockGmresSolver, Preconditioning::Inverse(&p));
    block!(BlockLgmresSolver, Preconditioning::Inverse(&p));
    block!(BlockGcrotSolver, Preconditioning::Inverse(&p));
}
#[test]
fn real_f64() {
    qualify::<f64>(1e-11);
}
#[test]
fn real_f32() {
    qualify::<f32>(2e-5);
}
#[test]
fn complex_f64() {
    qualify::<Complex<f64>>(1e-11);
}
#[test]
fn complex_f32() {
    qualify::<Complex<f32>>(2e-5);
}

#[test]
fn complex_phase_requires_conjugate_projection() {
    let i = Complex::new(0., 1.);
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            i,
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(1., 0.),
        ],
    )
    .unwrap();
    let b = Vector::from_vec(vec![Complex::new(1., 0.), Complex::new(0., 0.)]);
    let solver = GmresSolver::new(ResidualTolerance::new(1e-12, 0.).unwrap(), budget(4));
    let result = solver
        .solve_with_options(
            &a.view().try_finite().unwrap(),
            &b,
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert!(result.is_success());
    assert!(error(result.solution().as_slice(), &[-i, i]) < 1e-12);
}

#[test]
fn happy_and_serious_breakdown_are_typed() {
    let a = Matrix::from_vec(2, 2, vec![1., 0., 0., 0.]).unwrap();
    let solver = GmresSolver::new(ResidualTolerance::new(1e-12, 0.).unwrap(), budget(4));
    let good = solver
        .solve_with_options(
            &a.view().try_finite().unwrap(),
            &Vector::from_vec(vec![1., 0.]),
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert!(good.is_success());
    assert_eq!(good.report().iterations(), 1);
    let bad = solver
        .solve_with_options(
            &a.view().try_finite().unwrap(),
            &Vector::from_vec(vec![0., 1.]),
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert!(matches!(
        bad.report().termination(),
        LinearSolveTermination::Breakdown(_)
    ));
    assert_eq!(bad.report().true_residual().unwrap().raw_norm(), 1.);
}

#[test]
fn nearly_dependent_rhs_keep_individual_stopping_tests() {
    let a = Matrix::from_vec(2, 2, vec![1., 0., 0., 2.]).unwrap();
    let e = 1e-8;
    let rhs = [1., 1., 1. + e, 1. - e];
    let solver = BlockGmresSolver::new(ResidualTolerance::new(1e-12, 0.).unwrap(), budget(8))
        .with_rank_threshold(scioxide_core::RankThreshold::new(0., 1e-6).unwrap());
    let result = solver
        .solve(
            &a.view().try_finite().unwrap(),
            BlockVectorView::new(&rhs, 2, 2).unwrap(),
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert!(result.report().termination().is_success());
    assert!(result.report().deflated() > 0);
    assert!(error(result.solution(), &[1., 0.5, 1. + e, (1. - e) / 2.]) < 1e-12);
    assert!(result.residuals().iter().all(|&r| r < 1e-12));
}

fn uniform_scaling<S: NumericScalar>(small: f64, large: f64, tol: f64) {
    for scale in [small, large] {
        let a = Matrix::from_vec(
            2,
            2,
            vec![
                real::<S>(2. * scale),
                real(scale),
                real(0.),
                real(3. * scale),
            ],
        )
        .unwrap();
        let b = Vector::from_vec(vec![real::<S>(4. * scale), real(6. * scale)]);
        let solver = GmresSolver::new(
            ResidualTolerance::new(S::Real::ZERO, S::Real::try_from_f64_lossy(tol).unwrap())
                .unwrap(),
            budget(4),
        );
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.is_success());
        assert!(error(result.solution().as_slice(), &[real::<S>(1.), real(2.)]) < tol * 10.);
    }
}
#[test]
fn scalar_range_scaling() {
    uniform_scaling::<f64>(1e-150, 1e150, 1e-12);
    uniform_scaling::<f32>(1e-20, 1e20, 2e-6);
    uniform_scaling::<Complex<f64>>(1e-150, 1e150, 1e-12);
    uniform_scaling::<Complex<f32>>(1e-20, 1e20, 2e-6);
}

/// Row-major dense operator with plain left-to-right products.
struct Dense {
    n: usize,
    a: Vec<f64>,
}
impl scioxide_core::LinearOperator<f64> for Dense {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        for (row, out) in self.a.chunks(self.n).zip(y.iter_mut()) {
            *out = row.iter().zip(x).fold(0., |sum, (&a, &x)| sum + a * x);
        }
        Ok(())
    }
}

/// Linear congruential stream in `[−1/2, 1/2)`.
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
}

/// Nonsymmetric tridiagonal `tridiag(−1.5, 2 + i/(2n), −0.3)` with `fill`
/// random entries added, and a random right-hand side.
fn nonsymmetric(n: usize, fill: usize, seed: &mut u64) -> (Dense, Vec<f64>) {
    let mut a = vec![0.; n * n];
    for i in 0..n {
        a[i * n + i] = 2. + 0.5 * (i as f64 / n as f64);
        if i + 1 < n {
            a[i * n + i + 1] = -0.3;
            a[(i + 1) * n + i] = -1.5;
        }
    }
    for _ in 0..fill {
        let i = ((lcg(seed) + 0.5) * n as f64) as usize;
        let j = ((lcg(seed) + 0.5) * n as f64) as usize;
        a[i * n + j] += 0.4 * lcg(seed);
    }
    let b = (0..n).map(|_| lcg(seed)).collect();
    (Dense { n, a }, b)
}

fn residual_norm(a: &Dense, b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.; b.len()];
    scioxide_core::LinearOperator::apply_into(a, x, &mut ax, &mut []).unwrap();
    ax.iter().zip(b).fold(0_f64, |n, (&ax, &b)| n.hypot(b - ax))
}

#[test]
fn gcrot_projection_onto_recycled_images_does_not_stagnate_the_cycle() {
    use scioxide_core::{ExecutionContext, ProblemToken};
    use scioxide_linalg::{IterativeWorkspace, KrylovRecycleSpace, VectorViewMut};
    // The second cycle starts by projecting onto the images retained from the
    // first; that projection can leave `x` unchanged before Arnoldi steps make
    // progress.
    let n = 100;
    let tolerance = ResidualTolerance::new(0., 1e-10).unwrap();
    for seed in 1..=12 {
        let mut stream = seed;
        let (a, b) = nonsymmetric(n, 150, &mut stream);
        for (restart, retained) in [(5, 3), (10, 4)] {
            let mut space =
                KrylovRecycleSpace::<f64>::new(n, retained, ProblemToken::new(1)).unwrap();
            let mut workspace = IterativeWorkspace::new();
            let mut x = vec![0.; n];
            let report = GcrotSolver::new(tolerance, budget(400))
                .with_restart(budget(restart))
                .solve_recycled_into(
                    &a,
                    &b[..],
                    Preconditioning::None,
                    ProblemToken::new(1),
                    &mut space,
                    VectorViewMut::from_slice(&mut x),
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                LinearSolveTermination::Converged,
                "seed {seed} GCROT({restart}, {retained}): {report:?}"
            );
            let reference = b.iter().fold(0_f64, |s, &v| s.hypot(v));
            assert!(residual_norm(&a, &b, &x) <= 1.1e-10 * reference);
        }
    }
}

#[test]
fn block_gmres_judges_an_unchanged_iterate_at_the_end_of_the_cycle() {
    // `A = diag(S(1), S(2))` with `S(t) = [[0, t], [−t, 0]]` and `B = [e₁, e₃]`:
    // the first block step is orthogonal to both residuals, the second solves
    // both columns exactly.
    let a = Matrix::from_vec(
        4,
        4,
        vec![
            0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 2., 0., 0., -2., 0.,
        ],
    )
    .unwrap();
    let b = [1., 0., 0., 0., 0., 0., 1., 0.];
    let result = BlockGmresSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), budget(20))
        .solve(
            &a.view().try_finite().unwrap(),
            BlockVectorView::new(&b, 4, 2).unwrap(),
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert_eq!(
        result.report().termination(),
        LinearSolveTermination::Converged
    );
    assert_eq!(result.report().iterations(), 2);
    let expected = [0., 1., 0., 0., 0., 0., 0., 0.5];
    for (x, e) in result.solution().iter().zip(expected) {
        assert!((x - e).abs() <= 1e-15, "{:?}", result.solution());
    }
}

#[test]
fn recycled_pairs_keep_their_images_after_cycles_at_the_rounding_floor() {
    use scioxide_core::{ExecutionContext, LinearOperator, ProblemToken};
    use scioxide_linalg::{IterativeWorkspace, KrylovRecycleSpace, VectorViewMut};
    let n = 100;
    let mut stream = 5;
    let (a, b) = nonsymmetric(n, 0, &mut stream);
    // An unattainable tolerance runs several cycles whose corrections change
    // the residual only at the rounding level.
    let tolerance = ResidualTolerance::new(0., 1e-20).unwrap();
    for gcrot in [true, false] {
        let mut space = KrylovRecycleSpace::<f64>::new(n, 4, ProblemToken::new(1)).unwrap();
        let mut workspace = IterativeWorkspace::new();
        let mut x = vec![0.; n];
        let report = if gcrot {
            GcrotSolver::new(tolerance, budget(300))
                .with_restart(budget(10))
                .solve_recycled_into(
                    &a,
                    &b[..],
                    Preconditioning::None,
                    ProblemToken::new(1),
                    &mut space,
                    VectorViewMut::from_slice(&mut x),
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
        } else {
            LgmresSolver::new(tolerance, budget(300))
                .with_restart(budget(10))
                .solve_recycled_into(
                    &a,
                    &b[..],
                    Preconditioning::None,
                    ProblemToken::new(1),
                    &mut space,
                    VectorViewMut::from_slice(&mut x),
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
        }
        .unwrap();
        assert!(!report.is_success());
        assert_eq!(space.len(), 4);
        for index in 0..space.len() {
            let (u, c) = space.pair(index).unwrap();
            let mut au = vec![0.; n];
            a.apply_into(u, &mut au, &mut []).unwrap();
            let defect = au.iter().zip(c).fold(0_f64, |s, (&au, &c)| s.hypot(au - c));
            let image = c.iter().fold(0_f64, |s, &c| s.hypot(c));
            assert!(
                defect <= 1e-13 * image,
                "gcrot={gcrot} pair {index}: ‖A u − c‖ = {defect:e}"
            );
        }
    }
}
