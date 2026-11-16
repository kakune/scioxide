//! Complex Hermitian CG/MINRES contracts on ill-conditioned positive-definite,
//! indefinite and non-Hermitian operators.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Budget, Complex, NumericScalar, RealScalar, ResidualTolerance, SciOxideErrorKind,
};
use scioxide_linalg::{
    CgSolver, LinearSolveBreakdown, LinearSolveTermination, Matrix, MinresSolver, Preconditioning,
    Vector,
};
use scioxide_sparse::{CsrMatrix, JacobiPreconditioner};

/// Deterministic splitmix64 stream in `[0, 1)`.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) as f64 / 18_446_744_073_709_551_616.0
    }
    fn phase(&mut self) -> Complex<f64> {
        let angle = core::f64::consts::TAU * self.next();
        Complex::new(angle.cos(), angle.sin())
    }
}

/// `D F diag(λ) Fᴴ Dᴴ` with the unitary DFT `F` and a random phase diagonal
/// `D`, stored exactly Hermitian in row-major order.
fn hermitian(spectrum: &[f64], stream: &mut Stream) -> Vec<Complex<f64>> {
    let n = spectrum.len();
    let phases: Vec<_> = (0..n).map(|_| stream.phase()).collect();
    let mut a = vec![Complex::new(0., 0.); n * n];
    for row in 0..n {
        for col in row..n {
            let sum = spectrum
                .iter()
                .enumerate()
                .map(|(mode, &lambda)| {
                    let shift = (row as f64 - col as f64) * mode as f64;
                    let angle = core::f64::consts::TAU * shift / n as f64;
                    Complex::new(angle.cos(), angle.sin()) * lambda
                })
                .fold(Complex::new(0., 0.), |acc, term| acc + term)
                / n as f64;
            let entry = phases[row] * sum * phases[col].conj();
            if row == col {
                a[row * n + col] = Complex::new(entry.re, 0.);
            } else {
                a[row * n + col] = entry;
                a[col * n + row] = entry.conj();
            }
        }
    }
    a
}

/// `λ = logspace(-log10 κ, 0)`, shuffled.
fn spectrum(n: usize, condition: f64, stream: &mut Stream) -> Vec<f64> {
    let mut values: Vec<_> = (0..n)
        .map(|index| condition.powf(index as f64 / (n - 1) as f64 - 1.))
        .collect();
    for index in (1..n).rev() {
        values.swap(index, (stream.next() * (index + 1) as f64) as usize);
    }
    values
}

struct Problem<R: RealScalar> {
    matrix: Matrix<Complex<R>>,
    rhs: Vector<Complex<R>>,
    jacobi: JacobiPreconditioner<Complex<R>>,
}

fn problem<R: RealScalar>(spectrum: &[f64], stream: &mut Stream) -> Problem<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    let n = spectrum.len();
    let cast = |z: Complex<f64>| {
        Complex::new(
            R::try_from_f64_lossy(z.re).unwrap(),
            R::try_from_f64_lossy(z.im).unwrap(),
        )
    };
    let values: Vec<_> = hermitian(spectrum, stream).into_iter().map(cast).collect();
    let diagonal = CsrMatrix::from_raw(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        (0..n).map(|index| values[index * n + index]).collect(),
    )
    .unwrap();
    let rhs = (0..n)
        .map(|_| cast(Complex::new(stream.next() - 0.5, stream.next() - 0.5)))
        .collect();
    Problem {
        matrix: Matrix::from_vec(n, n, values).unwrap(),
        rhs: Vector::from_vec(rhs),
        jacobi: JacobiPreconditioner::new(&diagonal).unwrap(),
    }
}

fn assert_accepted(termination: LinearSolveTermination, label: &str) {
    assert!(
        matches!(
            termination,
            LinearSolveTermination::Converged | LinearSolveTermination::Stagnation
        ),
        "{label}: {termination:?}"
    );
}

fn assert_residual<R: RealScalar>(
    problem: &Problem<R>,
    result: &scioxide_linalg::IterativeSolveResult<Complex<R>>,
    tolerance: ResidualTolerance<R>,
    label: &str,
) where
    Complex<R>: NumericScalar<Real = R>,
{
    let x = result.solution().as_slice();
    let n = x.len();
    let residual: Vec<_> = problem
        .matrix
        .as_slice()
        .chunks(n)
        .zip(problem.rhs.as_slice())
        .map(|(row, &b)| {
            let mut sum = scioxide_core::ProductSum::new();
            sum.add_product(b, Complex::new(R::ONE, R::ZERO)).unwrap();
            for (&a, &x) in row.iter().zip(x) {
                sum.subtract_product(a, x).unwrap();
            }
            sum.finish().unwrap()
        })
        .collect();
    let residual = Vector::from_vec(residual).view().norm2().unwrap();
    let rhs = problem.rhs.view().norm2().unwrap();
    if result.report().termination() == LinearSolveTermination::Converged {
        // Exact verification products can differ from the solver's Standard
        // products by their componentwise rounding envelope.
        let rounding = backward_scale(problem, x)
            * R::try_from_f64_exact(8.0).unwrap()
            * R::EPSILON
            * R::try_from_f64_exact(n as f64).unwrap();
        assert!(
            residual <= tolerance.threshold(rhs).unwrap() + rounding,
            "{label}"
        );
    } else {
        // Stagnation is an unsuccessful stop, not a machine-precision
        // backward-stability guarantee. Require a useful iterate at the
        // library's default sqrt-epsilon relative tolerance independently.
        let bound = R::SQRT_EPSILON;
        assert!(
            residual / backward_scale(problem, x) <= bound,
            "{label}: true residual {residual:?}, scale={:?}, bound={bound:?}, report={:?}",
            backward_scale(problem, x),
            result.report()
        );
    }
}

fn backward_scale<R: RealScalar>(problem: &Problem<R>, x: &[Complex<R>]) -> R
where
    Complex<R>: NumericScalar<Real = R>,
{
    let matrix_norm = Vector::from_vec(problem.matrix.as_slice().to_vec())
        .view()
        .norm2()
        .unwrap();
    let solution_norm = scioxide_linalg::VectorView::from_slice(x).norm2().unwrap();
    matrix_norm * solution_norm + problem.rhs.view().norm2().unwrap()
}

fn ill_conditioned_family<R: RealScalar>(sizes: &[usize], conditions: &[f64], relative: f64)
where
    Complex<R>: NumericScalar<Real = R>,
{
    let tolerance =
        ResidualTolerance::new(R::ZERO, R::try_from_f64_lossy(relative).unwrap()).unwrap();
    for &n in sizes {
        for &condition in conditions {
            // Severe conditioning can require more than 20n iterations to
            // reach the rounding floor even for small systems.
            let factor = if condition > 1e6 { 100 } else { 20 };
            let budget = Budget::try_from(factor * n).unwrap();
            for seed in 0..3 {
                let mut stream = Stream(seed);
                let spectrum = spectrum(n, condition, &mut stream);
                let p = problem::<R>(&spectrum, &mut stream);
                let operator = p.matrix.view().try_finite().unwrap();
                for (name, preconditioning) in [
                    ("none", Preconditioning::None),
                    ("jacobi", Preconditioning::Inverse(&p.jacobi)),
                ] {
                    let label = format!("n={n} κ={condition:e} seed={seed} {name}");
                    let cg = CgSolver::new(tolerance, budget)
                        .solve_with_options(&operator, &p.rhs, preconditioning, None)
                        .unwrap();
                    assert_accepted(cg.report().termination(), &format!("CG {label}"));
                    assert_residual(&p, &cg, tolerance, &format!("CG {label}"));
                    let minres = MinresSolver::new(tolerance, budget)
                        .solve_with_options(&operator, &p.rhs, preconditioning, None)
                        .unwrap();
                    assert_accepted(minres.report().termination(), &format!("MINRES {label}"));
                    assert_residual(&p, &minres, tolerance, &format!("MINRES {label}"));
                }
            }
        }
    }
}

#[test]
fn ill_conditioned_complex64_hermitian_systems_converge() {
    ill_conditioned_family::<f64>(&[16, 64], &[1e2, 1e3, 1e4, 1e6, 1e8], 1e-10);
}

#[test]
fn ill_conditioned_complex32_hermitian_systems_converge() {
    ill_conditioned_family::<f32>(&[16, 64], &[1e2, 1e3], 1e-4);
}

#[test]
fn indefinite_complex_hermitian_cg_reports_nonpositive_curvature() {
    let mut stream = Stream(7);
    let spectrum: Vec<_> = (0..16)
        .map(|index| if index % 2 == 0 { 1. } else { -0.5 })
        .collect();
    let p = problem::<f64>(&spectrum, &mut stream);
    let operator = p.matrix.view().try_finite().unwrap();
    let result = CgSolver::new(
        ResidualTolerance::new(0., 1e-10).unwrap(),
        Budget::try_from(320_usize).unwrap(),
    )
    .solve_with_options(&operator, &p.rhs, Preconditioning::None, None)
    .unwrap();
    assert_eq!(
        result.report().termination(),
        LinearSolveTermination::Breakdown(LinearSolveBreakdown::NonPositiveCurvature)
    );
}

#[test]
fn non_hermitian_complex_operator_is_rejected_by_minres() {
    let mut stream = Stream(11);
    let spectrum = spectrum(16, 1e2, &mut stream);
    let p = problem::<f64>(&spectrum, &mut stream);
    // Add a comparable anti-Hermitian part `i·0.1·I`.
    let values: Vec<_> = (0..16 * 16)
        .map(|index| {
            let entry = p.matrix.as_slice()[index];
            if index % 17 == 0 {
                entry + Complex::new(0., 0.1)
            } else {
                entry
            }
        })
        .collect();
    let matrix = Matrix::from_vec(16, 16, values).unwrap();
    let error = MinresSolver::new(
        ResidualTolerance::new(0., 1e-10).unwrap(),
        Budget::try_from(320_usize).unwrap(),
    )
    .solve_with_options(
        &matrix.view().try_finite().unwrap(),
        &p.rhs,
        Preconditioning::None,
        None,
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
}
