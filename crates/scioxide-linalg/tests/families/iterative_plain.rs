//! Plain solves of the single-vector iterative solvers: `solve(a, b)` is
//! `solve_with_options` from zero without a preconditioner followed by
//! `into_converged_solution`, bit for bit, for converged and unconverged solves.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, ConvergenceReason, NestedLevel, ResidualTolerance, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView,
};
use scioxide_linalg::{
    BicgSolver, BicgstabSolver, CgSolver, CgsSolver, GaussSeidelSolver, GcrotSolver, GmresSolver,
    IterativeLinearSolver, IterativeSolveResult, JacobiSolver, LgmresSolver,
    LinearSolveTermination, LsmrSolver, LsqrSolver, Matrix, MinresSolver, Preconditioning,
    QmrSolver, RelaxationWeight, RichardsonSolver, SorSolver, SorWeight, StationaryMatrix,
    TfqmrSolver, Vector, VectorView,
};

/// `A = [[4, 1], [1, 3]]` is symmetric positive definite and strictly
/// diagonally dominant, so the Krylov and the stationary solvers converge.
fn spd() -> Matrix<f64> {
    Matrix::from_vec(2, 2, vec![4., 1., 1., 3.]).unwrap()
}
const B: [f64; 2] = [1., 2.];
/// `A⁻¹ b`.
const X: [f64; 2] = [1. / 11., 7. / 11.];

/// Inconsistent problem with rows `(1, 0)`, `(0, 1)`, `(1, 1)`: the normal
/// equations give `x = (4/3, 7/3)` and the residual `(−1, −1, 1)/3`.
fn tall() -> Matrix<f64> {
    Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap()
}
const B_TALL: [f64; 3] = [1., 2., 4.];
const X_TALL: [f64; 2] = [4. / 3., 7. / 3.];

type Plain = Result<Vector<f64>, SciOxideError>;
type Configured = Result<IterativeSolveResult<f64>, SciOxideError>;

fn bits(x: &Vector<f64>) -> Vec<u64> {
    x.as_slice().iter().map(|v| v.to_bits()).collect()
}

/// Compares the plain solve with the configured solve built from the same
/// tolerance and iteration limit: once with a limit that converges, and once
/// with one iteration and tolerance `1e-14`, which cannot converge on these
/// systems and gives `ConvergenceFailure` (`NestedSolver`).
fn check(
    plain: impl Fn(ResidualTolerance<f64>, Budget) -> Plain,
    configured: impl Fn(ResidualTolerance<f64>, Budget) -> Configured,
    expected: &[f64],
) {
    let tolerance = ResidualTolerance::new(0., 1e-12).unwrap();
    let limit = Budget::try_from(200).unwrap();
    let result = configured(tolerance, limit).unwrap();
    assert!(result.is_success(), "{:?}", result.report().termination());
    let reference = result.into_converged_solution().unwrap();
    let solution = plain(tolerance, limit).unwrap();
    assert_eq!(bits(&solution), bits(&reference));
    assert_eq!(solution.len(), expected.len());
    for (x, e) in solution.as_slice().iter().zip(expected) {
        assert!((x - e).abs() < 1e-7, "{x} != {e}");
    }

    let tolerance = ResidualTolerance::new(0., 1e-14).unwrap();
    let result = configured(tolerance, Budget::ONE).unwrap();
    assert!(!result.is_success(), "{:?}", result.report().termination());
    let (algorithm, iterations) = (result.algorithm(), result.report().iterations());
    assert!(iterations <= 1);
    let reference = result.into_converged_solution().unwrap_err();
    let error = plain(tolerance, Budget::ONE).unwrap_err();
    assert_eq!(error, reference);
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::NestedSolver,
                diagnostics,
                ..
            } if diagnostics.nested_levels().map(NestedLevel::algorithm).eq([algorithm])
                && diagnostics
                    .nested_levels()
                    .all(|level| level.counts().own().iterations() == iterations)
        ),
        "{error:?}"
    );
}

/// Solvers with a preconditioning argument: `&dyn LinearOperator` or
/// `&dyn AdjointOperator`.
macro_rules! preconditioned {
    ($test:ident, $solver:ident $(, $arg:expr)*) => {
        #[test]
        fn $test() {
            let matrix = spd();
            let a = matrix.view().try_finite().unwrap();
            let make = |tolerance, limit| $solver::<f64>::new(tolerance, limit $(, $arg)*);
            check(
                |tolerance, limit| make(tolerance, limit).solve(&a, &B[..]),
                |tolerance, limit| {
                    make(tolerance, limit).solve_with_options(
                        &a,
                        &B[..],
                        Preconditioning::None,
                        None,
                    )
                },
                &X,
            );
        }
    };
}

/// The plain solve of `IterativeLinearSolver` through a trait object.
macro_rules! object_safe {
    ($test:ident, $solver:ident $(, $arg:expr)*) => {
        #[test]
        fn $test() {
            let matrix = spd();
            let a = matrix.view().try_finite().unwrap();
            let make = |tolerance, limit| $solver::<f64>::new(tolerance, limit $(, $arg)*);
            check(
                |tolerance, limit| {
                    let solver = make(tolerance, limit);
                    let solver: &dyn IterativeLinearSolver<f64> = &solver;
                    solver.solve(&a, VectorView::from_slice(&B))
                },
                |tolerance, limit| {
                    make(tolerance, limit).solve_with_options(
                        &a,
                        &B[..],
                        Preconditioning::None,
                        None,
                    )
                },
                &X,
            );
        }
    };
}

/// Matrix-splitting solvers, which take only a start vector.
macro_rules! stationary {
    ($test:ident, $solver:ident $(, $arg:expr)*) => {
        #[test]
        fn $test() {
            let matrix = spd();
            let a = StationaryMatrix::Dense(matrix.view().try_finite().unwrap());
            let make = |tolerance, limit| $solver::<f64>::new(tolerance, limit $(, $arg)*);
            check(
                |tolerance, limit| make(tolerance, limit).solve(a, &B[..]),
                |tolerance, limit| make(tolerance, limit).solve_with_options(a, &B[..], None),
                &X,
            );
        }
    };
}

/// Least-squares solvers, which take only a start vector: the square system
/// and the inconsistent rectangular one, which stops on the normal residual.
macro_rules! least_squares {
    ($square:ident, $rectangular:ident, $solver:ident) => {
        #[test]
        fn $square() {
            let matrix = spd();
            let a = matrix.view().try_finite().unwrap();
            check(
                |tolerance, limit| $solver::<f64>::new(tolerance, limit).solve(&a, &B[..]),
                |tolerance, limit| {
                    $solver::<f64>::new(tolerance, limit).solve_with_options(&a, &B[..], None)
                },
                &X,
            );
        }

        #[test]
        fn $rectangular() {
            let matrix = tall();
            let a = matrix.view().try_finite().unwrap();
            let configured = |tolerance, limit| {
                $solver::<f64>::new(tolerance, limit).solve_with_options(&a, &B_TALL[..], None)
            };
            check(
                |tolerance, limit| $solver::<f64>::new(tolerance, limit).solve(&a, &B_TALL[..]),
                configured,
                &X_TALL,
            );
            let tolerance = ResidualTolerance::new(0., 1e-12).unwrap();
            let report = configured(tolerance, Budget::try_from(200).unwrap())
                .unwrap()
                .report();
            assert_eq!(
                report.termination(),
                LinearSolveTermination::LeastSquaresConverged
            );
        }
    };
}

fn richardson_weight() -> RelaxationWeight<f64> {
    // The eigenvalues of A are (7 ± √5)/2, so ω = 1/4 gives spectral radius
    // max|1 − ω λ| ≈ 0.40.
    RelaxationWeight::new(0.25).unwrap()
}

fn sor_weight() -> SorWeight<f64> {
    SorWeight::new(1.1).unwrap()
}

preconditioned!(cg_plain_solve, CgSolver);
preconditioned!(gmres_plain_solve, GmresSolver);
preconditioned!(bicgstab_plain_solve, BicgstabSolver);
preconditioned!(minres_plain_solve, MinresSolver);
preconditioned!(
    richardson_plain_solve,
    RichardsonSolver,
    richardson_weight()
);
preconditioned!(cgs_plain_solve, CgsSolver);
preconditioned!(tfqmr_plain_solve, TfqmrSolver);
preconditioned!(lgmres_plain_solve, LgmresSolver);
preconditioned!(gcrot_plain_solve, GcrotSolver);
preconditioned!(bicg_plain_solve, BicgSolver);
preconditioned!(qmr_plain_solve, QmrSolver);

object_safe!(cg_plain_solve_through_trait_object, CgSolver);
object_safe!(gmres_plain_solve_through_trait_object, GmresSolver);
object_safe!(bicgstab_plain_solve_through_trait_object, BicgstabSolver);
object_safe!(minres_plain_solve_through_trait_object, MinresSolver);
object_safe!(
    richardson_plain_solve_through_trait_object,
    RichardsonSolver,
    richardson_weight()
);
object_safe!(cgs_plain_solve_through_trait_object, CgsSolver);
object_safe!(tfqmr_plain_solve_through_trait_object, TfqmrSolver);
object_safe!(lgmres_plain_solve_through_trait_object, LgmresSolver);
object_safe!(gcrot_plain_solve_through_trait_object, GcrotSolver);

least_squares!(lsqr_plain_solve, lsqr_plain_least_squares_solve, LsqrSolver);
least_squares!(lsmr_plain_solve, lsmr_plain_least_squares_solve, LsmrSolver);

stationary!(jacobi_plain_solve, JacobiSolver);
stationary!(gauss_seidel_plain_solve, GaussSeidelSolver);
stationary!(sor_plain_solve, SorSolver, sor_weight());

/// A returned iterate is `ResidualChecked` only when its true residual passed
/// the tolerance; a verified iterate that did not is `Admissible`.
#[test]
fn trust_follows_the_residual_test_of_the_returned_iterate() {
    use scioxide_core::PartialResultTrust;
    let a = Matrix::from_vec(3, 3, vec![4., 1., 0., 1., 3., 1., 0., 1., 2.]).unwrap();
    let a = a.view().try_finite().unwrap();
    let b = Vector::from_vec(vec![1., 2., 3.]);
    let tight = ResidualTolerance::new(0., 1e-14).unwrap();
    let loose = ResidualTolerance::new(0., 1e-8).unwrap();
    macro_rules! check {
        ($solve:expr) => {{
            let solve = $solve;
            for (tolerance, limit, success) in [(tight, 1, false), (loose, 500, true)] {
                let result: IterativeSolveResult<f64> =
                    solve(tolerance, Budget::try_from(limit).unwrap());
                let report = result.report();
                assert_eq!(report.is_success(), success, "{report:?}");
                assert_eq!(
                    report.trust(),
                    if success {
                        PartialResultTrust::ResidualChecked
                    } else {
                        PartialResultTrust::Admissible
                    },
                    "{report:?}"
                );
            }
        }};
    }
    let none = Preconditioning::None;
    check!(|t, l| CgSolver::new(t, l)
        .solve_with_options(&a, &b, none, None)
        .unwrap());
    check!(|t, l| MinresSolver::new(t, l)
        .solve_with_options(&a, &b, none, None)
        .unwrap());
    check!(|t, l| BicgstabSolver::new(t, l)
        .solve_with_options(&a, &b, none, None)
        .unwrap());
    check!(|t, l| GmresSolver::new(t, l)
        .solve_with_options(&a, &b, none, None)
        .unwrap());
    check!(|t, l| GcrotSolver::new(t, l)
        .solve_with_options(&a, &b, none, None)
        .unwrap());
    check!(|t, l| TfqmrSolver::new(t, l)
        .solve_with_options(&a, &b, none, None)
        .unwrap());
    check!(|t, l| LsqrSolver::new(t, l)
        .solve_with_options(&a, &b, None)
        .unwrap());
    check!(|t, l| JacobiSolver::new(t, l)
        .solve_with_options(StationaryMatrix::Dense(a), &b, None)
        .unwrap());
    check!(
        |t, l| RichardsonSolver::new(t, l, RelaxationWeight::new(0.2).unwrap())
            .solve_with_options(&a, &b, none, None)
            .unwrap()
    );

    // Block solves report the same trust through the workspace progress.
    let rhs = [1., 2., 3., 3., 2., 1.];
    for (tolerance, limit, success) in [(tight, 1, false), (loose, 500, true)] {
        let result =
            scioxide_linalg::BlockGmresSolver::new(tolerance, Budget::try_from(limit).unwrap())
                .solve(
                    &a,
                    scioxide_core::BlockVectorView::new(&rhs, 3, 2).unwrap(),
                    none,
                    None,
                )
                .unwrap();
        assert_eq!(result.report().is_success(), success);
        let mut workspace = scioxide_linalg::IterativeWorkspace::new();
        let mut x = [0.; 6];
        let _ = scioxide_linalg::BlockGmresSolver::new(tolerance, Budget::try_from(limit).unwrap())
            .solve_into(
                &a,
                scioxide_core::BlockVectorView::new(&rhs, 3, 2).unwrap(),
                none,
                scioxide_core::BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            workspace.last_accepted().unwrap().progress().trust(),
            if success {
                PartialResultTrust::ResidualChecked
            } else {
                PartialResultTrust::Admissible
            }
        );
    }
}
