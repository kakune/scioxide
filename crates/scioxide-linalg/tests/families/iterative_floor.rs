//! Rounding-floor and best-iterate regressions shared across solver families.
#![allow(clippy::unwrap_used)]
use crate::support::krylov_reference;
use scioxide_core::{Budget, ExecutionContext, LinearOperator, ResidualTolerance};
use scioxide_linalg::*;

#[test]
fn reserved_floor_all_scalar_families() {
    let mut a = krylov_reference::fixtures()
        .find(|f| f.problem == "poisson")
        .unwrap();
    a.order = 256;
    let b = a.rhs();
    let tolerance = ResidualTolerance::new(0., 1e-16).unwrap();
    let limit = Budget::try_from(4000).unwrap();
    macro_rules! check {
        ($solver:expr $(, $p:expr)?) => {{
            let solver = $solver.with_history(true);
            let mut x = Vector::zeros(a.cols()).unwrap();
            let mut w = solver.workspace(&IterativeWorkspaceRequest::adjoint(&a).unwrap()).unwrap();
            let report = { let mut evaluate_case = || solver.solve_into(&a, &b, $($p,)? x.view_mut(), &mut w, ExecutionContext::sequential()).unwrap(); evaluate_case() };
            let mut ax = vec![0.; a.rows()];
            a.apply_into(x.as_slice(), &mut ax, &mut []).unwrap();
            let residual = ax.iter().fold(0_f64, |n, &v| n.hypot(1. - v));
            assert!(matches!(report.termination(), LinearSolveTermination::Stagnation | LinearSolveTermination::Breakdown(LinearSolveBreakdown::RecurrenceDenominator)), "{}: {report:?}", stringify!($solver));
            assert!((report.diagnostic_residual_norm().unwrap() - residual).abs() <= 1e-12 * residual.max(1.));
            assert_eq!(w.last_accepted().unwrap().history().unwrap().last(), report.diagnostic_residual_norm().as_ref());
        }};
    }
    check!(CgSolver::new(tolerance, limit), Preconditioning::None);
    check!(BicgstabSolver::new(tolerance, limit), Preconditioning::None);
    check!(GmresSolver::new(tolerance, limit), Preconditioning::None);
    check!(LgmresSolver::new(tolerance, limit), Preconditioning::None);
    check!(GcrotSolver::new(tolerance, limit), Preconditioning::None);
    check!(MinresSolver::new(tolerance, limit), Preconditioning::None);
    check!(LsqrSolver::new(tolerance, limit));
    check!(LsmrSolver::new(tolerance, limit));
    check!(BicgSolver::new(tolerance, limit), Preconditioning::None);
    check!(CgsSolver::new(tolerance, limit), Preconditioning::None);
    check!(QmrSolver::new(tolerance, limit), Preconditioning::None);
    check!(TfqmrSolver::new(tolerance, limit), Preconditioning::None);
}

#[test]
fn larger_budgets_never_worsen_scalar_answers() {
    let mut a = krylov_reference::fixtures()
        .find(|f| f.problem == "poisson")
        .unwrap();
    a.order = 256;
    let b = a.rhs();
    let tolerance = ResidualTolerance::new(0., 1e-16).unwrap();
    macro_rules! check {
        ($solver:ident $(, $p:expr)?) => {{
            let mut previous = f64::INFINITY;
            for budget in [50, 100, 200, 400, 800] {
                let result = $solver::new(tolerance, Budget::try_from(budget).unwrap()).solve_with_options(&a, &b, $($p,)? None).unwrap();
                let mut ax = vec![0.; a.rows()];
                a.apply_into(result.solution().as_slice(), &mut ax, &mut []).unwrap();
                let residual = ax.iter().fold(0_f64, |n, &v| n.hypot(1. - v));
                assert!(residual <= previous * (1. + 32. * f64::EPSILON), "{} budget={budget}: {residual:e} > {previous:e}", stringify!($solver));
                previous = residual;
            }
        }};
    }
    check!(CgSolver, Preconditioning::None);
    check!(BicgstabSolver, Preconditioning::None);
    check!(GmresSolver, Preconditioning::None);
    check!(LgmresSolver, Preconditioning::None);
    check!(GcrotSolver, Preconditioning::None);
    check!(MinresSolver, Preconditioning::None);
    check!(LsqrSolver);
    check!(LsmrSolver);
    check!(BicgSolver, Preconditioning::None);
    check!(CgsSolver, Preconditioning::None);
    check!(QmrSolver, Preconditioning::None);
    check!(TfqmrSolver, Preconditioning::None);
}

#[test]
fn dense_bicgstab_retains_accuracy_with_long_budget() {
    let a = krylov_reference::fixtures()
        .find(|f| f.problem == "poisson" && f.order == 4096)
        .unwrap();
    let n = a.rows();
    let side = 64;
    let mut values = vec![0.; n * n];
    for i in 0..n {
        values[i * n + i] = 4.;
        for j in [
            i.checked_sub(side),
            (i + side < n).then_some(i + side),
            (i % side > 0).then(|| i - 1),
            (i % side + 1 < side).then_some(i + 1),
        ]
        .into_iter()
        .flatten()
        {
            values[i * n + j] = -1.;
        }
    }
    let matrix = Matrix::from_vec(n, n, values).unwrap();
    for budget in [200, 5000] {
        let result = BicgstabSolver::new(
            ResidualTolerance::new(0., 1e-15).unwrap(),
            Budget::try_from(budget).unwrap(),
        )
        .solve_with_options(
            &matrix.view().try_finite().unwrap(),
            &a.rhs(),
            Preconditioning::None,
            None,
        )
        .unwrap();
        let mut ax = vec![0.; n];
        a.apply_into(result.solution().as_slice(), &mut ax, &mut [])
            .unwrap();
        let relative = ax.iter().fold(0_f64, |n, &v| n.hypot(1. - v)) / 64.;
        assert!(
            relative <= 1e-12,
            "budget={budget}, {relative:e}, {:?}",
            result.report()
        );
    }
}

fn poisson_csr(side: usize) -> scioxide_sparse::CsrMatrix<f64> {
    let n = side * side;
    let mut pointers = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        for j in [
            i.checked_sub(side),
            (i % side > 0).then(|| i - 1),
            Some(i),
            (i % side + 1 < side).then_some(i + 1),
            (i + side < n).then_some(i + side),
        ]
        .into_iter()
        .flatten()
        {
            columns.push(j);
            values.push(if i == j { 4. } else { -1. });
        }
        pointers.push(columns.len());
    }
    scioxide_sparse::CsrMatrix::from_raw(n, n, pointers, columns, values).unwrap()
}

#[test]
fn stationary_floors_and_budget_sweeps() {
    let matrix = poisson_csr(8);
    let a = StationaryMatrix::Csr(&matrix);
    let rhs = Vector::from_vec((0..64).map(|i| 0.1 + (i % 7) as f64 / 13.).collect());
    let tolerance = ResidualTolerance::new(0., 1e-30).unwrap();
    macro_rules! check {
        ($solver:expr, $operator:expr $(, $inverse:expr)?) => {{
            let mut previous = f64::INFINITY;
            for budget in [50, 100, 200, 400, 800, 4000] {
                let solver = ($solver)(Budget::try_from(budget).unwrap()).with_history(true);
                let mut w = solver.workspace(&IterativeWorkspaceRequest::matrix(a).unwrap()).unwrap();
                let mut x = Vector::zeros(64).unwrap();
                let report = { let mut evaluate_case = || solver.solve_into($operator, &rhs, $($inverse,)? x.view_mut(), &mut w, ExecutionContext::sequential()).unwrap(); evaluate_case() };
                let mut ax = vec![0.; 64];
                let mut scratch = vec![0.; matrix.apply_workspace_len().unwrap()];
                matrix.apply_into(x.as_slice(), &mut ax, &mut scratch).unwrap();
                let residual = ax.iter().zip(rhs.as_slice()).fold(0_f64, |n, (&ax, &b)| n.hypot(b - ax));
                assert!(residual <= previous * (1. + 32. * f64::EPSILON), "{} budget={budget}: {residual:e} > {previous:e}", stringify!($solver));
                assert!((residual - report.diagnostic_residual_norm().unwrap()).abs() <= 1e-13);
                assert!(w.last_accepted().unwrap().history().unwrap().iter().all(|&value| value >= report.diagnostic_residual_norm().unwrap()));
                if budget == 4000 { assert_eq!(report.termination(), LinearSolveTermination::Stagnation); }
                previous = residual;
            }
        }};
    }
    check!(|limit| JacobiSolver::new(tolerance, limit), a);
    check!(|limit| GaussSeidelSolver::new(tolerance, limit), a);
    check!(
        |limit| SorSolver::new(tolerance, limit, SorWeight::new(1.3).unwrap()),
        a
    );
    check!(
        |limit| RichardsonSolver::new(tolerance, limit, RelaxationWeight::new(0.25).unwrap()),
        &matrix,
        Preconditioning::None
    );
}

#[test]
fn stationary_large_poisson_does_not_falsely_stagnate() {
    let matrix = poisson_csr(128);
    let rhs = Vector::from_vec(vec![1.; 16384]);
    let result = JacobiSolver::new(
        ResidualTolerance::new(0., 1e-16).unwrap(),
        Budget::try_from(4000).unwrap(),
    )
    .solve_with_options(StationaryMatrix::Csr(&matrix), &rhs, None)
    .unwrap();
    assert_eq!(
        result.report().termination(),
        LinearSolveTermination::IterationLimit
    );
    assert!(result.report().diagnostic_residual_norm().unwrap() / 128. >= 0.249);
}

#[test]
fn coupled_blocks_keep_the_best_complete_block_across_budgets() {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut};
    let mut a = krylov_reference::fixtures()
        .find(|f| f.problem == "poisson")
        .unwrap();
    a.order = 256;
    let b: Vec<_> = (0..512)
        .map(|i| {
            if i < 256 {
                1.
            } else {
                ((i % 11) as f64 + 1.) / 13.
            }
        })
        .collect();
    let tolerance = ResidualTolerance::new(0., 1e-16).unwrap();
    macro_rules! check {
        ($solver:ident) => {{
            let mut previous = f64::INFINITY;
            for budget in [50, 100, 200, 400, 800] {
                let solver = $solver::new(tolerance, Budget::try_from(budget).unwrap());
                let mut workspace = solver
                    .workspace(
                        &IterativeWorkspaceRequest::operator(&a)
                            .unwrap()
                            .with_right_hand_sides(2),
                    )
                    .unwrap();
                let mut x = vec![0.; 512];
                let _ = solver
                    .solve_into(
                        &a,
                        BlockVectorView::new(&b, 256, 2).unwrap(),
                        Preconditioning::None,
                        BlockVectorViewMut::new(&mut x, 256, 2).unwrap(),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                let mut residual = 0_f64;
                for (col, (x, b)) in x.chunks_exact(256).zip(b.chunks_exact(256)).enumerate() {
                    let mut ax = vec![0.; 256];
                    a.apply_into(x, &mut ax, &mut []).unwrap();
                    let norm = ax.iter().zip(b).fold(0_f64, |n, (&ax, &b)| n.hypot(b - ax));
                    assert!(
                        (workspace
                            .last_accepted()
                            .unwrap()
                            .columns()
                            .unwrap()
                            .residuals()[col]
                            - norm)
                            .abs()
                            <= 1e-12
                    );
                    residual = residual.max(norm);
                }
                assert!(
                    residual <= previous * (1. + 32. * f64::EPSILON),
                    "{} budget={budget}: {residual:e} > {previous:e}",
                    stringify!($solver)
                );
                previous = residual;
            }
        }};
    }
    check!(BlockGmresSolver);
    check!(BlockLgmresSolver);
    check!(BlockGcrotSolver);
}

/// Finite-volume heat operator on an `n × n` grid with zero Dirichlet
/// boundaries: conductivity `high` in the central square and one elsewhere,
/// with harmonic means on the faces.
fn heat(n: usize, high: f64) -> scioxide_sparse::CsrMatrix<f64> {
    let conductivity = |i: usize, j: usize| {
        if (n / 4..3 * n / 4).contains(&i) && (n / 4..3 * n / 4).contains(&j) {
            high
        } else {
            1.
        }
    };
    let mut pointers = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let mut row = Vec::new();
            let mut diagonal = 0.;
            for (di, dj) in [(-1_isize, 0_isize), (0, -1), (0, 1), (1, 0)] {
                let (ii, jj) = (i as isize + di, j as isize + dj);
                let inside = (0..n as isize).contains(&ii) && (0..n as isize).contains(&jj);
                let neighbour = if inside {
                    conductivity(ii as usize, jj as usize)
                } else {
                    conductivity(i, j)
                };
                let face = 2. * conductivity(i, j) * neighbour / (conductivity(i, j) + neighbour);
                diagonal += face;
                if inside {
                    row.push((ii as usize * n + jj as usize, -face));
                }
            }
            row.push((i * n + j, diagonal));
            row.sort_by_key(|&(column, _)| column);
            for (column, value) in row {
                columns.push(column);
                values.push(value);
            }
            pointers.push(columns.len());
        }
    }
    scioxide_sparse::CsrMatrix::from_raw(n * n, n * n, pointers, columns, values).unwrap()
}

fn true_residual(a: &scioxide_sparse::CsrMatrix<f64>, b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.; b.len()];
    let mut scratch = vec![0.; LinearOperator::<f64>::apply_workspace_len(a).unwrap()];
    LinearOperator::<f64>::apply_into(a, x, &mut ax, &mut scratch).unwrap();
    ax.iter().zip(b).fold(0_f64, |n, (&ax, &b)| n.hypot(b - ax))
}

#[test]
fn cg_floor_follows_the_magnitudes_of_the_operator_products() {
    // IC(0) smooths the Krylov directions, so `‖A v‖ / ‖v‖` along the
    // recurrence underestimates `‖A‖` by four orders; the residual still
    // stops improving near `ε ‖ |A| |x| ‖` within a few hundred steps.
    let n = 32;
    let a = heat(n, 1e8);
    let h = 1. / (n + 1) as f64;
    let b = vec![h * h; n * n];
    let ic = scioxide_sparse::IncompleteFactor::ic0(&a).unwrap();
    let result = CgSolver::new(
        ResidualTolerance::new(0., 1e-12).unwrap(),
        Budget::try_from(4000).unwrap(),
    )
    .solve_with_options(&a, &b[..], Preconditioning::Inverse(&ic), None)
    .unwrap();
    let report = result.report();
    assert_eq!(report.termination(), LinearSolveTermination::Stagnation);
    assert!(report.iterations() <= 300, "{report:?}");
    let residual = true_residual(&a, &b, result.solution().as_slice());
    assert!((report.diagnostic_residual_norm().unwrap() - residual).abs() <= 1e-3 * residual);
    assert!(report.true_residual().unwrap().scaled_norm() <= 1e-5);

    // Without preconditioning, a tolerance a hundred times above the attainable
    // residual converges instead of stopping at a model of `n ε ‖A‖ ‖x‖`.
    let n = 24;
    let a = heat(n, 1e6);
    let b: Vec<f64> = (0..n * n)
        .map(|i| 1. + 0.5 * ((i * 7919 % 101) as f64 / 101. - 0.5))
        .collect();
    let tolerance = ResidualTolerance::new(0., 1e-7).unwrap();
    let result = CgSolver::new(tolerance, Budget::try_from(20000).unwrap())
        .solve_with_options(&a, &b[..], Preconditioning::None, None)
        .unwrap();
    assert_eq!(
        result.report().termination(),
        LinearSolveTermination::Converged,
        "{:?}",
        result.report()
    );
    let reference = b.iter().fold(0_f64, |n, &b| n.hypot(b));
    assert!(true_residual(&a, &b, result.solution().as_slice()) <= 1e-7 * reference);
}

#[test]
fn richardson_with_an_exact_inverse_stagnates_at_the_floor() {
    // `A = [[p, −q], [−q, p]]` with eigenvalues 1 and 2·10⁸ − 1: `M⁻¹ = A⁻¹`
    // reaches the rounding floor `ε ‖ |A| |x| ‖ ≈ 4·10⁻⁸` in one step, far
    // above `n ε ‖b‖`.
    let (p, q) = (1e8_f64, 1e8 - 1.);
    let determinant = p * p - q * q;
    let a = Matrix::from_vec(2, 2, vec![p, -q, -q, p]).unwrap();
    let inverse =
        scioxide_core::ClosurePreconditioner::<f64, _>::new(2, move |r: &[f64], z: &mut [f64]| {
            z[0] = (p * r[0] + q * r[1]) / determinant;
            z[1] = (q * r[0] + p * r[1]) / determinant;
            Ok(())
        });
    let b = [1., 0.3];
    let result = RichardsonSolver::new(
        ResidualTolerance::new(0., 1e-15).unwrap(),
        Budget::try_from(1000).unwrap(),
        RelaxationWeight::one(),
    )
    .with_history(true)
    .solve_with_options(
        &a.view().try_finite().unwrap(),
        &b[..],
        Preconditioning::Inverse(&inverse),
        None,
    )
    .unwrap();
    let report = result.report();
    assert_eq!(report.termination(), LinearSolveTermination::Stagnation);
    assert!(report.iterations() <= 100, "{report:?}");
    let x = result.solution().as_slice();
    let residual = (b[0] - p * x[0] + q * x[1]).hypot(b[1] + q * x[0] - p * x[1]);
    assert!(residual <= 1e-7, "{residual:e}");
    let history = result.history().unwrap();
    let minimum = history.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(report.diagnostic_residual_norm(), Some(minimum));
}

#[test]
fn bicgstab_history_covers_a_stop_after_the_half_step() {
    // The half-step residual `s` of the 10 × 10 second-difference matrix meets
    // `rtol = 10⁻¹⁵` only recursively; the verified half step ends the solve.
    let n = 10;
    let mut entries = vec![0.; n * n];
    for i in 0..n {
        entries[i * n + i] = 2.;
        if i + 1 < n {
            entries[i * n + i + 1] = -1.;
            entries[(i + 1) * n + i] = -1.;
        }
    }
    let a = Matrix::from_vec(n, n, entries).unwrap();
    let result = BicgstabSolver::new(
        ResidualTolerance::new(0., 1e-15).unwrap(),
        Budget::try_from(200).unwrap(),
    )
    .with_history(true)
    .solve_with_options(
        &a.view().try_finite().unwrap(),
        &[1.; 10][..],
        Preconditioning::None,
        None,
    )
    .unwrap();
    let report = result.report();
    assert_eq!(report.termination(), LinearSolveTermination::Stagnation);
    let history = result.history().unwrap();
    assert_eq!(history.len(), report.iterations() + 1);
    assert_eq!(history.last().copied(), report.diagnostic_residual_norm());
}

/// Convection-dominated central differences make BiCGSTAB's updated residuals
/// grow transiently far beyond `‖b‖`; the verified residual that replaces them
/// at the checkpoints lets the recurrence recover, so the rounding floor of the
/// best iterate must not end these solves early.
#[test]
fn bicgstab_recovers_from_transient_residual_growth() -> Result<(), scioxide_core::SciOxideError> {
    let (n, h) = (100_usize, 1.0 / 101.0);
    let b = Vector::from_vec(vec![1.0; n]);
    let mut recovered = 0;
    for peclet in 190..=230 {
        let peclet = f64::from(peclet);
        let matrix = TridiagonalMatrix::new(
            vec![-1.0 / (h * h) - peclet / (2.0 * h); n - 1],
            vec![2.0 / (h * h); n],
            vec![-1.0 / (h * h) + peclet / (2.0 * h); n - 1],
        )?;
        let result =
            BicgstabSolver::new(ResidualTolerance::new(0.0, 1e-10)?, Budget::try_from(3000)?)
                .with_history(true)
                .solve_with_options(&matrix.view(), &b, Preconditioning::None, None)?;
        let termination = result.report().termination();
        assert!(
            matches!(
                termination,
                LinearSolveTermination::Converged
                    | LinearSolveTermination::Breakdown(
                        LinearSolveBreakdown::RecurrenceDenominator
                    )
            ),
            "Pe = {peclet}: {termination:?}"
        );
        let peak = result
            .history()
            .unwrap_or(&[])
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        if termination == LinearSolveTermination::Converged && peak > 1e16 * (n as f64).sqrt() {
            recovered += 1;
        }
    }
    assert!(recovered > 0, "no solve recovered from residual growth");
    Ok(())
}
