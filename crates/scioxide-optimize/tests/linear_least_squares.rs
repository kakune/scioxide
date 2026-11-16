#![allow(clippy::unwrap_used)]
use scioxide_core::{Bounds, ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{BoundedLinearLeastSquares, LinearLeastSquaresProblem, Nnls};
#[test]
fn analytic_dense_and_operator() {
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let b = [-1., 2., 1.];
    for p in [
        LinearLeastSquaresProblem::from_dense(&a, &b).unwrap(),
        LinearLeastSquaresProblem::new(&a.view().try_finite().unwrap(), &b).unwrap(),
    ] {
        let r = Nnls::<f64>::default().solve(&p).unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!(r.x()[0].abs() < 1e-7);
        assert!((r.x()[1] - 1.5).abs() < 1e-7);
        let bounds = Bounds::new(vec![(Some(0.), Some(1.)), (Some(0.), Some(1.))]).unwrap();
        let p = p.with_bounds(&bounds).unwrap();
        let r = BoundedLinearLeastSquares::<f64>::default()
            .solve(&p)
            .unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!(r.x()[0].abs() < 1e-7);
        assert!((r.x()[1] - 1.).abs() < 1e-7);
    }
}
#[test]
fn dependent_and_fixed() {
    let a = Matrix::from_vec(2, 3, vec![1., 1., 0., 2., 2., 0.]).unwrap();
    let b = [3., 6.];
    let p = LinearLeastSquaresProblem::from_dense(&a, &b).unwrap();
    let r = Nnls::<f64>::default().solve(&p).unwrap();
    assert!(r.is_success(), "{r:?}");
    assert!((r.x()[0] + r.x()[1] - 3.).abs() < 1e-7);
    let bounds = Bounds::new(vec![
        (Some(1.), Some(1.)),
        (None, Some(1.)),
        (Some(2.), None),
    ])
    .unwrap();
    let p = p.with_bounds(&bounds).unwrap();
    let r = BoundedLinearLeastSquares::<f64>::default()
        .solve(&p)
        .unwrap();
    assert!(r.is_success(), "{r:?}");
    assert_eq!(r.x(), &[1., 1., 2.]);
}
#[test]
fn invalid_rhs() {
    let a = Matrix::<f64>::zeros(2, 1).unwrap();
    assert!(matches!(
        (LinearLeastSquaresProblem::new(&a.view().try_finite().unwrap(), &[1.]))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
    ));
}

#[test]
fn prepared_first_and_repeated_solves_execute() {
    for rows in [1, 4] {
        let a = Matrix::from_vec(
            rows,
            3,
            (0..rows).flat_map(|i| [1.0, (i + 1) as f64, 1.0]).collect(),
        )
        .unwrap();
        let b = vec![2.; rows];
        let bounds = Bounds::new(vec![(Some(0.), Some(3.)); 3]).unwrap();
        let sparse = scioxide_sparse::CsrMatrix::from_raw(
            rows,
            3,
            (0..=rows).map(|row| row * 3).collect(),
            (0..rows).flat_map(|_| 0..3).collect(),
            a.as_slice().to_vec(),
        )
        .unwrap();
        for p in [
            LinearLeastSquaresProblem::from_dense(&a, &b).unwrap(),
            LinearLeastSquaresProblem::new(&a.view().try_finite().unwrap(), &b).unwrap(),
            LinearLeastSquaresProblem::new(&sparse, &b).unwrap(),
        ] {
            let solver = Nnls::<f64>::default();
            let mut w = solver.workspace(&p).unwrap();
            let mut r = reserved(rows, 3);
            {
                solver
                    .solve_into(&p, &mut r, &mut w, ExecutionContext::sequential())
                    .unwrap();
            };
            assert!(r.is_success(), "{r:?}");
            {
                solver
                    .solve_into(&p, &mut r, &mut w, ExecutionContext::sequential())
                    .unwrap();
            };
            let p = p.with_bounds(&bounds).unwrap();
            let solver = BoundedLinearLeastSquares::<f64>::default();
            let mut w = solver.workspace(&p).unwrap();
            {
                solver
                    .solve_into(&p, &mut r, &mut w, ExecutionContext::sequential())
                    .unwrap();
            };
            assert!(r.is_success(), "{r:?}");
        }
    }
}

// Independent oracle: exact rational elimination on every face. Forming a Gram
// matrix is safe ONLY here, because arithmetic is exact and this is test code.
fn enumerated_objective(a: &[i64], b: &[i64], bounds: &[(Option<i64>, Option<i64>)]) -> f64 {
    use num::{BigRational as Q, ToPrimitive, Zero};
    let q = |x: i64| Q::from_integer(x.into());
    let n = bounds.len();
    let m = b.len();
    let mut best: Option<Q> = None;
    for code in 0..3_usize.pow(n as u32) {
        let mut code = code;
        let mut x = vec![q(0); n];
        let mut free = Vec::new();
        let mut valid = true;
        for (j, &(lo, hi)) in bounds.iter().enumerate() {
            match code % 3 {
                0 => free.push(j),
                1 => {
                    if let Some(v) = lo {
                        x[j] = q(v);
                    } else {
                        valid = false;
                    }
                }
                _ => {
                    if let Some(v) = hi {
                        x[j] = q(v);
                    } else {
                        valid = false;
                    }
                }
            }
            code /= 3;
        }
        if !valid {
            continue;
        }
        let k = free.len();
        let residual: Vec<_> = (0..m)
            .map(|i| q(b[i]) - (0..n).map(|j| q(a[i * n + j]) * &x[j]).sum::<Q>())
            .collect();
        let mut aug: Vec<Vec<Q>> = free
            .iter()
            .map(|&j| {
                free.iter()
                    .map(|&l| (0..m).map(|i| q(a[i * n + j]) * q(a[i * n + l])).sum())
                    .chain(core::iter::once(
                        (0..m).map(|i| q(a[i * n + j]) * &residual[i]).sum(),
                    ))
                    .collect()
            })
            .collect();
        for col in 0..k {
            let Some(pivot) = (col..k).find(|&r| !aug[r][col].is_zero()) else {
                valid = false;
                break;
            };
            aug.swap(pivot, col);
            let d = aug[col][col].clone();
            for v in &mut aug[col][col..] {
                *v /= d.clone();
            }
            let pivot_row = aug[col].clone();
            for (row, entries) in aug.iter_mut().enumerate() {
                if row == col {
                    continue;
                }
                let d = entries[col].clone();
                for (v, pivot) in entries[col..].iter_mut().zip(&pivot_row[col..]) {
                    *v -= &d * pivot;
                }
            }
        }
        if !valid {
            continue;
        }
        for (j, &index) in free.iter().enumerate() {
            x[index] = aug[j][k].clone();
        }
        if x.iter()
            .zip(bounds)
            .any(|(x, &(l, u))| l.is_some_and(|l| *x < q(l)) || u.is_some_and(|u| *x > q(u)))
        {
            continue;
        }
        let objective: Q = (0..m)
            .map(|i| {
                let r = (0..n).map(|j| q(a[i * n + j]) * &x[j]).sum::<Q>() - q(b[i]);
                &r * &r / q(2)
            })
            .sum();
        if best.as_ref().is_none_or(|v| objective < *v) {
            best = Some(objective);
        }
    }
    best.unwrap().to_f64().unwrap()
}
fn check_kkt(
    a: &[f64],
    b: &[f64],
    bounds: &[(Option<f64>, Option<f64>)],
    x: &[f64],
    tolerance: f64,
    gradient_tolerance: (f64, f64),
) {
    let n = x.len();
    let residual: Vec<_> = a
        .chunks(n)
        .zip(b)
        .map(|(row, b)| row.iter().zip(x).map(|(a, x)| a * x).sum::<f64>() - b)
        .collect();
    let gradients: Vec<f64> = (0..n)
        .map(|j| a.chunks(n).zip(&residual).map(|(row, r)| row[j] * r).sum())
        .collect();
    let objective = 0.5 * residual.iter().map(|r| r * r).sum::<f64>();
    let coordinate = x.iter().fold(1_f64, |a, &v| a.max(v.abs()));
    let scale = gradients
        .iter()
        .fold((objective.abs() / coordinate).max(1.), |a, &g| {
            a.max(g.abs())
        });
    let gradient_limit = tolerance.max(gradient_tolerance.0 + gradient_tolerance.1 * scale);
    for (j, (&x, &(l, u))) in x.iter().zip(bounds).enumerate() {
        assert!(l.is_none_or(|l| x >= l - tolerance) && u.is_none_or(|u| x <= u + tolerance));
        let g = a
            .chunks(n)
            .zip(&residual)
            .map(|(row, r)| row[j] * r)
            .sum::<f64>();
        if l.is_some() && l == u {
            continue;
        }
        if l.is_some_and(|l| x - l < tolerance) {
            assert!(g >= -gradient_limit, "lower gradient {g}");
        } else if u.is_some_and(|u| u - x < tolerance) {
            assert!(g <= gradient_limit, "upper gradient {g}");
        } else {
            assert!(g.abs() < gradient_limit, "free gradient {g}, x {x}");
        }
    }
}
macro_rules! enumeration {
    ($test:ident,$s:ty,$tol:expr) => {
        #[test]
        fn $test() {
            for seed in 0..24_i64 {
                let a: Vec<i64> = (0..12)
                    .map(|i| ((i * 7 + seed * 3 + i * i + seed * i) % 9) - 4)
                    .collect();
                let b: Vec<i64> = (0..4).map(|i| ((seed * 5 + i * 3) % 11) - 5).collect();
                let af: Vec<f64> = a.iter().map(|&x| x as f64).collect();
                let bf: Vec<f64> = b.iter().map(|&x| x as f64).collect();
                let matrix = Matrix::from_vec(4, 3, a.iter().map(|&x| x as $s).collect()).unwrap();
                let rhs: Vec<$s> = b.iter().map(|&x| x as $s).collect();
                for nonnegative in [true, false] {
                    let bounds = if nonnegative {
                        vec![(Some(0), None); 3]
                    } else {
                        vec![
                            (Some(-1), Some(2)),
                            (None, Some(1)),
                            (Some(0), Some(if seed % 3 == 0 { 0 } else { 2 })),
                        ]
                    };
                    let expected = enumerated_objective(&a, &b, &bounds);
                    let real_bounds = Bounds::new(
                        bounds
                            .iter()
                            .map(|&(l, u)| (l.map(|v| v as $s), u.map(|v| v as $s)))
                            .collect(),
                    )
                    .unwrap();
                    let double_bounds: Vec<_> = bounds
                        .iter()
                        .map(|&(l, u)| (l.map(|v| v as f64), u.map(|v| v as f64)))
                        .collect();
                    for p in [
                        LinearLeastSquaresProblem::from_dense(&matrix, &rhs).unwrap(),
                        LinearLeastSquaresProblem::new(&matrix.view().try_finite().unwrap(), &rhs)
                            .unwrap(),
                    ] {
                        let r = if nonnegative {
                            Nnls::<$s>::default().solve(&p)
                        } else {
                            BoundedLinearLeastSquares::<$s>::default()
                                .solve(&p.with_bounds(&real_bounds).unwrap())
                        }
                        .unwrap();
                        assert!(r.is_success(), "seed {seed}, nnls {nonnegative}: {r:?}");
                        assert!(
                            (r.objective() as f64 - expected).abs() < $tol * (1. + expected),
                            "seed {seed}: {r:?} expected {expected}"
                        );
                        check_kkt(
                            &af,
                            &bf,
                            &double_bounds,
                            &r.x().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                            $tol,
                            (
                                scioxide_core::GradientTolerance::<$s>::default().absolute() as f64,
                                scioxide_core::GradientTolerance::<$s>::default().relative() as f64,
                            ),
                        );
                    }
                }
            }
        }
    };
}
enumeration!(enumerated_f64, f64, 2e-6);
enumeration!(enumerated_f32, f32, 3e-3);

fn check_regression<S: RealScalar + NumericScalar<Real = S>>(
    coefficients: &[f64],
    rhs: &[f64],
    upper: Option<f64>,
    expected: &[f64],
) {
    use scioxide_optimize::OptimizeTermination;
    let cv = |x| S::try_from_f64_lossy(x).unwrap();
    let tolerance = if S::MANTISSA_DIGITS <= 24 { 3e-3 } else { 1e-7 };
    let matrix = Matrix::from_vec(
        rhs.len(),
        expected.len(),
        coefficients.iter().copied().map(cv).collect(),
    )
    .unwrap();
    let b: Vec<_> = rhs.iter().copied().map(cv).collect();
    let bounds = Bounds::new(vec![(Some(S::ZERO), upper.map(cv)); expected.len()]).unwrap();
    let finite = matrix.view().try_finite().unwrap();
    for problem in [
        LinearLeastSquaresProblem::from_dense(&matrix, &b).unwrap(),
        LinearLeastSquaresProblem::new(&finite, &b).unwrap(),
    ] {
        let result = if upper.is_some() {
            BoundedLinearLeastSquares::default().solve(&problem.with_bounds(&bounds).unwrap())
        } else {
            Nnls::default().solve(&problem)
        }
        .unwrap();
        assert_eq!(
            result.termination(),
            OptimizeTermination::Converged,
            "{result:?}"
        );
        assert!(result.iterations() < 20, "{result:?}");
        let point: Vec<_> = result
            .x()
            .iter()
            .map(|v| v.try_to_f64_exact().unwrap())
            .collect();
        for (&x, &expected) in point.iter().zip(expected) {
            assert!(
                (x - expected).abs() <= tolerance,
                "{result:?}, expected {expected}"
            );
        }
        let objective = 0.5
            * coefficients
                .chunks(expected.len())
                .zip(rhs)
                .map(|(row, b)| {
                    let residual = row.iter().zip(expected).map(|(a, x)| a * x).sum::<f64>() - b;
                    residual * residual
                })
                .sum::<f64>();
        assert!(
            (result.objective().try_to_f64_exact().unwrap() - objective).abs()
                <= tolerance * (1.0 + objective)
        );
        check_kkt(
            coefficients,
            rhs,
            &vec![(Some(0.0), upper); expected.len()],
            &point,
            tolerance,
            (
                scioxide_core::GradientTolerance::<S>::default()
                    .absolute()
                    .try_to_f64_exact()
                    .unwrap(),
                scioxide_core::GradientTolerance::<S>::default()
                    .relative()
                    .try_to_f64_exact()
                    .unwrap(),
            ),
        );
    }
}

fn unchanged_passive_resolves<S: RealScalar + NumericScalar<Real = S>>() {
    // The unconstrained solution (-2, -3.5) clips to the accepted zero point.
    // Neither bound variable may be blocked by that initialization.
    check_regression::<S>(
        &[2., -2., 3., -2., 0., 0.],
        &[3., 1., -1.],
        None,
        &[9. / 13., 0.],
    );

    let centers = [430., 470., 520., 580.];
    let widths = [25., 30., 25., 35.];
    let truth = [0.8, 0., 0.5, 1.2];
    let a: Vec<_> = (0..40)
        .flat_map(|i| {
            centers.into_iter().zip(widths).map(move |(center, width)| {
                let t = (400. + 6. * i as f64 - center) / width;
                (-0.5 * t * t).exp()
            })
        })
        .collect();
    let b: Vec<_> = a
        .chunks(4)
        .enumerate()
        .map(|(i, row)| {
            row.iter().zip(truth).map(|(a, x)| a * x).sum::<f64>()
                + 0.02 * (1.3 * i as f64 + 0.4).sin()
        })
        .collect();
    for upper in [0.4, 0.6, 0.7] {
        // At these ceilings coordinates 0, 2 and 3 are upper-active. Solve the
        // remaining scalar equation independently of the active-set engine.
        let numerator: f64 = a
            .chunks(4)
            .zip(&b)
            .map(|(row, b)| row[1] * (b - upper * (row[0] + row[2] + row[3])))
            .sum();
        let denominator: f64 = a.chunks(4).map(|row| row[1] * row[1]).sum();
        check_regression::<S>(
            &a,
            &b,
            Some(upper),
            &[upper, numerator / denominator, upper, upper],
        );
    }
}

#[test]
fn unchanged_passive_resolves_return_to_pricing_f64() {
    unchanged_passive_resolves::<f64>();
}
#[test]
fn unchanged_passive_resolves_return_to_pricing_f32() {
    unchanged_passive_resolves::<f32>();
}

fn seeded_enumeration<S: RealScalar + NumericScalar<Real = S>>() {
    use scioxide_random::{BitGenerator, SeedableRng, SplitMix64};
    let mut rng = SplitMix64::from_seed(0x2393_2394);
    let cv = |x: i64| S::try_from_f64_lossy(x as f64).unwrap();
    let tolerance = if S::MANTISSA_DIGITS <= 24 { 3e-3 } else { 2e-6 };
    for n in 2..=4 {
        let m = n + 2;
        for case in 0..32 {
            let a: Vec<_> = (0..m * n)
                .map(|_| (rng.next_u64().unwrap() % 9) as i64 - 4)
                .collect();
            let b: Vec<_> = (0..m)
                .map(|_| (rng.next_u64().unwrap() % 17) as i64 - 8)
                .collect();
            let upper = (rng.next_u64().unwrap() % 4) as i64 + 1;
            let matrix = Matrix::from_vec(m, n, a.iter().copied().map(cv).collect()).unwrap();
            let rhs: Vec<_> = b.iter().copied().map(cv).collect();
            let problem = LinearLeastSquaresProblem::from_dense(&matrix, &rhs).unwrap();
            for nonnegative in [true, false] {
                let endpoints = vec![(Some(0), (!nonnegative).then_some(upper)); n];
                let expected = enumerated_objective(&a, &b, &endpoints);
                let bounds = Bounds::new(
                    endpoints
                        .iter()
                        .map(|&(l, u)| (l.map(cv), u.map(cv)))
                        .collect(),
                )
                .unwrap();
                let result = if nonnegative {
                    Nnls::default().solve(&problem)
                } else {
                    BoundedLinearLeastSquares::default()
                        .solve(&problem.with_bounds(&bounds).unwrap())
                }
                .unwrap();
                assert_eq!(
                    result.termination(),
                    scioxide_optimize::OptimizeTermination::Converged,
                    "n {n}, case {case}, NNLS {nonnegative}: {result:?}"
                );
                assert!(
                    (result.objective().try_to_f64_exact().unwrap() - expected).abs()
                        <= tolerance * (1. + expected),
                    "n {n}, case {case}: {result:?}, expected {expected}"
                );
                check_kkt(
                    &a.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    &b.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    &endpoints
                        .iter()
                        .map(|&(l, u)| (l.map(|v| v as f64), u.map(|v| v as f64)))
                        .collect::<Vec<_>>(),
                    &result
                        .x()
                        .iter()
                        .map(|v| v.try_to_f64_exact().unwrap())
                        .collect::<Vec<_>>(),
                    tolerance,
                    (
                        scioxide_core::GradientTolerance::<S>::default()
                            .absolute()
                            .try_to_f64_exact()
                            .unwrap(),
                        scioxide_core::GradientTolerance::<S>::default()
                            .relative()
                            .try_to_f64_exact()
                            .unwrap(),
                    ),
                );
            }
        }
    }
}
#[test]
fn seeded_active_sets_match_exact_enumeration_f64() {
    seeded_enumeration::<f64>();
}
#[test]
fn seeded_active_sets_match_exact_enumeration_f32() {
    seeded_enumeration::<f32>();
}

struct Observed<'a> {
    a: &'a Matrix<f64>,
    calls: std::cell::Cell<usize>,
    fail: usize,
    scratch: usize,
}
impl scioxide_core::LinearOperator<f64> for Observed<'_> {
    fn rows(&self) -> usize {
        self.a.rows()
    }
    fn cols(&self) -> usize {
        self.a.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], w: &mut [f64]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        assert!(w.len() >= self.scratch);
        if self.calls.get() == self.fail {
            return Err(SciOxideError::not_implemented(
                "deliberate provider failure",
            ));
        }
        scioxide_core::LinearOperator::apply_into(&self.a.view().try_finite()?, x, y, w)
    }
}
impl scioxide_core::AdjointOperator<f64> for Observed<'_> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        assert!(w.len() >= self.scratch);
        if self.calls.get() == self.fail {
            return Err(SciOxideError::not_implemented(
                "deliberate provider failure",
            ));
        }
        scioxide_core::AdjointOperator::apply_adjoint_into(&self.a.view().try_finite()?, x, y, w)
    }
}
#[test]
fn exact_parent_budgets_and_transactional_failures() {
    use scioxide_core::{
        Budget, BudgetKind, EvaluationRole, ExecutionContext, SolveBudgets, SolveState,
    };
    use scioxide_optimize::OptimizeTermination;
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let b = [1., 2., 3.];
    for limit in 1..35 {
        let op = Observed {
            a: &a,
            calls: 0.into(),
            fail: usize::MAX,
            scratch: 7,
        };
        let p = LinearLeastSquaresProblem::new(&op, &b).unwrap();
        let solver = BoundedLinearLeastSquares::<f64>::default();
        let mut w = solver.workspace(&p).unwrap();
        let mut out = reserved(3, 2);
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut state = SolveState::new();
        let mut ctx = state
            .begin_solve(ExecutionContext::sequential(), parent(1), &mut budgets)
            .unwrap();
        let result = solver.solve_into(&p, &mut out, &mut w, &mut ctx);
        assert_eq!(
            ctx.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::LinearOperator),
            op.calls.get()
        );
        assert!(ctx.counts().total().unwrap().total_providers().unwrap() <= limit);
        if limit < 2 {
            assert!(result.is_err());
            assert_eq!(out.termination(), OptimizeTermination::NotStarted);
        } else {
            result.unwrap();
            assert_eq!(out.counts(), w.last_attempt().unwrap().counts());
            assert!(
                out.is_success() || out.termination() == OptimizeTermination::EvaluationLimit,
                "{out:?}"
            );
        }
    }
    for fail in 1..15 {
        let op = Observed {
            a: &a,
            calls: 0.into(),
            fail,
            scratch: 0,
        };
        let p = LinearLeastSquaresProblem::new(&op, &b).unwrap();
        let solver = Nnls::<f64>::default();
        let mut w = solver.workspace(&p).unwrap();
        let mut out = reserved(3, 2);
        assert!(
            solver
                .solve_into(&p, &mut out, &mut w, ExecutionContext::sequential())
                .is_err()
        );
        assert_eq!(out.termination(), OptimizeTermination::NotStarted);
        assert!(out.x().is_empty());
        assert_eq!(
            w.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Failed
        );
    }
}
#[test]
fn empty_dimensions_output_growth_and_mismatched_workspace() {
    for (m, n) in [(0, 0), (0, 3), (3, 0)] {
        let a = Matrix::<f64>::zeros(m, n).unwrap();
        let b = vec![2.; m];
        let p = LinearLeastSquaresProblem::from_dense(&a, &b).unwrap();
        for r in [
            Nnls::default().solve(&p).unwrap(),
            BoundedLinearLeastSquares::default().solve(&p).unwrap(),
        ] {
            assert!(r.is_success());
            assert_eq!(r.objective(), 2. * m as f64);
            assert_eq!(r.x().len(), n);
        }
    }
    let a = Matrix::from_vec(2, 1, vec![1., 1.]).unwrap();
    let p = LinearLeastSquaresProblem::from_dense(&a, &[1., 1.]).unwrap();
    let solver = Nnls::<f64>::default();
    let mut w = solver.workspace(&p).unwrap();
    let mut out = reserved(1, 1);
    solver
        .solve_into(&p, &mut out, &mut w, ExecutionContext::sequential())
        .unwrap();
    assert_eq!((out.x().len(), out.residual().len()), (1, 2));
    let op = Observed {
        a: &a,
        calls: 0.into(),
        fail: usize::MAX,
        scratch: 0,
    };
    let p = LinearLeastSquaresProblem::new(&op, &[1., 1.]).unwrap();
    let wide = Matrix::from_vec(3, 1, vec![1., 1., 1.]).unwrap();
    let other = LinearLeastSquaresProblem::from_dense(&wide, &[1., 1., 1.]).unwrap();
    let calls = op.calls.get();
    let saved = out.x().to_vec();
    let mut mismatched = solver.workspace(&other).unwrap();
    assert!(
        solver
            .solve_into(
                &p,
                &mut out,
                &mut mismatched,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(op.calls.get(), calls);
    assert_eq!(out.x(), saved);
    assert_eq!(
        mismatched.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
}
#[test]
fn injected_executor_and_cancelled_epoch() {
    use scioxide_core::{CancellationToken, ExecutionPlan, Executor, PartitionFn};
    struct Caller;
    impl Executor for Caller {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::not_implemented(
                "sequential kernels should not dispatch",
            ))
        }
    }
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let p = LinearLeastSquaresProblem::from_dense(&a, &[1., 2., 3.]).unwrap();
    let solver = Nnls::<f64>::default();
    let mut w = solver.workspace(&p).unwrap();
    let mut expected = reserved(3, 2);
    solver
        .solve_into(&p, &mut expected, &mut w, ExecutionContext::sequential())
        .unwrap();
    for cancelled in [false, true] {
        let token = CancellationToken::new();
        if cancelled {
            token.cancel();
        }
        let mut out = reserved(3, 2);
        let r = solver
            .solve_into(
                &p,
                &mut out,
                &mut w,
                ExecutionContext::new(&Caller).with_cancellation(&token),
            )
            .map(|()| out);
        if cancelled {
            assert!(matches!(
                (r).as_ref().map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::ExecutionStopped { .. })
            ));
            assert_eq!(
                w.last_attempt().unwrap().outcome(),
                scioxide_core::AttemptOutcome::PreflightRejected
            );
        } else {
            let r = r.unwrap();
            assert_eq!(r.x(), expected.x());
            assert_eq!(r.counts(), expected.counts());
        }
    }
}

#[test]
fn sparse_owned_borrowed_and_dense_views() {
    use scioxide_sparse::CsrMatrix;
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let view = a.view();
    let sparse = CsrMatrix::from_raw(
        3,
        2,
        vec![0, 1, 2, 4],
        vec![0, 1, 0, 1],
        vec![1., 1., 1., 1.],
    )
    .unwrap();
    let sparse_view = sparse.view();
    let b = [-1., 2., 1.];
    let bounds = Bounds::new(vec![(Some(0.), Some(1.)); 2]).unwrap();
    for p in [
        LinearLeastSquaresProblem::from_view(&view, &b).unwrap(),
        LinearLeastSquaresProblem::new(&sparse, &b).unwrap(),
        LinearLeastSquaresProblem::new(&sparse_view, &b).unwrap(),
    ] {
        let r = Nnls::<f64>::default().solve(&p).unwrap();
        assert!(r.is_success());
        assert!((r.x()[1] - 1.5).abs() < 1e-7);
        let r = BoundedLinearLeastSquares::<f64>::default()
            .solve(&p.with_bounds(&bounds).unwrap())
            .unwrap();
        assert!(r.is_success());
        assert!((r.x()[1] - 1.).abs() < 1e-7);
    }
}
#[test]
fn dependent_rank_reports_original_columns_and_scale_stress() {
    let a = Matrix::from_vec(2, 3, vec![1., 1., 0., 2., 2., 0.]).unwrap();
    let p = LinearLeastSquaresProblem::from_dense(&a, &[3., 6.]).unwrap();
    let r = BoundedLinearLeastSquares::<f64>::default()
        .solve(&p)
        .unwrap();
    assert!(r.is_success(), "{r:?}");
    assert_eq!(r.rank().unwrap().rank(), 1);
    for scale in [1e-100_f64, 1., 1e100] {
        use scioxide_core::{FeasibilityTolerance, GradientTolerance, ResidualTolerance};
        use scioxide_optimize::LinearLeastSquaresOptions;
        let tolerance = if scale < 1. {
            1e-210
        } else if scale > 1. {
            1e186
        } else {
            1e-12
        };
        let options = LinearLeastSquaresOptions::default().with_tolerances(
            GradientTolerance::new(tolerance, 0.).unwrap(),
            FeasibilityTolerance::default(),
            ResidualTolerance::default(),
        );
        let a = Matrix::from_vec(2, 2, vec![scale, 0., 0., scale]).unwrap();
        let b = [scale, -scale];
        let p = LinearLeastSquaresProblem::from_dense(&a, &b).unwrap();
        let r = Nnls::<f64>::new(options).solve(&p).unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!((r.x()[0] - 1.).abs() < 1e-10);
    }
}
#[test]
fn iteration_limits_do_not_overspend() {
    use scioxide_core::Budget;
    use scioxide_optimize::LinearLeastSquaresOptions;
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let p = LinearLeastSquaresProblem::new(&a_finite, &[1., 2., 3.]).unwrap();
    for limit in 1..12 {
        let solver = Nnls::<f64>::new(
            LinearLeastSquaresOptions::default()
                .with_max_iterations(Budget::try_from(limit).unwrap()),
        );
        let r = solver.solve(&p).unwrap();
        assert!(r.iterations() <= limit);
        assert!(
            r.is_success()
                || r.termination() == scioxide_optimize::OptimizeTermination::IterationLimit,
            "{r:?}"
        );
    }
}

#[test]
fn f32_sparse_views_and_transposed_dense_layout() {
    use scioxide_sparse::CsrMatrix;
    fn scalar_defaults<S: scioxide_core::RealScalar>() {
        let _: Nnls<S> = Nnls::default();
        let _: BoundedLinearLeastSquares<S> = BoundedLinearLeastSquares::default();
    }
    scalar_defaults::<f32>();
    scalar_defaults::<f64>();
    let storage = Matrix::from_vec(2, 3, vec![1_f32, 0., 1., 0., 1., 1.]).unwrap();
    let dense = storage.view().transpose();
    let sparse =
        CsrMatrix::from_raw(3, 2, vec![0, 1, 2, 4], vec![0, 1, 0, 1], vec![1_f32; 4]).unwrap();
    let sparse_view = sparse.view();
    let rhs = [-1_f32, 2., 1.];
    let bounds = Bounds::new(vec![(Some(0_f32), Some(1.)); 2]).unwrap();
    for problem in [
        LinearLeastSquaresProblem::from_view(&dense, &rhs).unwrap(),
        LinearLeastSquaresProblem::new(&sparse, &rhs).unwrap(),
        LinearLeastSquaresProblem::new(&sparse_view, &rhs).unwrap(),
    ] {
        let result = Nnls::<f32>::default().solve(&problem).unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[1] - 1.5).abs() < 1e-3);
        let result = BoundedLinearLeastSquares::<f32>::default()
            .solve(&problem.with_bounds(&bounds).unwrap())
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[1] - 1.).abs() < 1e-3);
    }
}

#[test]
fn uniformly_small_coefficients_do_not_trap_the_initial_radius() {
    use scioxide_core::{FeasibilityTolerance, GradientTolerance, ResidualTolerance};
    use scioxide_optimize::LinearLeastSquaresOptions;
    fn check_scale<S: scioxide_core::RealScalar>(scale: S, tolerance: S) {
        let zero = S::try_from_f64_exact(0.).unwrap();
        let one = S::try_from_f64_exact(1.).unwrap();
        let a = Matrix::from_vec(1, 1, vec![scale]).unwrap();
        let rhs = [one];
        let bounds = Bounds::new(vec![(Some(zero), None)]).unwrap();
        let options = LinearLeastSquaresOptions::default().with_tolerances(
            GradientTolerance::new(scale * tolerance, zero).unwrap(),
            FeasibilityTolerance::default(),
            ResidualTolerance::default(),
        );
        for problem in [
            LinearLeastSquaresProblem::from_dense(&a, &rhs).unwrap(),
            LinearLeastSquaresProblem::new(&a.view().try_finite().unwrap(), &rhs).unwrap(),
        ] {
            let result = BoundedLinearLeastSquares::new(options)
                .solve(&problem.with_bounds(&bounds).unwrap())
                .unwrap();
            assert!(result.is_success(), "{result:?}");
            assert!(scioxide_core::NumericScalar::abs(result.x()[0] * scale - one) < tolerance);
            assert!(result.iterations() < 5);
        }
    }
    check_scale(1e-200_f64, 1e-10);
    check_scale(1e200_f64, 1e-10);
    check_scale(1e-100_f64, 1e-10);
    check_scale(1e100_f64, 1e-10);
    check_scale(1e-20_f32, 1e-4);
    check_scale(1e20_f32, 1e-4);
}

#[test]
fn cancellation_from_the_final_provider_preempts_success() {
    use scioxide_core::{AdjointOperator, CancellationToken, ExecutionContext, LinearOperator};
    struct CancelOnAdjoint<'a>(&'a CancellationToken);
    impl LinearOperator<f64> for CancelOnAdjoint<'_> {
        fn rows(&self) -> usize {
            1
        }
        fn cols(&self) -> usize {
            1
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            y.copy_from_slice(x);
            Ok(())
        }
    }
    impl AdjointOperator<f64> for CancelOnAdjoint<'_> {
        fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_adjoint_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            y.copy_from_slice(x);
            self.0.cancel();
            Ok(())
        }
    }
    let cancellation = CancellationToken::new();
    let operator = CancelOnAdjoint(&cancellation);
    let problem = LinearLeastSquaresProblem::new(&operator, &[0.]).unwrap();
    let solver = Nnls::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut result = reserved(1, 1);
    solver
        .solve_into(
            &problem,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert_eq!(
        result.termination(),
        scioxide_optimize::OptimizeTermination::Cancelled
    );
    assert!(!result.is_success());
    assert_eq!(result.x(), &[0.]);
    assert_eq!(
        result.counts().total().unwrap().total_providers().unwrap(),
        2
    );
}

#[test]
fn bvls_initial_clipping_resolves_the_remaining_free_coordinate() {
    use scioxide_core::Budget;
    use scioxide_optimize::{LinearLeastSquaresOptions, OptimizeTermination};
    // The unconstrained solution (-0.1, 2) clips to (0, 2). BVLS must
    // then solve the remaining free column, giving the optimum (0, 1.5).
    let a = Matrix::from_vec(2, 2, vec![10_f64, 1., 0., 1.]).unwrap();
    let rhs = [1., 2.];
    let bounds = Bounds::new(vec![(Some(0.), None); 2]).unwrap();
    let solver = BoundedLinearLeastSquares::new(
        LinearLeastSquaresOptions::default().with_max_iterations(Budget::ONE),
    );
    for problem in [
        LinearLeastSquaresProblem::from_dense(&a, &rhs).unwrap(),
        LinearLeastSquaresProblem::new(&a.view().try_finite().unwrap(), &rhs).unwrap(),
    ] {
        let result = solver
            .solve(&problem.with_bounds(&bounds).unwrap())
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
        assert!(result.x()[0].abs() < 1e-8, "{result:?}");
        assert!((result.x()[1] - 2.).abs() < 1e-8, "{result:?}");
        let result = BoundedLinearLeastSquares::default()
            .solve(&problem.with_bounds(&bounds).unwrap())
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!(result.x()[0].abs() < 1e-8);
        assert!((result.x()[1] - 1.5).abs() < 1e-8);
    }
}

// The scoped entry points consume a shared resource to expose lost caller ledgers.
struct ScopeRequiredOperator<'a> {
    inner: &'a dyn scioxide_core::AdjointOperator<f64>,
    calls: std::cell::Cell<usize>,
}
impl ScopeRequiredOperator<'_> {
    fn unscoped() -> Result<(), scioxide_core::SciOxideError> {
        Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("context"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "caller scope required",
        ))
    }
    fn scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::SolveContext<'_, '_>,
        adjoint: bool,
    ) -> Result<(), scioxide_core::SciOxideError> {
        c.execution().check()?;
        c.try_consume(scioxide_core::BudgetKind::Node, 1)?
            .require()?;
        self.calls.set(self.calls.get() + 1);
        if adjoint {
            self.inner.apply_adjoint_into(x, y, w)
        } else {
            self.inner.apply_into(x, y, w)
        }
    }
}
impl scioxide_core::LinearOperator<f64> for ScopeRequiredOperator<'_> {
    fn rows(&self) -> usize {
        self.inner.rows()
    }
    fn cols(&self) -> usize {
        self.inner.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        self.inner.apply_workspace_len()
    }
    fn apply_into(
        &self,
        _: &[f64],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        Self::unscoped()
    }
    fn apply_into_scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, scioxide_core::SciOxideError> {
        self.scoped(x, y, w, c, false)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl scioxide_core::AdjointOperator<f64> for ScopeRequiredOperator<'_> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        self.inner.apply_adjoint_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        _: &[f64],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        Self::unscoped()
    }
    fn apply_adjoint_into_scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, scioxide_core::SciOxideError> {
        self.scoped(x, y, w, c, true)
            .map(scioxide_core::Invocation::Evaluated)
    }
}

#[test]
fn nnls_retains_scope_through_model_products_and_gradient_checks() {
    use scioxide_core::{BudgetKind, ExecutionContext, SolveBudgets, SolveState};
    let matrix = Matrix::from_vec(3, 3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).unwrap();
    let a = ScopeRequiredOperator {
        inner: &matrix.view().try_finite().unwrap(),
        calls: std::cell::Cell::new(0),
    };
    let problem = LinearLeastSquaresProblem::new(&a, &[2.0, 3.0, 4.0]).unwrap();
    let solver = Nnls::default();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(1), &mut budgets)
        .unwrap();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut result = reserved(3, 3);
    solver
        .solve_into(&problem, &mut result, &mut workspace, &mut context)
        .unwrap();
    assert!(result.is_success());
    assert!(result.x().iter().all(|v| (*v - 1.0).abs() < 1e-7));
    assert!(a.calls.get() > 0);
    assert_eq!(
        context.budget_report(BudgetKind::Node).consumed(),
        a.calls.get()
    );
}

fn reserved(rows: usize, cols: usize) -> scioxide_optimize::LinearLeastSquaresResult<f64> {
    let mut result = scioxide_optimize::LinearLeastSquaresResult::empty();
    result.try_reserve(rows, cols).unwrap();
    result
}

#[test]
fn updated_dense_workspaces_reinitialize_coefficients_rhs_and_bounds() {
    macro_rules! check {
        ($solver:expr, $bounded:expr) => {{
            let solver = $solver;
            let initial = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap();
            let p = LinearLeastSquaresProblem::from_dense(&initial, &[1., 2., 3.]).unwrap();
            let mut workspace = solver.workspace(&p).unwrap();
            let mut out = reserved(3, 2);
            for (coefficients, rhs, endpoints) in [
                (
                    [1., 0., 0., 1., 1., 1.],
                    [1., 2., 3.],
                    [(Some(-0.5), Some(0.5)); 2],
                ),
                (
                    [2., 1., 1., 3., 0., 1.],
                    [-1., 4., 2.],
                    [(Some(0.25), Some(0.25)), (None, Some(1.))],
                ),
                ([1., 1., 2., 2., 3., 3.], [2., 4., 6.], [(None, None); 2]),
            ] {
                let a = Matrix::from_vec(3, 2, coefficients.to_vec()).unwrap();
                let bounds = Bounds::new(endpoints.to_vec()).unwrap();
                let mut problem = LinearLeastSquaresProblem::from_dense(&a, &rhs).unwrap();
                if $bounded {
                    problem = problem.with_bounds(&bounds).unwrap();
                }
                let expected = solver.solve(&problem).unwrap();
                {
                    solver
                        .solve_into(
                            &problem,
                            &mut out,
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                };
                assert!(out.is_success(), "{out:?}");
                assert!((out.objective() - expected.objective()).abs() < 1e-12);
                for (&x, &y) in out.x().iter().zip(expected.x()) {
                    assert!((x - y).abs() < 1e-10);
                }
            }
        }};
    }
    check!(Nnls::<f64>::default(), false);
    check!(BoundedLinearLeastSquares::<f64>::default(), true);
}

#[test]
fn rank_truncation_never_substitutes_for_original_kkt_checks() {
    use scioxide_core::{
        FeasibilityTolerance, GradientTolerance, RankThreshold, ResidualTolerance,
    };
    use scioxide_optimize::LinearLeastSquaresOptions;
    let a = Matrix::from_vec(2, 2, vec![1., 0., 0., 1e-10]).unwrap();
    let p = LinearLeastSquaresProblem::from_dense(&a, &[1., 1e-10]).unwrap();
    let options = LinearLeastSquaresOptions::default()
        .with_rank_policy(scioxide_linalg::RankPolicy::Threshold(
            RankThreshold::new(0., 1e-6).unwrap(),
        ))
        .with_tolerances(
            GradientTolerance::new(1e-30, 0.).unwrap(),
            FeasibilityTolerance::default(),
            ResidualTolerance::default(),
        );
    for result in [
        Nnls::new(options).solve(&p).unwrap(),
        BoundedLinearLeastSquares::new(options).solve(&p).unwrap(),
    ] {
        assert!(!result.is_success(), "{result:?}");
        assert_eq!(result.rank().unwrap().rank(), 1);
        assert!(result.kkt().stationarity() > 1e-30);
    }
}

#[test]
fn strict_original_gradients_refine_or_report_rounding_stagnation() {
    use scioxide_core::{FeasibilityTolerance, GradientTolerance, ResidualTolerance};
    use scioxide_optimize::LinearLeastSquaresOptions;
    let a = Matrix::from_vec(
        5,
        4,
        vec![
            2., 1., 0., 1., 1., 3., 1., 0., 0., 1., 4., 1., 1., 0., 1., 5., 2., -1., 1., 2.,
        ],
    )
    .unwrap();
    let expected = [1., 2., 3., 4.];
    let b: Vec<f64> = a
        .as_slice()
        .chunks_exact(4)
        .map(|row| row.iter().zip(expected).map(|(a, x)| a * x).sum())
        .collect();
    let problem = LinearLeastSquaresProblem::from_dense(&a, &b).unwrap();
    let options = LinearLeastSquaresOptions::default().with_tolerances(
        GradientTolerance::new(1e-25, 0.).unwrap(),
        FeasibilityTolerance::default(),
        ResidualTolerance::default(),
    );
    for result in [
        Nnls::new(options).solve(&problem).unwrap(),
        BoundedLinearLeastSquares::new(options)
            .solve(&problem)
            .unwrap(),
    ] {
        assert!(result.kkt().stationarity() < 1e-12, "{result:?}");
        if result.is_success() {
            assert!(result.kkt().stationarity() < 1e-25);
        } else {
            assert_eq!(
                result.termination(),
                scioxide_optimize::OptimizeTermination::Stagnation
            );
            assert!(result.kkt().stationarity() > 1e-25);
        }
        for (&x, expected) in result.x().iter().zip(expected) {
            assert!((x - expected).abs() < 1e-13);
        }
    }
}

#[test]
fn dense_update_budget_stops_publish_only_validated_feasible_points() {
    use scioxide_core::{Budget, BudgetKind, SolveBudgets, SolveState};
    macro_rules! check {
        ($solver:expr, $problem:expr, $matrix:expr, $rhs:expr) => {{
            let solver = $solver;
            let problem = $problem;
            let mut workspace = solver.workspace(&problem).unwrap();
            for limit in 1..40 {
                let mut out = reserved(4, 3);
                let mut state = SolveState::new();
                let mut budgets = SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
                let mut ctx = state
                    .begin_solve(ExecutionContext::sequential(), parent(1), &mut budgets)
                    .unwrap();
                let result = solver.solve_into(&problem, &mut out, &mut workspace, &mut ctx);
                assert!(ctx.counts().total().unwrap().total_providers().unwrap() <= limit);
                if limit == 1 {
                    assert!(result.is_err());
                    assert!(out.x().is_empty());
                } else {
                    result.unwrap();
                    assert!(
                        out.is_success()
                            || out.termination()
                                == scioxide_optimize::OptimizeTermination::EvaluationLimit
                    );
                    assert_eq!(out.counts(), workspace.last_attempt().unwrap().counts());
                    let residual: Vec<_> = $matrix
                        .as_slice()
                        .chunks_exact(3)
                        .zip($rhs)
                        .map(|(row, b)| {
                            row.iter().zip(out.x()).map(|(a, x)| a * x).sum::<f64>() - b
                        })
                        .collect();
                    for (&actual, &expected) in out.residual().iter().zip(&residual) {
                        assert!((actual - expected).abs() < 1e-12);
                    }
                    assert!(out.kkt().primal_feasibility() < 1e-12);
                }
            }
        }};
    }
    let matrix =
        Matrix::from_vec(4, 3, vec![2., 1., 0., 0., 2., 1., 1., 0., 2., 1., 1., 1.]).unwrap();
    let rhs = [-2., 3., 1., -1.];
    let problem = LinearLeastSquaresProblem::from_dense(&matrix, &rhs).unwrap();
    let bounds = Bounds::new(vec![(Some(-0.25), Some(0.25)); 3]).unwrap();
    check!(Nnls::default(), problem, matrix, rhs);
    check!(
        BoundedLinearLeastSquares::default(),
        problem.with_bounds(&bounds).unwrap(),
        matrix,
        rhs
    );
}

fn parent(token: u128) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        1,
        0,
    )
}
