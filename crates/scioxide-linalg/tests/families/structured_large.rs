#![allow(clippy::unwrap_used, clippy::needless_range_loop)]
use scioxide_core::{ExecutionContext, Tolerance};
use scioxide_linalg::{
    StructuredMatrixView, StructuredOperation, StructuredSolveOptions, StructuredSolvePlan,
    VectorViewMut,
};

// Independent compensated residual, including product roundoff. This reads the
// original matrix, not the normalized coefficients or the solver's diagnostics.
fn backward_error(matrix: StructuredMatrixView<'_, f64>, rhs: &[f64], x: &[f64]) -> f64 {
    let norm = x.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    rhs.iter().enumerate().fold(0.0_f64, |maximum, (i, &b)| {
        let mut sum = -b;
        let mut correction = 0.0;
        let mut row = 0.0;
        for (j, &value) in x.iter().enumerate() {
            let a = matrix.get(i, j).unwrap();
            row += a.abs();
            let product = a * value;
            let next = sum + product;
            correction += if sum.abs() >= product.abs() {
                (sum - next) + product
            } else {
                (product - next) + sum
            };
            correction += a.mul_add(value, -product);
            sum = next;
        }
        maximum.max((sum + correction).abs() / (b.abs() + row * norm))
    })
}

#[test]
fn random_toeplitz_and_hankel_large_gko() {
    for line in records(include_str!("../data/structured.tsv")) {
        for context in [
            ExecutionContext::sequential(),
            ExecutionContext::sequential_native(),
        ] {
            check_case(line, true, context);
        }
    }
}

#[test]
fn two_hundred_seeded_nonsymmetric_systems_use_gko() {
    const ORDERS: [usize; 5] = [64, 128, 256, 512, 1024];
    let lines: Vec<_> = records(include_str!("../data/structured_sweep.tsv")).collect();
    assert_eq!(lines.len(), 200);
    for (seed, line) in lines.into_iter().enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields[0], "nonsymmetric");
        assert_eq!(
            fields[1].parse::<usize>().unwrap(),
            ORDERS[seed % ORDERS.len()]
        );
        assert_eq!(fields[2].parse::<usize>().unwrap(), seed);
        check_case(line, false, ExecutionContext::sequential_native());
    }
}

fn records(data: &str) -> impl Iterator<Item = &str> {
    data.lines().filter(|line| !line.starts_with('#'))
}

fn check_case(line: &str, include_hankel: bool, context: ExecutionContext<'_>) {
    let fields: Vec<_> = line.split('\t').collect();
    let n: usize = fields[1].parse().unwrap();
    let condition: f64 = fields[3].parse().unwrap();
    assert!(condition > 1.0 && condition <= 1e8);
    let vectors: Vec<Vec<f64>> = fields[4..]
        .iter()
        .map(|field| {
            field
                .split(',')
                .map(|value| value.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(vectors.len(), 4);
    let (column, row, rhs, expected) = (&vectors[0], &vectors[1], &vectors[2], &vectors[3]);
    assert!(vectors.iter().all(|v| v.len() == n));
    let reversed_row: Vec<_> = row.iter().rev().copied().collect();
    for &hankel in if include_hankel {
        &[false, true][..]
    } else {
        &[false][..]
    } {
        let matrix = if hankel {
            StructuredMatrixView::hankel(&reversed_row, column).unwrap()
        } else {
            StructuredMatrixView::toeplitz(column, row).unwrap()
        };
        let options =
            StructuredSolveOptions::new().with_tolerance(Tolerance::new(0.0, 1e-13).unwrap());
        let plan = StructuredSolvePlan::new(matrix, &options).unwrap();
        let mut x = vec![0.0; n];
        let result = plan.solve_into(
            StructuredOperation::Normal,
            rhs.as_slice(),
            VectorViewMut::from_slice(&mut x),
            &mut plan
                .workspace(&scioxide_linalg::StructuredWorkspaceRequest::solve(
                    plan.order(),
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ))
                .unwrap(),
            context,
        );
        assert!(
            result.is_ok(),
            "{} n={n} seed={} hankel={hankel}: {result:?}",
            fields[0],
            fields[2]
        );
        let report = result.unwrap();
        assert_eq!(report.gko_solves(), 1);
        assert_eq!(report.dense_lu_solves(), 0);
        assert!(report.backward_error() <= 1e-13);
        let error = backward_error(matrix, rhs, &x);
        assert!(
            error <= 1e-13,
            "{} n={n} hankel={hankel}: {error}",
            fields[0]
        );
        let norm = expected.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        for i in 0..n {
            let reference = expected[if hankel { n - 1 - i } else { i }];
            assert!((x[i] - reference).abs() <= 128.0 * f64::EPSILON * condition * norm);
        }
    }
}
