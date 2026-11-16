//! Complete iterate comparisons with independent dense NumPy recurrences.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
use scioxide_core::{Budget, RealScalar, ResidualTolerance, StepTolerance};
use scioxide_rootfind::{
    FixedPointInput, FixedPointMethod, FixedPointOptions, FixedPointSolver, HistoryRestart,
    ResidualFn,
};
use std::cell::RefCell;
fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn residual<S: RealScalar>(problem: &str, x: &[S], out: &mut [S]) {
    for (i, out) in out.iter_mut().enumerate() {
        *out = match problem {
            "affine" => {
                (r::<S>(0.12) + r::<S>(0.43) * r::<S>(i as f64 / 3.0)) * x[i]
                    + if i < 3 {
                        r::<S>(0.07) * x[i + 1]
                    } else {
                        S::ZERO
                    }
                    + r::<S>(-0.6 + 1.4 * i as f64 / 3.0)
                    - x[i]
            }
            "cosine" => {
                r::<S>(0.45) * x[i].cos()
                    + r::<S>(0.08) * x[(i + 3) % 4]
                    + r::<S>(-0.3 + 0.6 * i as f64 / 3.0)
                    - x[i]
            }
            _ => {
                r::<S>(0.3) * (x.iter().copied().fold(S::ZERO, |a, b| a + b) / r::<S>(4.0)).cos()
                    + r::<S>(0.1)
                    - x[i]
            }
        };
    }
}
fn compare<S: scioxide_core::NumericScalar<Real = S> + RealScalar>(tolerance: f64) {
    let rows: Vec<Vec<_>> = include_str!("data/fixed_point.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').collect())
        .collect();
    let mut configurations = 0;
    for row in rows.iter().filter(|r| r[3] == "0") {
        let method = match row[1] {
            "anderson" => FixedPointMethod::Anderson,
            "first" => FixedPointMethod::BroydenFirst,
            _ => FixedPointMethod::BroydenSecond,
        };
        let restart = if row[2] == "oldest" {
            HistoryRestart::Oldest
        } else {
            HistoryRestart::Restart
        };
        for input in [FixedPointInput::Map, FixedPointInput::Residual] {
            let options = FixedPointOptions::default()
                .with_damping(r::<S>(0.7))
                .with_history(2, restart)
                .with_residual_tolerance(ResidualTolerance::new(r(0.0), r(0.0)).unwrap())
                .with_step_tolerance(StepTolerance::new(r(0.0), r(0.0)).unwrap())
                .with_max_iterations(Budget::try_from(12).unwrap());
            let observed = RefCell::new(Vec::new());
            let callback = ResidualFn::new(4, |x: &[S], out: &mut [S]| {
                residual(row[0], x, out);
                observed.borrow_mut().push((x.to_vec(), out.to_vec()));
                if input == FixedPointInput::Map {
                    for (y, &v) in out.iter_mut().zip(x) {
                        *y += v;
                    }
                }
                Ok(())
            });
            let result = FixedPointSolver::new(method, input, (options).clone())
                .and_then(|solver| solver.solve(&callback, &[r::<S>(0.0); 4]))
                .unwrap();
            let observed = observed.borrow();
            assert!(observed.len() >= 3, "{row:?}");
            if observed.len() < 13 {
                assert!(
                    result.is_success()
                        || result.termination() == scioxide_rootfind::RootTermination::Stagnation
                );
            }
            for (iteration, reference) in rows.iter().filter(|r| r[..3] == row[..3]).enumerate() {
                // A rounded early stop must also agree with the remainder of
                // the independent trajectory, not merely its evaluated prefix.
                let (point, f) = observed
                    .get(iteration)
                    .unwrap_or_else(|| observed.last().unwrap());
                for (actual, expected) in [(&point[..], reference[4]), (&f[..], reference[5])] {
                    for (&a, b) in actual
                        .iter()
                        .zip(expected.split(',').map(|v| v.parse::<f64>().unwrap()))
                    {
                        assert!(
                            (a.into_f64() - b).abs() <= tolerance * (1.0 + b.abs()),
                            "{:?} {input:?}, iteration {}: {:?} vs {b}",
                            &row[..3],
                            reference[3],
                            a
                        );
                    }
                }
            }
        }
        configurations += 1;
    }
    assert_eq!(configurations, 18);
}
#[test]
fn numpy_iterates_f64() {
    compare::<f64>(5e-11);
}
#[test]
fn numpy_iterates_f32() {
    compare::<f32>(2e-3);
}
