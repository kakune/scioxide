//! Scale invariance and independently computed termination residuals.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::SciOxideError;
use scioxide_optimize::{
    AugmentedLagrangian, ConstrainedProblem, ConstraintFn, Constraints, InteriorPoint,
    ObjectiveFunction, PenaltyMethod, Sqp,
};

struct Quadratic(f64);
impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.0 * ((x[0] - 2.0).powi(2) + x[1].powi(2)))
    }
    fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        out[0] = 2.0 * self.0 * (x[0] - 2.0);
        out[1] = 2.0 * self.0 * x[1];
        Ok(true)
    }
}

#[test]
fn constrained_quadratic_is_invariant_under_objective_and_row_scaling() {
    for s in [1e-8, 1.0, 1e8] {
        for c in [1e-8, 1.0, 1e8] {
            let constraints = Constraints::new(2)
                .unwrap()
                .equality(
                    ConstraintFn::new(2, move |x: &[f64]| Ok(c * (x[0] + x[1] - 1.0)))
                        .with_jacobian(move |_: &[f64], output| {
                            output.copy_from_slice(&[c, c]);
                            Ok(())
                        }),
                )
                .unwrap();
            let f = Quadratic(s);
            let problem = ConstrainedProblem::new(&f, &constraints).unwrap();
            let results = [
                ("SQP", Sqp::default().minimize(&problem, &[0.0, 0.0])),
                (
                    "interior point",
                    InteriorPoint::default().minimize(&problem, &[0.0, 0.0]),
                ),
                (
                    "penalty",
                    PenaltyMethod::default().minimize(&problem, &[0.0, 0.0]),
                ),
                (
                    "augmented",
                    AugmentedLagrangian::default().minimize(&problem, &[0.0, 0.0]),
                ),
            ];
            for (method, result) in results {
                let result = result.unwrap_or_else(|e| panic!("{method} s={s} c={c}: {e:?}"));
                let x = result.x();
                assert!(result.is_success(), "{method} s={s} c={c}: {result:?}");
                assert!(
                    (x[0] - 1.5).hypot(x[1] + 0.5) <= 1e-6,
                    "{method} s={s} c={c}: {result:?}"
                );
                assert!((x[0] + x[1] - 1.0).abs() <= 1e-6);
                // Eliminate the equality multiplier analytically. Dividing by
                // objective units prevents tiny objectives masking stationarity.
                assert!((2.0 * (x[0] - 2.0) - 2.0 * x[1]).abs() <= 4e-6);
                assert!((result.fun() - f.eval(x).unwrap()).abs() <= s * 1e-12);
            }
        }
    }
}

#[test]
fn derivative_free_constraint_rows_preserve_small_and_large_units() {
    use scioxide_core::Budget;
    use scioxide_optimize::{Cobyla, CobylaOptions, Cobyqa, CobyqaOptions};
    for c in [1e-8, 1.0, 1e8] {
        let constraints = Constraints::new(2)
            .unwrap()
            .equality(ConstraintFn::new(2, move |x: &[f64]| {
                Ok(c * (x[0] + x[1] - 1.0))
            }))
            .unwrap();
        let objective = Quadratic(1.0);
        let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
        for (method, result) in [
            (
                "COBYLA",
                Cobyla::new(
                    CobylaOptions::default()
                        .with_radii(0.5, 1e-8)
                        .with_max_evaluations(Budget::try_from(2000).unwrap()),
                )
                .unwrap()
                .minimize(&problem, &[0.0, 0.0]),
            ),
            (
                "COBYQA",
                Cobyqa::new(
                    CobyqaOptions::default()
                        .with_radii(0.5, 1e-8)
                        .with_max_evaluations(Budget::try_from(2000).unwrap()),
                )
                .unwrap()
                .minimize(&problem, &[0.0, 0.0]),
            ),
        ] {
            let result = result.unwrap();
            let x = result.x();
            assert!(
                (x[0] + x[1] - 1.0).abs() < 1e-7,
                "{method} c={c}: {result:?}"
            );
            assert!(
                (x[0] - 1.5).hypot(x[1] + 0.5) < 1e-5,
                "{method} c={c}: {result:?}"
            );
            assert_eq!(result.stationarity(), None);
        }
    }
}

#[test]
fn local_relative_gradient_threshold_uses_objective_per_coordinate() {
    use scioxide_core::GradientTolerance;
    use scioxide_optimize::{
        Bfgs, BfgsOptions, Lbfgs, LbfgsOptions, NonlinearCg, NonlinearCgOptions, ObjectiveFn,
        TrustRegionOptimizer, TrustRegionOptions,
    };
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(1e12 + 1e8 * (x[0] - 1.0).powi(2)))
        .with_gradient(|x: &[f64], output| {
            output.copy_from_slice(&[2e8 * (x[0] - 1.0)]);
            Ok(())
        })
        .with_hessian(|_: &[f64], output| {
            output.as_mut_slice().copy_from_slice(&[2e8]);
            Ok(())
        });
    let relative = GradientTolerance::new(0.0, 1e-5).unwrap();
    let absolute = GradientTolerance::new(1e-8, 0.0).unwrap();
    macro_rules! check {
        ($solver:ident,$options:ident) => {{
            let result = $solver::new($options::default().with_gradient_tolerance(relative))
                .minimize(&f, &[0.99])
                .unwrap();
            assert!(result.is_success());
            assert_eq!(result.x(), &[0.99]);
            let result = $solver::new($options::default().with_gradient_tolerance(absolute))
                .minimize(&f, &[0.99])
                .unwrap();
            assert!((result.x()[0] - 1.0).abs() < 1e-8, "{result:?}");
        }};
    }
    check!(Bfgs, BfgsOptions);
    check!(Lbfgs, LbfgsOptions);
    check!(NonlinearCg, NonlinearCgOptions);
    check!(TrustRegionOptimizer, TrustRegionOptions);
}
