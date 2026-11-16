//! Solver-proposed trials with non-finite values are rejected, never errors;
//! caller starts remain strict.
#![allow(clippy::unwrap_used, clippy::panic)]
use core::cell::Cell;
use scioxide_core::{Bounds, Budget, Interval, SciOxideError, SciOxideErrorView};
use scioxide_linalg::Matrix;
use scioxide_optimize::*;

/// `sum (x_i - center)^2` below a wall at `x_i >= at`, where it returns `wall`.
/// The supplied Hessian is `curvature * I`; values below 2 make Newton steps overshoot.
/// The `huber` variant `sum sqrt(1 + (x_i - center)^2) - 1` defeats exact
/// quadratic interpolation models.
struct Wall {
    n: usize,
    huber: bool,
    center: f64,
    at: f64,
    wall: f64,
    curvature: f64,
    beyond: Cell<usize>,
}
impl Wall {
    fn new(n: usize, wall: f64, curvature: f64) -> Self {
        Self {
            n,
            huber: false,
            center: 3.0,
            at: 3.5,
            wall,
            curvature,
            beyond: Cell::new(0),
        }
    }
    fn shifted(self, center: f64, at: f64) -> Self {
        Self { center, at, ..self }
    }
}
impl ObjectiveFunction<f64> for Wall {
    fn dim(&self) -> usize {
        self.n
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        if x.iter().any(|&v| v >= self.at) {
            self.beyond.set(self.beyond.get() + 1);
            return Ok(self.wall);
        }
        Ok(x.iter()
            .map(|v| {
                let d = v - self.center;
                if self.huber {
                    d.hypot(1.0) - 1.0
                } else {
                    d * d
                }
            })
            .sum())
    }
    fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        for (o, v) in out.iter_mut().zip(x) {
            let d = v - self.center;
            *o = if self.huber {
                d / d.hypot(1.0)
            } else {
                2.0 * d
            };
        }
        Ok(true)
    }
    fn hessian_into(&self, _x: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        let n = self.n;
        for i in 0..n {
            for j in 0..n {
                out.as_mut_slice()[i * n + j] = if i == j { self.curvature } else { 0.0 };
            }
        }
        Ok(true)
    }
}

fn check(name: &str, wall: &Wall, result: Result<OptimizeResult<f64>, SciOxideError>, tol: f64) {
    let result = result.unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert!(result.is_success(), "{name}: {result:?}");
    assert!(
        result.x().iter().all(|&v| (v - wall.center).abs() < tol),
        "{name}: {:?}",
        result.x()
    );
    assert!(
        result.fun().is_finite() && result.fun() < tol * tol,
        "{name}: {result:?}"
    );
    assert!(
        wall.beyond.get() > 0,
        "{name} never proposed a trial beyond the wall"
    );
}

#[test]
fn line_search_and_trust_region_methods_reject_non_finite_trials() {
    for value in [f64::INFINITY, f64::NAN] {
        macro_rules! run {
            ($name:literal, $curvature:expr, $solver:expr) => {{
                let wall = Wall::new(2, value, $curvature);
                let result = $solver.minimize(&wall, &[0.0, 0.0]);
                check($name, &wall, result, 1e-6);
            }};
        }
        run!("BFGS", 2.0, Bfgs::new(BfgsOptions::default()));
        run!("L-BFGS", 2.0, Lbfgs::new(LbfgsOptions::default()));
        run!("NCG", 2.0, NonlinearCg::new(NonlinearCgOptions::default()));
        run!(
            "gradient descent",
            2.0,
            GradientDescent::new(GradientDescentOptions::default())
        );
        run!(
            "Newton",
            0.2,
            NewtonOptimizer::new(NewtonOptions::default())
        );
        run!(
            "dogleg",
            0.2,
            TrustRegionOptimizer::new(TrustRegionOptions::default())
        );
        run!(
            "Steihaug",
            0.2,
            TrustRegionOptimizer::new(TrustRegionOptions::default().with_subproblem(
                TrustRegionSubproblem::SteihaugCg {
                    max_inner_iterations: Budget::try_from(10).unwrap(),
                    second_order_stopping: false,
                },
            ))
        );
        let wall = Wall::new(2, value, 2.0);
        let bounds = Bounds::new(vec![(Some(-10.0), Some(10.0)); 2]).unwrap();
        let bounded = BoundedObjective::new(&wall, &bounds).unwrap();
        let result = LbfgsB::new(LbfgsBOptions::default()).minimize(&bounded, &[0.0, 0.0]);
        check("L-BFGS-B", &wall, result, 1e-6);
    }
}

#[test]
fn derivative_free_local_searches_rank_non_finite_trials_last() {
    for value in [f64::INFINITY, f64::NAN] {
        let wall = Wall::new(2, value, 2.0);
        let result = NelderMead::new(NelderMeadOptions::default())
            .unwrap()
            .minimize(&wall, &[0.0, 0.0]);
        check("Nelder-Mead", &wall, result, 1e-3);
        let wall = Wall::new(2, value, 2.0);
        let result = Powell::new(PowellOptions::default())
            .unwrap()
            .minimize(&wall, &[0.0, 0.0]);
        check("Powell", &wall, result, 1e-3);
    }
}

#[test]
fn scalar_searches_never_publish_rejected_trials() {
    let beyond = Cell::new(0);
    let f = |x: f64| -> Result<f64, SciOxideError> {
        if x >= 3.5 {
            beyond.set(beyond.get() + 1);
            Ok(f64::NAN)
        } else {
            Ok((x - 3.0).powi(2))
        }
    };
    for method in [
        ScalarMinimumMethod::Brent,
        ScalarMinimumMethod::GoldenSection,
    ] {
        let solver = ScalarMinimizer::new(method, ScalarMinimumOptions::default());
        for start in [
            ScalarMinimumStart::Seeds(0.0, 1.0),
            ScalarMinimumStart::Bounded(Interval::new(0.0, 10.0).unwrap()),
        ] {
            beyond.set(0);
            let result = solver.minimize(&f, start).unwrap();
            assert!(result.is_success(), "{method:?} {start:?}: {result:?}");
            assert!((result.x() - 3.0).abs() < 1e-6, "{result:?}");
            assert!(result.fun().is_finite());
            assert!(beyond.get() > 0, "{method:?} {start:?} never hit the wall");
            if let Some(samples) = result.samples() {
                assert!(samples.iter().all(|s| s.value().is_finite()));
            }
        }
    }
    // The seeds 0 and 1 expand to 3 and then 7, beyond the wall. That bracket has
    // a rejected endpoint, so it is withheld and bracketing is unsuccessful.
    let bracket = ScalarMinimizer::new(ScalarMinimumMethod::Brent, ScalarMinimumOptions::default())
        .bracket_minimum(&f, 0.0, 1.0)
        .unwrap();
    assert!(!bracket.is_success(), "{bracket:?}");
    assert!(bracket.samples().is_none() && bracket.bracket().is_none());
    assert_eq!(bracket.x(), 3.0);
    assert_eq!(bracket.fun(), 0.0);
}

#[test]
fn constrained_solvers_reject_non_finite_trials() {
    for value in [f64::INFINITY, f64::NAN] {
        // A nonbinding inequality keeps the minimizer at (3, 3).
        let constraints = Constraints::new(2)
            .unwrap()
            .inequality(
                ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 10.0)).with_jacobian(
                    |_: &[f64], output| {
                        output.copy_from_slice(&[1.0, 1.0]);
                        Ok(())
                    },
                ),
            )
            .unwrap();
        macro_rules! run {
            ($name:literal, $tol:expr, $solver:expr) => {{
                // A first unit step from the origin crosses the wall.
                let wall = Wall::new(2, value, 2.0).shifted(0.8, 0.85);
                let problem = ConstrainedProblem::new(&wall, &constraints).unwrap();
                let result = $solver.minimize(&problem, &[0.0, 0.0]);
                check($name, &wall, result, $tol);
            }};
        }
        run!("SQP", 1e-6, Sqp::new(SqpOptions::default()));
        run!(
            "interior point",
            1e-4,
            InteriorPoint::new(InteriorPointOptions::default())
        );
        run!(
            "penalty",
            1e-4,
            PenaltyMethod::new(PenaltyOptions::default())
        );
        run!(
            "augmented Lagrangian",
            1e-4,
            AugmentedLagrangian::new(AugmentedLagrangianOptions::default())
        );
        run!(
            "COBYLA",
            1e-4,
            Cobyla::new(CobylaOptions::default().with_radii(0.5, 1e-8)).unwrap()
        );
        // COBYQA's quadratic model is exact for the quadratic wall and never
        // overshoots; the pseudo-Huber variant makes its model steps cross.
        let wall = Wall {
            huber: true,
            ..Wall::new(2, value, 2.0).shifted(0.8, 0.85)
        };
        let problem = ConstrainedProblem::new(&wall, &constraints).unwrap();
        let result = Cobyqa::new(CobyqaOptions::default().with_radii(0.5, 1e-8))
            .unwrap()
            .minimize(&problem, &[-2.0, -2.0]);
        check("COBYQA", &wall, result, 1e-4);
    }
}

#[test]
fn caller_starts_remain_strict() {
    fn non_finite<T: core::fmt::Debug>(name: &str, result: Result<T, SciOxideError>) {
        let error = result.expect_err(name);
        assert!(
            matches!(error.view(), SciOxideErrorView::NonFinite { .. }),
            "{name}: {error:?}"
        );
    }
    let start = [4.0, 0.0];
    let wall = Wall::new(2, f64::NAN, 2.0);
    non_finite(
        "BFGS",
        Bfgs::new(BfgsOptions::default()).minimize(&wall, &start),
    );
    non_finite(
        "dogleg",
        TrustRegionOptimizer::new(TrustRegionOptions::default()).minimize(&wall, &start),
    );
    non_finite(
        "Nelder-Mead",
        NelderMead::new(NelderMeadOptions::default())
            .unwrap()
            .minimize(&wall, &start),
    );
    non_finite(
        "Powell",
        Powell::new(PowellOptions::default())
            .unwrap()
            .minimize(&wall, &start),
    );
    let constraints = Constraints::new(2).unwrap();
    let problem = ConstrainedProblem::new(&wall, &constraints).unwrap();
    non_finite(
        "SQP",
        Sqp::new(SqpOptions::default()).minimize(&problem, &start),
    );
    non_finite(
        "interior point",
        InteriorPoint::new(InteriorPointOptions::default()).minimize(&problem, &start),
    );
    non_finite(
        "COBYLA",
        Cobyla::new(CobylaOptions::default())
            .unwrap()
            .minimize(&problem, &start),
    );
    let solver = ScalarMinimizer::new(ScalarMinimumMethod::Brent, ScalarMinimumOptions::default());
    non_finite(
        "scalar seeds",
        solver.minimize(
            &|_: f64| -> Result<f64, SciOxideError> { Ok(f64::INFINITY) },
            ScalarMinimumStart::Seeds(0.0, 1.0),
        ),
    );
}

#[test]
fn complex_least_squares_rejects_non_finite_trial_residuals() {
    use scioxide_core::Complex;
    // r(z) = z^2 - 9 below a wall at Re z >= 3.5. From z = 0.5 the first
    // Gauss-Newton step overshoots to 9.25; that trial is rejected.
    let beyond = Cell::new(0);
    let callback = |z: &[Complex<f64>], r: &mut [Complex<f64>]| -> Result<(), SciOxideError> {
        if z[0].re >= 3.5 {
            beyond.set(beyond.get() + 1);
            r[0] = Complex::new(f64::NAN, 0.0);
        } else {
            r[0] = z[0] * z[0] - Complex::new(9.0, 0.0);
        }
        Ok(())
    };
    let mut x = [Complex::new(0.0, 0.0)];
    let mut residual = [Complex::new(0.0, 0.0)];
    let problem = ComplexLeastSquares::new(callback, &mut x, &mut residual).unwrap();
    for globalization in [
        LeastSquaresGlobalization::gauss_newton(),
        LeastSquaresGlobalization::levenberg_marquardt(),
    ] {
        beyond.set(0);
        let result =
            LeastSquares::new(LeastSquaresOptions::default().with_globalization(globalization))
                .unwrap()
                .solve(&problem, &[0.5, 0.0])
                .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!(
            (result.x()[0] - 3.0).abs() < 1e-8 && result.x()[1].abs() < 1e-8,
            "{result:?}"
        );
        assert!(
            beyond.get() > 0,
            "{globalization:?} never proposed a trial beyond the wall"
        );
    }
}

#[test]
fn proximal_backtracking_rejects_non_finite_trials() {
    /// `(x - 3)^2 / 2` below a wall at `x >= 3.5`.
    struct Smooth(Cell<usize>);
    impl SmoothGradient<f64> for Smooth {
        fn dimension(&self) -> usize {
            1
        }
        fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            if x[0] >= 3.5 {
                self.0.set(self.0.get() + 1);
                return Ok(f64::NAN);
            }
            Ok(0.5 * (x[0] - 3.0).powi(2))
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = x[0] - 3.0;
            Ok(())
        }
    }
    struct Identity;
    impl ProximalMap<f64> for Identity {
        fn apply_into(&self, point: &[f64], _: f64, out: &mut [f64]) -> Result<(), SciOxideError> {
            out.copy_from_slice(point);
            Ok(())
        }
    }
    // The underestimated Lipschitz bound 0.1 proposes x = 30 first.
    let smooth = Smooth(Cell::new(0));
    let options = ProximalOptions::new(0.1).with_backtracking(2.0, Budget::try_from(40).unwrap());
    let result = ProximalGradient::new(options)
        .unwrap()
        .minimize(&CompositeProblem::new(&smooth, &Identity), &[0.0])
        .unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 3.0).abs() < 1e-6, "{result:?}");
    assert!(smooth.0.get() > 0);
}
