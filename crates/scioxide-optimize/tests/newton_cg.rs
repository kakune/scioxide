//! Independent dense/HVP checks of truncated Newton globalization.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::SciOxideError;
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    NewtonOptimizer, ObjectiveFunction, TrustRegionOptimizer, TrustRegionOptions,
    TrustRegionSubproblem,
};

/// Steihaug-CG trust region with the default 100-product inner budget.
fn steihaug(second_order_stopping: bool) -> TrustRegionOptimizer<f64> {
    TrustRegionOptimizer::new(TrustRegionOptions::default().with_subproblem(
        TrustRegionSubproblem::SteihaugCg {
            max_inner_iterations: scioxide_core::Budget::try_from(100).unwrap(),
            second_order_stopping,
        },
    ))
}

struct Bowl {
    hvp: bool,
}
impl ObjectiveFunction<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.5 * x[0] * x[0] + 2. * x[1] * x[1])
    }
    fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        out.copy_from_slice(&[x[0], 4. * x[1]]);
        Ok(true)
    }
    fn hessian_into(&self, _: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        assert!(!self.hvp);
        out.as_mut_slice().copy_from_slice(&[1., 0., 0., 4.]);
        Ok(true)
    }
    fn has_hessian_vector_product(&self) -> bool {
        self.hvp
    }
    fn hessian_vector_product_into(
        &self,
        _: &[f64],
        p: &[f64],
        out: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[p[0], 4. * p[1]]);
        Ok(())
    }
}
#[test]
fn hvp_and_dense_converge_to_independent_minimum() {
    let solver = steihaug(false);
    for hvp in [false, true] {
        let result = solver.minimize(&Bowl { hvp }, &[3., -2.]).unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!(result.x().iter().all(|x| x.abs() < 1e-7));
    }
}
struct DoubleWell;
impl ObjectiveFunction<f64> for DoubleWell {
    fn dim(&self) -> usize {
        1
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((x[0] * x[0] - 1.).powi(2))
    }
    fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        out[0] = 4. * x[0] * (x[0] * x[0] - 1.);
        Ok(true)
    }
    fn hessian_into(&self, x: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        out.as_mut_slice()[0] = 12. * x[0] * x[0] - 4.;
        Ok(true)
    }
}
#[test]
fn indefinite_models_reach_a_minimum() {
    let cg = steihaug(false).minimize(&DoubleWell, &[0.25]).unwrap();
    let newton = NewtonOptimizer::default()
        .minimize(&DoubleWell, &[0.25])
        .unwrap();
    for result in [cg, newton] {
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[0].abs() - 1.).abs() < 1e-7);
    }
}

#[test]
fn explicit_second_order_stopping_escapes_a_stationary_saddle() {
    let default = steihaug(false).minimize(&DoubleWell, &[0.]).unwrap();
    assert_eq!(
        default.termination(),
        scioxide_optimize::OptimizeTermination::GradientSmall
    );
    assert!(!newton_of(&default).second_order_checked());
    let checked = steihaug(true).minimize(&DoubleWell, &[0.]).unwrap();
    assert!(checked.is_success(), "{checked:?}");
    assert!((checked.x()[0].abs() - 1.).abs() < 1e-8);
    assert!(newton_of(&checked).second_order_checked());
    assert_eq!(newton_of(&checked).inertia().unwrap().negative(), 0);
}
#[test]
fn second_order_stopping_forms_the_dense_hessian_of_an_hvp_objective() {
    // Only Hessian-vector products are analytic; the dense Hessian needed by
    // the curvature check comes from finite differences.
    struct HvpOnly;
    impl ObjectiveFunction<f64> for HvpOnly {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Bowl { hvp: true }.eval(x)
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
            Bowl { hvp: true }.gradient_into(x, out)
        }
        fn has_hessian_vector_product(&self) -> bool {
            true
        }
        fn hessian_vector_product_into(
            &self,
            x: &[f64],
            p: &[f64],
            out: &mut [f64],
        ) -> Result<(), SciOxideError> {
            Bowl { hvp: true }.hessian_vector_product_into(x, p, out)
        }
    }
    let result = steihaug(true).minimize(&HvpOnly, &[3., -2.]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(result.x().iter().all(|x| x.abs() < 1e-6));
}

fn newton_of<S: scioxide_core::RealScalar>(
    result: &scioxide_optimize::OptimizeResult<S>,
) -> scioxide_optimize::NewtonDiagnostics<S> {
    match result.method_diagnostics() {
        scioxide_optimize::OptimizeDiagnostics::Newton(diagnostics) => diagnostics,
        other => panic!("unexpected method diagnostics {other:?}"),
    }
}
