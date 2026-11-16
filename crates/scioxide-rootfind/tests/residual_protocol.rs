//! One residual type serves every nonlinear solver family.

use scioxide_core::{Budget, JacobianOperator, ResidualOperator, SciOxideError, SciOxideErrorView};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    Broyden, BroydenUpdate, Continuation, ContinuationOptions, ContinuationPath,
    ContinuationProblem, DeflatedProblem, DeflationOptions, DeflationSet, DenseJacobian,
    FixedPointInput, FixedPointMethod, FixedPointOptions, FixedPointSolver, LinearHomotopy,
    NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver, NewtonSystem, ResidualFn,
    SystemRootSolver, TrustRegionSystem,
};

/// `F(x) = c − x²` with every derivative capability: roots `±√c`.
struct Square(f64);

impl ResidualOperator<f64> for Square {
    fn input_dimension(&self) -> usize {
        1
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        y[0] = self.0 - x[0] * x[0];
        Ok(())
    }
}
impl JacobianOperator<f64> for Square {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        y[0] = -2.0 * x[0] * v[0];
        Ok(())
    }
}
impl DenseJacobian<f64> for Square {
    fn jacobian_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<bool, SciOxideError> {
        y[0] = -2.0 * x[0];
        Ok(true)
    }
}

fn close(x: f64, root: f64) -> bool {
    (x - root).abs() <= 1e-8
}

#[test]
fn dense_fixed_point_and_newton_krylov_solvers_share_one_residual() {
    let square = Square(0.25);

    for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Central] {
        for result in [
            NewtonSystem::new(
                (scioxide_rootfind::NewtonSystemOptions::default())
                    .with_derivative_strategy(strategy),
            )
            .unwrap()
            .solve(&square, &[1.0]),
            Broyden::new(
                BroydenUpdate::First,
                (scioxide_rootfind::BroydenOptions::default()).with_derivative_strategy(strategy),
            )
            .unwrap()
            .solve(&square, &[1.0]),
            TrustRegionSystem::new(
                (scioxide_rootfind::TrustRegionSystemOptions::default())
                    .with_derivative_strategy(strategy),
            )
            .unwrap()
            .solve(&square, &[1.0]),
        ] {
            let result = result.unwrap();
            assert!(result.is_success(), "{result:?}");
            assert!(close(result.point()[0], 0.5), "{result:?}");
        }
    }

    let fixed_point = FixedPointSolver::new(
        FixedPointMethod::Anderson,
        FixedPointInput::Residual,
        FixedPointOptions::default(),
    )
    .and_then(|solver| solver.solve(&square, &[1.0]))
    .unwrap();
    assert!(fixed_point.is_success(), "{fixed_point:?}");
    assert!(close(fixed_point.point()[0], 0.5));

    for problem in [
        NewtonKrylovProblem::with_jacobian(&square),
        NewtonKrylovProblem::numerical(&square),
    ] {
        let result = NewtonKrylovSolver::new((NewtonKrylovOptions::default()).clone())
            .and_then(|root_plan| root_plan.solve(&problem, &[1.0]))
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!(close(result.point()[0], 0.5));
    }
}

#[test]
fn continuation_and_deflation_take_the_same_residual() {
    let start = Square(0.25);
    let target = Square(1.0);
    let homotopy = LinearHomotopy::new(&start, &target).unwrap();
    let problem = ContinuationProblem::new(&homotopy).unwrap();
    let options =
        ContinuationOptions::new(0.1, 1e-6, 0.15, Budget::try_from(200).unwrap()).unwrap();
    let continuation = Continuation::new(ContinuationPath::Parameters(vec![1.0]), options).unwrap();
    let path = continuation.state(&problem, &[0.5, 0.0]).unwrap();
    let path = continuation.follow(&problem, path).unwrap();
    assert!(path.is_success());
    assert!(close(path.point()[0], 1.0));

    let mut known =
        DeflationSet::new(1, Budget::try_from(2).unwrap(), DeflationOptions::default()).unwrap();
    known.insert(&start, &[0.5]).unwrap();
    let deflated = DeflatedProblem::new(&start, &known, NewtonKrylovOptions::default()).unwrap();
    let result = deflated.solve(&[-0.2]).unwrap();
    let original = result.residual_report().unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(close(result.point()[0], -0.5));
    assert!(original.raw_norm() <= 1e-8);
}

/// `F(x) = c − x²` as closures, with the arithmetic of [`Square`].
fn square_fn(c: f64) -> impl JacobianOperator<f64> {
    ResidualFn::new(1, move |x: &[f64], y: &mut [f64]| {
        y[0] = c - x[0] * x[0];
        Ok(())
    })
    .with_jvp(|x: &[f64], v: &[f64], y: &mut [f64]| {
        y[0] = -2.0 * x[0] * v[0];
        Ok(())
    })
}

#[test]
fn closure_products_serve_newton_krylov_deflation_and_homotopy() {
    let closures = square_fn(0.25);
    let solver = NewtonKrylovSolver::new(NewtonKrylovOptions::default()).unwrap();
    let result = solver
        .solve(&NewtonKrylovProblem::with_jacobian(&closures), &[1.0])
        .unwrap();
    let reference = solver
        .solve(&NewtonKrylovProblem::with_jacobian(&Square(0.25)), &[1.0])
        .unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(close(result.point()[0], 0.5));
    assert_eq!(result.point(), reference.point());
    assert_eq!(result.counts(), reference.counts());

    let mut known =
        DeflationSet::new(1, Budget::try_from(2).unwrap(), DeflationOptions::default()).unwrap();
    known.insert(&closures, &[0.5]).unwrap();
    let deflated = DeflatedProblem::new(&closures, &known, NewtonKrylovOptions::default()).unwrap();
    let result = deflated.solve(&[-0.2]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(close(result.point()[0], -0.5));

    let target = square_fn(1.0);
    let homotopy = LinearHomotopy::new(&closures, &target).unwrap();
    let problem = ContinuationProblem::new(&homotopy).unwrap();
    let options =
        ContinuationOptions::new(0.1, 1e-6, 0.15, Budget::try_from(200).unwrap()).unwrap();
    let continuation = Continuation::new(ContinuationPath::Parameters(vec![1.0]), options).unwrap();
    let path = continuation.state(&problem, &[0.5, 0.0]).unwrap();
    let path = continuation.follow(&problem, path).unwrap();
    assert!(path.is_success());
    assert!(close(path.point()[0], 1.0));
}

#[test]
fn parameter_closures_form_a_continuation_problem() {
    // F(x, λ) = 1/4 + 3λ/4 − x², the homotopy above in one closure.
    let map = ResidualFn::with_parameter(1, |y: &[f64], out: &mut [f64]| {
        out[0] = 0.25 + 0.75 * y[1] - y[0] * y[0];
        Ok(())
    })
    .with_jvp(|y: &[f64], v: &[f64], out: &mut [f64]| {
        out[0] = -2.0 * y[0] * v[0] + 0.75 * v[1];
        Ok(())
    });
    assert_eq!((map.input_dimension(), map.output_dimension()), (2, 1));
    let problem = ContinuationProblem::new(&map).unwrap();
    assert_eq!(problem.dimension(), 1);
    let options =
        ContinuationOptions::new(0.1, 1e-6, 0.15, Budget::try_from(200).unwrap()).unwrap();
    let continuation = Continuation::new(ContinuationPath::Parameters(vec![1.0]), options).unwrap();
    let path = continuation.state(&problem, &[0.5, 0.0]).unwrap();
    let path = continuation.follow(&problem, path).unwrap();
    assert!(path.is_success());
    assert_eq!(path.point()[1], 1.0);
    assert!(close(path.point()[0], 1.0));
    // A square map has no parameter and is rejected.
    assert!(ContinuationProblem::new(&square_fn(1.0)).is_err());
}

fn dimension_mismatch<T>(result: Result<T, SciOxideError>) -> bool {
    matches!(
        result,
        Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. })
    )
}

#[test]
fn closure_adapter_checks_slice_lengths() {
    // F(x, λ) = x² − λ with its 1 × 2 Jacobian and its products.
    let fold = ResidualFn::with_parameter(1, |y: &[f64], out: &mut [f64]| {
        out[0] = y[0] * y[0] - y[1];
        Ok(())
    })
    .with_jacobian(|y: &[f64], out: &mut [f64]| {
        out.copy_from_slice(&[2.0 * y[0], -1.0]);
        Ok(())
    })
    .with_jvp(|y: &[f64], v: &[f64], out: &mut [f64]| {
        out[0] = 2.0 * y[0] * v[0] - v[1];
        Ok(())
    });
    let mut one = [7.0];
    let mut two = [7.0; 2];
    assert!(dimension_mismatch(fold.residual_into(
        &[3.0],
        &mut one,
        &mut []
    )));
    assert!(dimension_mismatch(fold.residual_into(
        &[3.0, 1.0],
        &mut two,
        &mut []
    )));
    assert!(dimension_mismatch(fold.jvp_into(
        &[3.0],
        &[1.0, 2.0],
        &mut one,
        &mut []
    )));
    assert!(dimension_mismatch(fold.jvp_into(
        &[3.0, 1.0],
        &[1.0],
        &mut one,
        &mut []
    )));
    assert!(dimension_mismatch(fold.jvp_into(
        &[3.0, 1.0],
        &[1.0, 2.0],
        &mut two,
        &mut []
    )));
    assert!(dimension_mismatch(fold.jacobian_into(
        &[3.0],
        &mut two,
        &mut []
    )));
    assert!(dimension_mismatch(fold.jacobian_into(
        &[3.0, 1.0],
        &mut [0.0; 4],
        &mut []
    )));
    // A rejected call leaves its output unwritten.
    assert_eq!((one, two), ([7.0], [7.0; 2]));

    fold.residual_into(&[3.0, 1.0], &mut one, &mut []).unwrap();
    assert_eq!(one, [8.0]);
    fold.jvp_into(&[3.0, 1.0], &[1.0, 2.0], &mut one, &mut [])
        .unwrap();
    assert_eq!(one, [4.0]);
    assert_eq!(fold.jvp_workspace_len().unwrap(), 0);
    assert!(fold.jacobian_into(&[3.0, 1.0], &mut two, &mut []).unwrap());
    assert_eq!(two, [6.0, -1.0]);

    // The dense solvers require a square map.
    let newton = NewtonSystem::new(scioxide_rootfind::NewtonSystemOptions::default()).unwrap();
    assert!(dimension_mismatch(newton.workspace_requirements(&fold)));
    let broyden = Broyden::new(
        BroydenUpdate::First,
        scioxide_rootfind::BroydenOptions::default(),
    )
    .unwrap();
    assert!(dimension_mismatch(broyden.workspace_requirements(&fold)));
    let trust_region =
        TrustRegionSystem::new(scioxide_rootfind::TrustRegionSystemOptions::default()).unwrap();
    assert!(dimension_mismatch(
        trust_region.workspace_requirements(&fold)
    ));
}
