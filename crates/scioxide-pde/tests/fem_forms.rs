//! Analytical element moments and manufactured-solution FEM tests on intervals.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
mod support;

use std::sync::atomic::{AtomicUsize, Ordering};

use mesh_fixture::{explicit, uniform};
use scioxide_core::LinearOperator;
use scioxide_core::{
    ExecutionContext, ResidualTolerance, SciOxideError, SciOxideErrorView, Tolerance,
};
use scioxide_integrate::{GeometricOptions, SimplexRule};
use scioxide_linalg::{CompatibilityPolicy, Preconditioning};
use scioxide_linalg::{GmresSolver, IterativeLinearSolver, LuDecomposition, Matrix};
use scioxide_pde::{
    BoundaryData, EllipticOptions, EllipticProblem, FieldData, FieldPoint, ParabolicOptions,
    ParabolicProblem, ParabolicScheme, PdeSolution, PdeSolveReport, SimplexFemCoefficients,
    SimplexFemGeometry, SimplexFemOrder, SimplexFemPlan, SpatialDiscretization,
};
use scioxide_spatial::{IntervalMesh, Point, SegmentMesh};
use support::{Discretization, EllipticPde, End, Ends, Plan1d, Steady, TestScalar, as_f64, value};

/// Pointwise `(diffusion, reaction)` of a scalar interval problem.
struct Pointwise<F>(F);
impl<S: TestScalar, F: Fn(S) -> Result<(S, S), SciOxideError> + Sync>
    SimplexFemCoefficients<S, S, 1> for Pointwise<F>
{
    fn evaluate(
        &self,
        point: [S; 1],
        diffusion: &mut [S],
        reaction: &mut [S],
    ) -> Result<(), SciOxideError> {
        (diffusion[0], reaction[0]) = (self.0)(point[0])?;
        Ok(())
    }
}

fn source<S: TestScalar>(
    f: impl Fn(S) -> Result<S, SciOxideError> + Send + Sync + 'static,
) -> FieldData<S, S, 1> {
    FieldData::time_invariant(move |p: FieldPoint<S, 1>| f(p.point[0]))
}

fn zero<S: TestScalar>() -> BoundaryData<S, S, 1> {
    BoundaryData::zero()
}

/// Natural ends with zero conormal flux leave element forms and loads unmodified.
fn natural<S: TestScalar>() -> Ends<S> {
    Ends::new(End::natural(zero()), End::natural(zero()))
}

/// P1 plan on the intervals of `nodes`, with an optional volume rule.
fn plan<S: TestScalar>(
    nodes: &[f64],
    coefficients: &dyn SimplexFemCoefficients<S, S, 1>,
    ends: &Ends<S>,
    rule: Option<SimplexRule<S>>,
) -> Result<SimplexFemPlan<S, S, 1>, SciOxideError> {
    let mesh = support::interval_mesh(&explicit(
        nodes.iter().map(|&x| S::scalar_from_f64(x)).collect(),
    ))?;
    let boundaries = ends.tagged(&mesh)?;
    let mut options = support::fem_options();
    if let Some(rule) = rule {
        options = options.with_volume_rule(rule);
    }
    SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P1,
        coefficients,
        &boundaries,
        options,
    )
}

/// Steady P1 solve through the shared one-dimensional problem adapter, with an
/// optional injected linear solver; the values are the nodal field.
fn solve_steady<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
    options: EllipticOptions<S>,
    solver: Option<Box<dyn IterativeLinearSolver<S> + Send>>,
) -> Result<(Vec<S>, PdeSolveReport<S>), SciOxideError> {
    let steady = Steady(problem.clone());
    let Plan1d::Fem(plan) = Discretization::Fem.plan(mesh, &steady, ends)? else {
        unreachable!("a FEM plan was requested")
    };
    let source = support::source(&steady);
    let elliptic = options;
    let mut workspace = scioxide_pde::EllipticWorkspace::new();
    if let Some(solver) = solver {
        workspace = workspace.with_linear_solver(solver);
    }
    let mut values = vec![S::zero_value(); plan.unknowns()];
    let report = EllipticProblem::new(&plan, &source).solve_into(
        &elliptic,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok((values, report))
}

fn gauss<S: TestScalar>(degree: usize, dimension: usize) -> SimplexRule<S> {
    SimplexRule::positive(dimension, degree, GeometricOptions::default()).unwrap()
}

fn is_invalid<T>(result: &Result<T, SciOxideError>) -> bool {
    matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::InvalidInput { .. }))
}

/// Inadmissible sampled diffusion is a domain error in every family.
fn is_outside_domain<T>(result: &Result<T, SciOxideError>) -> bool {
    matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::DomainError { .. }))
}

fn moments<S: TestScalar>() {
    // k = 1+x³, c = 1+x and f = x² on [0, 1] with the default two-point rule.
    let coefficients = Pointwise(|x: S| Ok((S::one_value() + x * x * x, S::one_value() + x)));
    let plan = plan(&[0.0, 1.0], &coefficients, &natural(), None).unwrap();
    let matrix = plan.element_matrix(0).unwrap();
    let load = plan
        .load(&source(|x: S| Ok(x * x)), S::zero_value())
        .unwrap();
    let tolerance = if S::NAME == "f32" { 8e-7 } else { 2e-14 };
    // Independently integrated monomials: int k=5/4, reaction=[5/12,1/4,7/12].
    for (actual, expected) in
        matrix
            .iter()
            .zip([5.0 / 4.0 + 5.0 / 12.0, -1.0, -1.0, 5.0 / 4.0 + 7.0 / 12.0])
    {
        assert!((as_f64(*actual) - expected).abs() < tolerance);
    }
    for (&actual, expected) in load.iter().zip([1.0 / 12.0, 1.0 / 4.0]) {
        assert!((as_f64(actual) - expected).abs() < tolerance);
    }
    assert_eq!(matrix[1], matrix[2]);
    // u_h=2-3x: integral k*9 + (1+x)*(2-3x)^2 = 25/2.
    let matrix: Vec<f64> = matrix.iter().copied().map(as_f64).collect();
    let energy = 4.0 * matrix[0] - 4.0 * matrix[1] + matrix[3];
    assert!((energy - 12.5).abs() < tolerance * 20.0);
}

fn higher_order<S: TestScalar>() {
    let unit = Pointwise(|_: S| Ok((S::one_value(), S::zero_value())));
    let quartic = || source(|x: S| Ok(x * x * x * x));
    let low = plan(&[0.0, 1.0], &unit, &natural(), None)
        .unwrap()
        .load(&quartic(), S::zero_value())
        .unwrap();
    // Three Gauss points integrate the degree-five load exactly.
    let high = plan(&[0.0, 1.0], &unit, &natural(), Some(gauss(4, 1)))
        .unwrap()
        .load(&quartic(), S::zero_value())
        .unwrap();
    let tolerance = if S::NAME == "f32" { 2e-7 } else { 2e-14 };
    assert!((as_f64(high[0]) - 1.0 / 30.0).abs() < tolerance);
    assert!((as_f64(high[1]) - 1.0 / 6.0).abs() < tolerance);
    assert!((as_f64(low[1]) - 1.0 / 6.0).abs() > 1e-3);
    // Insufficient exactness (one point) and a triangle rule are rejected.
    for rule in [gauss(1, 1), gauss(2, 2)] {
        assert!(is_invalid(&plan(
            &[0.0, 1.0],
            &unit,
            &natural(),
            Some(rule)
        )));
    }
    // So are rules beyond the node ceiling and facet rules for point facets.
    let mesh = support::interval_mesh(&explicit(vec![S::zero_value(), S::one_value()])).unwrap();
    let boundaries = natural().tagged(&mesh).unwrap();
    let limits = support::fem_options::<S>().limits();
    let capped = support::fem_options()
        .with_volume_rule(gauss(4, 1))
        .with_limits(scioxide_pde::SimplexFemLimits::new(
            limits.max_unknowns(),
            limits.max_element_entries(),
            2,
            limits.max_storage_bytes(),
        ));
    let facet = support::fem_options().with_facet_rule(gauss(2, 1));
    for options in [capped, facet] {
        assert!(is_invalid(&SimplexFemPlan::with_options(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P1,
            &unit,
            &boundaries,
            options,
        )));
    }
}

fn domains<S: TestScalar>() {
    // Diffusion must be positive definite at every quadrature node.
    for value in [0.0, -0.0, -1.0] {
        let coefficients = Pointwise(move |_: S| Ok((S::scalar_from_f64(value), S::one_value())));
        assert!(is_outside_domain(&plan(
            &[0.0, 1.0],
            &coefficients,
            &natural(),
            None
        )));
    }
    let non_finite =
        |error: SciOxideError| matches!(error.view(), SciOxideErrorView::NonFinite { .. });
    for bad in [S::NAN, S::INFINITY] {
        for field in 0..2 {
            let coefficients = Pointwise(move |_: S| {
                Ok((
                    if field == 0 { bad } else { S::one_value() },
                    if field == 1 { bad } else { S::zero_value() },
                ))
            });
            let error = plan(&[0.0, 1.0], &coefficients, &natural(), None)
                .err()
                .unwrap();
            assert!(non_finite(error));
        }
        // Sources are evaluated with loads, after preparation.
        let unit = Pointwise(|_: S| Ok((S::one_value(), S::zero_value())));
        let prepared = plan(&[0.0, 1.0], &unit, &natural(), None).unwrap();
        let error = prepared
            .load(&source(move |_| Ok(bad)), S::zero_value())
            .unwrap_err();
        assert!(non_finite(error));
    }
    let signed = Pointwise(|_: S| Ok((S::one_value(), -S::one_value())));
    assert!(plan(&[0.0, 1.0], &signed, &natural(), None).is_ok());
    // A collapsed interval has no positive Jacobian.
    let empty = IntervalMesh::new(
        vec![
            Point::new([S::zero_value()]).unwrap(),
            Point::new([S::zero_value()]).unwrap(),
        ],
        vec![[0, 1]],
    )
    .and_then(|mesh| {
        let boundaries = natural().tagged(&mesh)?;
        SimplexFemPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P1,
            &signed,
            &boundaries,
            support::fem_options(),
        )
    });
    assert!(is_invalid(&empty), "{:?}", empty.err());
}

#[derive(Clone)]
struct Quadratic;
impl<S: TestScalar> EllipticPde<S> for Quadratic {
    fn diffusion(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value() + x[0] * x[0])
    }
    fn reaction(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value() + x[0])
    }
    fn source(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(-S::one_value() - S::scalar_from_f64(4.0) * x[0] * x[0] + x[0] * x[0] * x[0])
    }
}
fn exact(x: f64) -> f64 {
    1.0 + x + x * x
}
/// Dirichlet values 1/3, or `2u + 3 du/dn = g` (outward derivative −1 at the
/// left, +3 at the right). Natural data is the conormal flux, so the Robin law
/// becomes `k du/dn + (2k/3) u = k g/3` with k = 1 and 2 at the ends.
fn ends<S: TestScalar>(robin: bool) -> Ends<S> {
    let constant = |g: f64| value(move |_| Ok(S::scalar_from_f64(g)));
    if robin {
        let end =
            |k: f64, g: f64| End::robin(S::scalar_from_f64(2.0 * k / 3.0), constant(k * g / 3.0));
        Ends::new(end(1.0, -1.0), end(2.0, 15.0))
    } else {
        Ends::new(End::dirichlet(constant(1.0)), End::dirichlet(constant(3.0)))
    }
}
// Materialize the shared sparse/operator boundary and use independent dense LU
// to separate spatial convergence from finite-precision Krylov stagnation.
struct Dense;
impl<S: TestScalar> IterativeLinearSolver<S> for Dense {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "Dense", 1)
    }

    fn solve_into(
        &self,
        operator: &dyn LinearOperator<S>,
        rhs: scioxide_linalg::VectorView<'_, S>,
        _preconditioning: Preconditioning<'_, S>,
        mut x: scioxide_linalg::VectorViewMut<'_, S>,
        _workspace: &mut scioxide_linalg::IterativeWorkspace<S>,
        _context: scioxide_linalg::LinalgExecutionContext<'_, '_>,
    ) -> Result<scioxide_linalg::IterativeSolveReport<S>, SciOxideError> {
        let n = operator.rows();
        let mut matrix = vec![S::zero_value(); n * n];
        let mut basis = vec![S::zero_value(); n];
        let mut output = vec![S::zero_value(); n];
        let mut scratch = vec![S::zero_value(); operator.apply_workspace_len()?];
        for column in 0..n {
            basis[column] = S::one_value();
            operator.apply_into(&basis, &mut output, &mut scratch)?;
            for (row, &value) in output.iter().enumerate() {
                matrix[row * n + column] = value;
            }
            basis[column] = S::zero_value();
        }
        let matrix = Matrix::from_vec(n, n, matrix)?;
        let solution = LuDecomposition::factor(&matrix)?.solve(rhs)?;
        let tolerance = S::scalar_from_f64(if S::NAME == "f32" { 2e-5 } else { 1e-11 });
        let result = GmresSolver::new(
            scioxide_core::ResidualTolerance::new(tolerance, tolerance)?,
            scioxide_core::Budget::try_from(n + 1)?,
        )
        .with_restart(scioxide_core::Budget::try_from(n)?)
        .solve_with_options(operator, rhs, Preconditioning::None, Some(solution.view()))?;
        scioxide_linalg::VectorView::from(result.solution()).copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (x).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(result.report())
    }
}

fn refinement<S: TestScalar>() {
    for robin in [false, true] {
        for nonuniform in [false, true] {
            let mut previous: Option<(f64, f64)> = None;
            for n in [8, 16, 32] {
                let coordinates: Vec<_> = (0..=n)
                    .map(|i| {
                        let t = i as f64 / n as f64;
                        S::scalar_from_f64(if nonuniform { (t + t * t) / 2.0 } else { t })
                    })
                    .collect();
                let mesh = explicit(coordinates.clone());
                let tolerance = ResidualTolerance::from_tolerance(Tolerance::from_decimal_places(
                    if S::NAME == "f32" { 6 } else { 12 },
                    if S::NAME == "f32" { 6 } else { 12 },
                ));
                let (values, report) = solve_steady(
                    &mesh,
                    &Quadratic,
                    &ends(robin),
                    EllipticOptions::new()
                        .with_residual_tolerance(tolerance)
                        .with_compatibility(CompatibilityPolicy::RequireCompatible, tolerance),
                    Some(Box::new(Dense)),
                )
                .unwrap();
                assert!(report.diagnostics().termination().is_success());
                let mut l2 = 0.0;
                let mut h1 = 0.0;
                for (pair, values) in coordinates.windows(2).zip(values.windows(2)) {
                    let (a, b) = (as_f64(pair[0]), as_f64(pair[1]));
                    let h = b - a;
                    // Error in t=(x-a)/h is p+q*t+r*t^2. Integrate its square analytically.
                    let p = as_f64(values[0]) - exact(a);
                    let q = as_f64(values[1]) - as_f64(values[0]) - h * (1.0 + 2.0 * a);
                    let r = -h * h;
                    l2 += h
                        * (p * p + p * q + (q * q + 2.0 * p * r) / 3.0 + q * r / 2.0 + r * r / 5.0);
                    h1 += (q * q + 2.0 * q * r + 4.0 * r * r / 3.0) / h;
                }
                let errors = (l2.sqrt(), h1.sqrt());
                if let Some(old) = previous {
                    assert!(
                        old.0 / errors.0 > 3.1,
                        "{} robin={robin} nonuniform={nonuniform} n={n}: L2 {old:?} -> {errors:?}",
                        S::NAME
                    );
                    assert!(old.1 / errors.1 > 1.8);
                }
                if n == 32 {
                    assert!(errors.0 < 0.002);
                    assert!(errors.1 < 0.04);
                }
                previous = Some(errors);
            }
        }
    }
}

macro_rules! cases {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            moments::<$scalar>();
            higher_order::<$scalar>();
            domains::<$scalar>();
            refinement::<$scalar>();
        }
    };
}
cases!(forms_f32, f32);
cases!(forms_f64, f64);

#[derive(Clone)]
struct Coefficients {
    diffusion: f64,
    reaction: f64,
}
impl EllipticPde<f64> for Coefficients {
    fn diffusion(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.diffusion)
    }
    fn reaction(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.reaction)
    }
    fn source(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
}
#[test]
fn elliptic_zero_diffusion_fails_even_with_reaction_or_no_free_dofs() {
    for nodes in [2, 7] {
        let mesh = uniform(0.0, 1.0, nodes);
        let zero = Coefficients {
            diffusion: 0.0,
            reaction: 1.0,
        };
        assert!(is_outside_domain(&solve_steady(
            &mesh,
            &zero,
            &support::homogeneous_dirichlet(),
            EllipticOptions::new(),
            None,
        )));
    }
}

#[derive(Clone)]
struct BoundaryDegeneracy;
impl EllipticPde<f64> for BoundaryDegeneracy {
    fn diffusion(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * (1.0 - x[0]))
    }
    fn reaction(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
}
#[test]
fn conormal_data_admits_diffusion_vanishing_at_the_boundary() {
    // Natural and Robin data prescribe the conormal flux k du/dn itself, so a
    // diffusion vanishing at an endpoint needs no derivative coefficient.
    let mesh = explicit(vec![0.0, 0.4, 1.0]);
    for left in [false, true] {
        for robin in [false, true] {
            let condition = || {
                if robin {
                    End::robin(1.0, zero())
                } else {
                    End::natural(zero())
                }
            };
            let fixed = || End::dirichlet(zero());
            let ends = if left {
                Ends::new(condition(), fixed())
            } else {
                Ends::new(fixed(), condition())
            };
            let (values, report) = solve_steady(
                &mesh,
                &BoundaryDegeneracy,
                &ends,
                EllipticOptions::new(),
                None,
            )
            .unwrap();
            assert!(report.is_success());
            assert!(values.iter().all(|&u| u == 0.0));
        }
    }
}

#[test]
fn every_parabolic_scheme_uses_the_selected_forms_and_time_boundary_lifting() {
    let nodes = [0.0, 0.2, 0.55, 1.0];
    let coefficients = Pointwise(|x: f64| Ok((1.0 + x.powi(4), 0.0)));
    let dirichlet = || End::dirichlet(support::moving(|x, t| Ok(t * (1.0 + x))));
    let plan = plan(
        &nodes,
        &coefficients,
        &Ends::new(dirichlet(), dirichlet()),
        Some(gauss(4, 1)),
    )
    .unwrap();
    let source = FieldData::time_dependent(|p: FieldPoint<f64, 1>, t| Ok(p.point[0].powi(4) + t));
    let initial = FieldData::zero();
    let solve = |options: ParabolicOptions<f64>, end: f64| -> PdeSolution<f64> {
        ParabolicProblem::new(&plan, &source, end)
            .unwrap()
            .solve_with_options(&initial, &(options))
            .unwrap()
    };
    let final_time = 0.02;
    let mut solutions = Vec::new();
    for scheme in [
        ParabolicScheme::ExplicitDopri5,
        ParabolicScheme::BackwardEuler,
        ParabolicScheme::CrankNicolson,
    ] {
        let step = 0.00001;
        let result = solve(
            ParabolicOptions::new().with_scheme(scheme).with_step(step),
            final_time,
        );
        assert!(result.report().is_success());
        let values = result.values().to_vec();
        assert!((values[0] - final_time).abs() < 1e-14);
        assert!((values[3] - 2.0 * final_time).abs() < 1e-14);
        solutions.push(values);
    }
    for values in &solutions[1..] {
        for (&actual, &expected) in values.iter().zip(&solutions[0]) {
            assert!(
                (actual - expected).abs() < 2e-6,
                "{actual} versus {expected}"
            );
        }
    }
    // Independent semidiscrete initial derivative: integral x^4*phi / lumped mass.
    let result = solve(
        ParabolicOptions::new()
            .with_scheme(ParabolicScheme::ExplicitDopri5)
            .with_step(1e-8),
        1e-8,
    );
    for index in 1..3 {
        let (a, x, b) = (nodes[index - 1], nodes[index], nodes[index + 1]);
        let load = ((x.powi(6) - a.powi(6)) / 6.0 - a * (x.powi(5) - a.powi(5)) / 5.0) / (x - a)
            + (b * (b.powi(5) - x.powi(5)) / 5.0 - (b.powi(6) - x.powi(6)) / 6.0) / (b - x);
        let expected = load / ((b - a) * 0.5);
        assert!((result.values()[index] / 1e-8 - expected).abs() < 2e-6);
    }
}

#[test]
fn load_callback_errors_propagate_without_replacing_previous_values() {
    let coefficients = Pointwise(|x: f64| Ok((1.0 + x * x * x, 1.0 + x)));
    let plan = plan(&[0.0, 1.0], &coefficients, &natural(), None).unwrap();
    let previous = plan.load(&source(|x| Ok(x * x)), 0.0).unwrap();
    let mut output = previous.clone();
    let broken = source(|_: f64| {
        Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "test callback failure",
        ))
    });
    let result = plan.load(&broken, 0.0).map(|load| output = load);
    let error = result.unwrap_err();
    let SciOxideErrorView::DomainError { detail, .. } = error.view() else {
        unreachable!("unexpected error kind {error:?}")
    };
    assert_eq!(detail, "test callback failure");
    assert_eq!(output, previous);
}

#[test]
fn coefficients_are_sampled_once_when_the_plan_is_prepared() {
    // Coefficients are time-invariant: three elements times two Gauss samples,
    // and no further sampling during a time integration.
    let calls = AtomicUsize::new(0);
    let coefficients = Pointwise(|x: f64| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok((1.0 + x, 0.0))
    });
    let plan = plan(
        &[0.0, 0.2, 0.55, 1.0],
        &coefficients,
        &support::homogeneous_dirichlet(),
        None,
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 6);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.0001);
    let source = FieldData::time_dependent(|p: FieldPoint<f64, 1>, t| {
        let x = p.point[0];
        Ok(x * (1.0 - x) * (1.0 + t))
    });
    let result = ParabolicProblem::new(&plan, &source, 0.02)
        .unwrap()
        .solve_with_options(&FieldData::zero(), &(options))
        .unwrap();
    assert!(result.report().is_success());
    assert_eq!(calls.load(Ordering::Relaxed), 6);
}
